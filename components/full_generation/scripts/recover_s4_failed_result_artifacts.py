"""Recover small S4 result artifacts when a failed run has no RRCTL artifact manifest.

The formal victim exited after its evaluations but before the pipeline wrote
comparison.json, so RRCTL had no terminal artifact manifest. This bounded
fallback reads the existing failed run, downloads an explicit small-file
allowlist through the configured RRCTL transport, verifies remote size/SHA,
and derives reports locally. It never launches or mutates a remote workload.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_ID = "s4_tfss_utility_seed0_20261017_v1_victim_recovery01"
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 5_000_000
CLASS_METRICS = ("AP50", "AP75", "AP50_95", "Recall", "Precision")
VOC_METHODS = ("Clean", "S0", "S1", "S2", "S3", "S4", "S4_best_secondary")
BASELINES = ("S0", "S1", "S2", "S3")

# Exact small metadata/results allowlist. We intentionally omit model weights,
# images/labels, the 65 MB AMP CSV, and the multi-megabyte console log.
OUTPUT_FILES = (
    "RECOVERY_PROVENANCE.json",
    "S0_person_dog.json",
    "S4_best_secondary_evaluation.json",
    "S4_evaluation.json",
    "S4_person_dog.json",
    "pipeline_status.json",
    "victim_completion.json",
    "victim_run_spec.json",
    "victim_run/victim/results.csv",
)
CONTROL_FILES = ("status.json", "events.jsonl")


def _json_bytes(data: bytes, source: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remote_inventory(transport, python: str, output_root: str, control_root: str) -> dict[str, Any]:
    items = [
        {"root": control_root, "relative": name, "destination": f"rrctl_control/{name}"}
        for name in CONTROL_FILES
    ] + [
        {"root": output_root, "relative": name, "destination": f"payload/{name}"}
        for name in OUTPUT_FILES
    ]
    code = (
        "import hashlib,json,os\n"
        f"items=json.loads({json.dumps(json.dumps(items))})\n"
        "out=[]\n"
        "for item in items:\n"
        " p=os.path.join(item['root'],item['relative'])\n"
        " try:\n"
        "  h=hashlib.sha256(); size=0\n"
        "  with open(p,'rb') as f:\n"
        "   while True:\n"
        "    b=f.read(1024*1024)\n"
        "    if not b: break\n"
        "    h.update(b); size+=len(b)\n"
        "  out.append({'root':item['root'],'relative':item['relative'],'destination':item['destination'],'exists':True,'size':size,'sha256':h.hexdigest()})\n"
        " except FileNotFoundError:\n"
        "  out.append({'root':item['root'],'relative':item['relative'],'destination':item['destination'],'exists':False})\n"
        " except Exception as e:\n"
        "  out.append({'root':item['root'],'relative':item['relative'],'destination':item['destination'],'error':type(e).__name__+':'+str(e)})\n"
        "print(json.dumps({'items':out},sort_keys=True))\n"
    )
    result = transport.run([python, "-B", "-c", code])
    if result.returncode != 0:
        raise RuntimeError(
            "RRCTL remote metadata read failed: "
            + result.stderr.decode("utf-8", "replace")[-2000:]
        )
    return _json_bytes(result.stdout, "RRCTL remote inventory")


def pull_small_outputs(run_id: str = RUN_ID) -> tuple[Path, dict[str, Any]]:
    from remote_run_control.controller import Controller

    if run_id != RUN_ID:
        raise ValueError("This recovery helper is bound to the single authorized RunID")
    controller = Controller(
        profiles_path=ROOT / "execution/profiles.json",
        state_root=ROOT / "execution/state",
    )
    ref, spec, transport = controller._runtime(run_id)
    if ref.run_id != RUN_ID or spec.run_id != RUN_ID:
        raise ValueError("RRCTL binding does not match the authorized recovery RunID")

    inspect = transport.run(
        [ref.remote_python, ref.worker_path, "inspect", "--control", ref.control_root]
    )
    if inspect.returncode != 0:
        raise RuntimeError(
            "RRCTL inspect failed: " + inspect.stderr.decode("utf-8", "replace")[-2000:]
        )
    remote_state = _json_bytes(inspect.stdout, "RRCTL inspect")
    status = remote_state.get("status", {})
    if status.get("state") != "failed" or status.get("detail", {}).get("exit_code") != 1:
        raise ValueError(f"Expected the observed post-processing failure; got {status!r}")

    destination = Path(spec.local_pull_root).expanduser() / run_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {destination}")
    expected_parent = Path(spec.local_pull_root).expanduser().resolve()
    if not destination.resolve().parent == expected_parent:
        raise ValueError("Local result destination escaped the configured RRCTL pull root")

    inventory = _remote_inventory(
        transport, ref.remote_python, spec.remote.output_root, ref.control_root
    )
    items = inventory.get("items")
    if not isinstance(items, list):
        raise ValueError("Remote inventory schema is invalid")

    destination.mkdir(parents=True, exist_ok=False)
    pulled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_bytes = 0
    try:
        for item in items:
            if "error" in item:
                raise RuntimeError(f"Remote file inventory failed: {item}")
            if not item.get("exists"):
                skipped.append({"path": item["relative"], "reason": "not_present_remotely"})
                continue
            size = item["size"]
            if size > MAX_FILE_BYTES or total_bytes + size > MAX_TOTAL_BYTES:
                skipped.append({"path": item["relative"], "reason": "bounded_pull_size_limit", "size": size})
                continue
            remote_path = f"{item['root']}/{item['relative']}"
            data = transport.download(remote_path)
            digest = _sha_bytes(data)
            if len(data) != size or digest != item["sha256"]:
                raise ValueError(f"RRCTL download integrity mismatch: {item['relative']}")
            local_path = destination / item["destination"]
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("xb") as stream:
                stream.write(data)
            pulled.append(
                {
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "bytes": size,
                    "sha256": digest,
                }
            )
            total_bytes += size
    except BaseException:
        # Preserve any successfully retrieved evidence; never clean it up.
        raise

    manifest_error = None
    try:
        transport.download(f"{ref.control_root}/artifact_manifest.json")
        manifest_error = "manifest unexpectedly became available; rerun after inspecting state"
    except Exception as exc:
        manifest_error = f"{type(exc).__name__}: {exc}"

    pull_record = {
        "schema": "s4.rrctl-bounded-results-pull.v1",
        "run_id": run_id,
        "remote_terminal_state": status,
        "rrctl_inspect_return_code": inspect.returncode,
        "protocol_artifact_manifest": "missing; Controller.pull cannot build a full artifact set",
        "manifest_read_error": manifest_error,
        "method": "RRCTL ProfileStore/transport exact-path downloads with independent size/SHA verification",
        "files_pulled": pulled,
        "files_skipped": skipped,
        "bytes_pulled": total_bytes,
        "excluded": [
            "all weights/checkpoints",
            "all data/images/labels",
            "victim_run/victim/amp_audit/AMP_SCALE_AND_SKIP_LOG.csv (about 65 MB; not needed for AP comparison)",
            "control/console.log (large; traceback already inspected read-only)",
        ],
    }
    (destination / "RRCTL_BOUNDED_RESULTS_PULL.json").write_text(
        json.dumps(pull_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination, pull_record


def _finite_arrays(value: dict[str, Any], role: str, class_names: list[str]) -> None:
    if value.get("class_names") != class_names:
        raise ValueError(f"Class order mismatch in {role}")
    for metric in CLASS_METRICS:
        key = f"{metric}_per_class"
        values = value.get(key)
        if not isinstance(values, list) or len(values) != 20:
            raise ValueError(f"Missing 20-class array {key} in {role}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite per-class values in {role}.{key}")


def build_reports(destination: Path) -> dict[str, Any]:
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES, collateral, pus
    from scripts.run_s4_tfss_pipeline import comparable_metric_delta

    payload = destination / "payload"
    historical_path = ROOT / "docs/S4_TFSS_UTILITY/HISTORICAL_METRICS.json"
    historical = _json_bytes(historical_path.read_bytes(), str(historical_path))
    primary = _json_bytes((payload / "S4_evaluation.json").read_bytes(), "S4_evaluation.json")
    secondary = _json_bytes(
        (payload / "S4_best_secondary_evaluation.json").read_bytes(),
        "S4_best_secondary_evaluation.json",
    )
    completion = _json_bytes((payload / "victim_completion.json").read_bytes(), "victim_completion.json")
    provenance = _json_bytes((payload / "RECOVERY_PROVENANCE.json").read_bytes(), "RECOVERY_PROVENANCE.json")
    pipeline_status = _json_bytes((payload / "pipeline_status.json").read_bytes(), "pipeline_status.json")
    spec_value = _json_bytes((payload / "victim_run_spec.json").read_bytes(), "victim_run_spec.json")
    class_names = list(NAMES)
    if completion.get("epochs") != 200 or completion.get("rows") != 200:
        raise ValueError("victim_completion does not verify 200 epochs/rows")
    if primary.get("checkpoint_role") != "last.pt@200" or secondary.get("checkpoint_role") != "best.pt":
        raise ValueError("Primary/secondary checkpoint roles are not the registered ones")
    if primary.get("checkpoint_sha256") != completion.get("last_sha256"):
        raise ValueError("last.pt checkpoint SHA does not match evaluation receipt")
    if secondary.get("checkpoint_sha256") != completion.get("best_sha256"):
        raise ValueError("best.pt checkpoint SHA does not match evaluation receipt")
    _finite_arrays(primary, "S4 last", class_names)
    _finite_arrays(secondary, "S4 best", class_names)
    for role in BASELINES:
        _finite_arrays(historical[role], role, class_names)
    _finite_arrays(historical["Clean"], "Clean", class_names)

    results_path = payload / "victim_run/victim/results.csv"
    with results_path.open(newline="", encoding="utf-8") as stream:
        results = list(csv.DictReader(stream))
    if len(results) != 200 or [int(float(row["epoch"])) for row in results] != list(range(1, 201)):
        raise ValueError("victim results.csv must contain exactly epochs 1..200")
    loss_fields = [name for name in results[0] if "loss" in name.lower()]
    for row in results:
        if not all(math.isfinite(float(row[name])) for name in loss_fields):
            raise ValueError("Non-finite recorded loss in victim results.csv")

    audit = provenance.get("admission", {}).get("audit", {})
    expected_frozen = "33d3d7af6c76c2c881db0dfc7d6fc66dafa9b5655026c07332701143ed52d6ec"
    if provenance.get("frozen_s4_config_sha256") != expected_frozen:
        raise ValueError("Frozen S4 config identity differs from the authorized lock")
    if audit.get("state") != "passed" or not audit.get("read_only_input"):
        raise ValueError("Existing poisoned dataset audit did not pass as a read-only input")

    metrics = json.loads(json.dumps(historical))
    metrics["S4"] = primary
    metrics["S4_best_secondary"] = secondary
    for value in metrics.values():
        value["selectivity"] = pus(metrics["Clean"], value)
        value["collateral"] = collateral(metrics["Clean"], value)

    candidate = metrics["S4"]
    deltas: dict[str, dict[str, float]] = {}
    unavailable: dict[str, list[str]] = {}
    for role in BASELINES:
        deltas[role], unavailable[role] = comparable_metric_delta(candidate, metrics[role])

    direct_s0 = _json_bytes((payload / "S0_person_dog.json").read_bytes(), "S0_person_dog.json")
    direct_s4 = _json_bytes((payload / "S4_person_dog.json").read_bytes(), "S4_person_dog.json")
    comparison = {
        "schema": "s4.voc20-recovered-comparison.v1",
        "status": "locally_reconstructed_from_hash-verified_existing_outputs",
        "original_rrctl_run_state": pipeline_status.get("state"),
        "original_pipeline_error": pipeline_status.get("error"),
        "method": "S4-TFSS-Utility",
        "primary": "last.pt@200",
        "secondary": "best.pt (secondary only)",
        "single_seed_not_significance": True,
        "causal_mechanism_established": False,
        "class_names": class_names,
        "metrics": metrics,
        "S4_primary_minus_baselines": deltas,
        "aggregate_metrics_unavailable_in_historical_schema": unavailable,
        "person_dog": {"S0": direct_s0, "S4": direct_s4},
        "person_dog_definition": "fixed-threshold GT-overlap diagnostic; separate from native AP and not proof of class migration",
        "source_provenance": {
            "formal_run_id": RUN_ID,
            "parent_noise_run_id": provenance.get("parent_noise_run_id"),
            "repair_commit_sha": provenance.get("repair_commit_sha"),
            "frozen_s4_config_sha256": provenance.get("frozen_s4_config_sha256"),
            "noise_checkpoint_sha256": provenance.get("parent_noise_checkpoint_sha256"),
            "dataset_manifest_sha256": provenance.get("parent_materialization_manifest_sha256"),
            "dataset_integrity_sha256": provenance.get("parent_dataset_integrity_sha256"),
            "last_sha256": completion.get("last_sha256"),
            "best_sha256": completion.get("best_sha256"),
        },
        "training": {
            "epochs": completion.get("epochs"),
            "results_rows": len(results),
            "args": spec_value.get("train_args", {}),
            "amp_failure_repair": "native optimizer_step semantics preserved; native scaler skip/backoff observed",
        },
        "dataset_audit": audit,
        "interpretation_limits": [
            "single seed; no statistical significance claim",
            "AP changes do not establish causal background absorption or Person-to-Dog migration",
            "historical aggregate Recall_all and Precision_all are absent; those aggregate deltas are intentionally unavailable, while per-class Recall/Precision deltas are reported",
            "RRCTL run remains terminal failed because report assembly raised KeyError('Recall_all'); this local reconstruction does not rewrite remote terminal status",
        ],
    }
    out = destination / "reports"
    out.mkdir(exist_ok=False)
    (out / "S4_COMPARISON.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    aggregate_fields = (
        "mAP50_all", "mAP50_target", "mAP50_non_target",
        "mAP75_all", "mAP75_target", "mAP75_non_target",
        "mAP50_95_all", "mAP50_95_target", "mAP50_95_non_target",
        "Recall_all", "Recall_target", "Recall_non_target",
        "Precision_all", "Precision_target", "Precision_non_target",
    )
    summary_roles = ("Clean", "S0", "S1", "S2", "S3", "S4", "S4_best_secondary")
    summary_path = out / "S4_S0_S1_S2_S3_AGGREGATE_COMPARISON.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["Method", "checkpoint_role", *aggregate_fields, "P", "U", "S", "WCD", "WCD_class", "TCD3"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role in summary_roles:
            value = metrics[role]
            row: dict[str, Any] = {"Method": role, "checkpoint_role": value.get("checkpoint_role", "reference")}
            row.update({key: value.get(key) for key in aggregate_fields})
            row.update({key: value.get("selectivity", {}).get(key) for key in ("P", "U", "S")})
            row.update({"WCD": value.get("collateral", {}).get("WCD"),
                        "WCD_class": value.get("collateral", {}).get("WCD_class"),
                        "TCD3": value.get("collateral", {}).get("TCD3")})
            writer.writerow(row)

    per_class_path = out / "S4_VOC20_PER_CLASS_COMPARISON.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["class"]
        for role in VOC_METHODS:
            fields.extend(f"{role}_{metric}" for metric in CLASS_METRICS)
        for baseline in BASELINES:
            fields.extend(f"S4_minus_{baseline}_{metric}" for metric in CLASS_METRICS)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            row: dict[str, Any] = {"class": class_name}
            for role in VOC_METHODS:
                for metric in CLASS_METRICS:
                    row[f"{role}_{metric}"] = metrics[role][f"{metric}_per_class"][index]
            for baseline in BASELINES:
                for metric in CLASS_METRICS:
                    row[f"S4_minus_{baseline}_{metric}"] = (
                        metrics["S4"][f"{metric}_per_class"][index]
                        - metrics[baseline][f"{metric}_per_class"][index]
                    )
            writer.writerow(row)

    delta_path = out / "S4_AGGREGATE_DELTAS_VS_S0_S1_S2_S3.csv"
    with delta_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["baseline", "metric", "S4_last", "baseline_value", "S4_minus_baseline", "availability"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for baseline in BASELINES:
            for key in aggregate_fields:
                current, old = candidate.get(key), metrics[baseline].get(key)
                available = (
                    isinstance(current, (int, float)) and math.isfinite(current)
                    and isinstance(old, (int, float)) and math.isfinite(old)
                )
                writer.writerow({
                    "baseline": baseline, "metric": key,
                    "S4_last": current, "baseline_value": old,
                    "S4_minus_baseline": current - old if available else None,
                    "availability": "available" if available else "not reported by historical aggregate schema",
                })
            for key in ("P", "U", "S"):
                current = metrics["S4"]["selectivity"][key]
                old = metrics[baseline]["selectivity"][key]
                writer.writerow({"baseline": baseline, "metric": key, "S4_last": current,
                                 "baseline_value": old, "S4_minus_baseline": current-old,
                                 "availability": "available"})
            for key in ("WCD", "TCD3"):
                current = metrics["S4"]["collateral"][key]
                old = metrics[baseline]["collateral"][key]
                writer.writerow({"baseline": baseline, "metric": key, "S4_last": current,
                                 "baseline_value": old, "S4_minus_baseline": current-old,
                                 "availability": "available"})

    p = metrics["S4"]["selectivity"]
    cd = metrics["S4"]["collateral"]
    last_row = results[-1]
    people = audit.get("changed_Person_GT")
    nt = audit.get("non_target_images_byte_identical")
    failed_at = pipeline_status.get("failed_at_unix")
    failed_at_hk = (
        dt.datetime.fromtimestamp(failed_at, dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()
        if isinstance(failed_at, (int, float)) else "unknown"
    )
    lines = [
        "# S4-TFSS-Utility 与 S0/S1/S2/S3 完整结果比较",
        "",
        f"- 正式恢复 RunID：`{RUN_ID}`。",
        "- 主评价：随机初始化 Victim `last.pt@200`；`best.pt` 仅作为次要结果。",
        "- 训练：200 epoch，seed 0，batch 36，workers 16，AMP 开启；原 S4 数据集复用，没有重生成噪声。",
        f"- 数据审计：{audit.get('train')} train / {audit.get('val')} clean-val，{audit.get('target_images')} 个目标图像，{people} 个 Person GT 改写；标签与 clean-val 字节一致，{nt} 个 non-target 图像字节不变，扰动预算为全图 8/255、重叠区 4/255、框外 0。",
        f"- 训练记录：{len(results)} 行连续 epoch 结果；最后一行 epoch={last_row.get('epoch')}，last/best SHA 分别为 `{completion.get('last_sha256')}` / `{completion.get('best_sha256')}`。",
        f"- RRCTL 原始终态：`{pipeline_status.get('state')}`，exit code 1，失败时间（香港时间）`{failed_at_hk}`；失败发生在指标汇总，异常为 `{pipeline_status.get('error')}`。训练与 last/best 评价文件已保存。",
        "- 拉取修复：远端未生成 RRCTL `artifact_manifest.json`，因为必需的比较文件在字段异常后未写出。已通过 RRCTL transport 按固定小文件清单逐个读取并核对远端 SHA；未下载权重、数据、AMP 大 CSV 或大日志。",
        "",
        "## 主结果",
        "",
        "AP 数值以 0–1 表示；R/P 是 validator 全局最佳 F1 操作点，不等同 AP 的完整置信度曲线。P/U/S 按清洁 S0 参考计算，WCD/TCD3 为对 clean 的有符号 non-target AP50 相对损伤；负值表示相对提升。",
        "",
        "|方法|Person AP50|NT mAP50|NT mAP75|NT mAP50–95|All mAP50|R target|R NT|P target|P NT|P|U|S|WCD 类别|WCD|TCD3|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for role in ("S0", "S1", "S2", "S3", "S4"):
        value = metrics[role]
        sel = value["selectivity"]
        coll_value = value["collateral"]
        row = [role, value["mAP50_target"], value["mAP50_non_target"], value["mAP75_non_target"],
               value["mAP50_95_non_target"], value["mAP50_all"], value.get("Recall_target"),
               value.get("Recall_non_target"), value.get("Precision_target"), value.get("Precision_non_target"),
               sel["P"], sel["U"], sel["S"], coll_value["WCD_class"], coll_value["WCD"], coll_value["TCD3"]]
        lines.append("|" + "|".join("—" if item is None else (f"{item:.6f}" if isinstance(item, (int, float)) else str(item)) for item in row) + "|")

    lines += ["", "## S4 last@200 相对各方法的差值", "",
              "差值为 S4 minus baseline；AP/R/P 以 0–1 原始尺度表示。S4 的目标 AP50 约束与 non-target 指标按冻结配置解释，不把单 seed 结果当显著性证据。", "",
              "|基线|Person AP50 Δ|NT AP50 Δ|NT AP75 Δ|NT AP50–95 Δ|All AP50 Δ|P Δ|U Δ|S Δ|WCD Δ|TCD3 Δ|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for baseline in BASELINES:
        old = metrics[baseline]
        lines.append("|" + "|".join([
            baseline,
            f"{candidate['mAP50_target']-old['mAP50_target']:+.6f}",
            f"{candidate['mAP50_non_target']-old['mAP50_non_target']:+.6f}",
            f"{candidate['mAP75_non_target']-old['mAP75_non_target']:+.6f}",
            f"{candidate['mAP50_95_non_target']-old['mAP50_95_non_target']:+.6f}",
            f"{candidate['mAP50_all']-old['mAP50_all']:+.6f}",
            f"{candidate['selectivity']['P']-old['selectivity']['P']:+.6f}",
            f"{candidate['selectivity']['U']-old['selectivity']['U']:+.6f}",
            f"{candidate['selectivity']['S']-old['selectivity']['S']:+.6f}",
            f"{candidate['collateral']['WCD']-old['collateral']['WCD']:+.6f}",
            f"{candidate['collateral']['TCD3']-old['collateral']['TCD3']:+.6f}",
        ]) + "|")
    lines += [
        "",
        "## 20 类逐类结果",
        "",
        "完整 AP50/AP75/AP50–95/Recall/Precision 原值及 S4 对 S0/S1/S2/S3 的逐类差值见 `S4_VOC20_PER_CLASS_COMPARISON.csv`。未从 AP 推算 R/P。历史 S0–S3 归档未提供 aggregate `Recall_all` / `Precision_all`，所以这些 aggregate 差值留空；逐类 R/P 数组仍完整可比。",
        "",
        "## 完整性与限制",
        "",
        "S4 的 last 和 best 评价均来自保存的 checkpoint SHA，并与 `victim_completion.json` 一致。已有 200 epoch CSV 与全量 4,952 张 clean-val 评价支持训练及验证完成。报告由 hash-verified 小型 RRCTL 输出与本地锁定的 S0–S3 历史 metrics 重建；原远端 pipeline terminal state 仍保留为 failed，不伪装成 RRCTL 完整接受。",
        "",
        "本结果是一个 seed 的描述性比较，不构成统计显著性或因果机制证明。Person AP 变化不能单独证明背景吸收或类别迁移。S0–S3 基线使用已归档的相同 VOC20 指标定义与类顺序；历史 aggregate R/P 缺项按不可用处理。",
        "",
        f"证据：`S4_COMPARISON.json`、`S4_S0_S1_S2_S3_AGGREGATE_COMPARISON.csv`、`S4_AGGREGATE_DELTAS_VS_S0_S1_S2_S3.csv`、`S4_VOC20_PER_CLASS_COMPARISON.csv`。",
    ]
    report_path = out / "S4_S0_S1_S2_S3_FINAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "report": str(report_path),
        "aggregate_csv": str(summary_path),
        "per_class_csv": str(per_class_path),
        "deltas_csv": str(delta_path),
        "primary_person_ap50": candidate["mAP50_target"],
        "primary_non_target_ap50": candidate["mAP50_non_target"],
        "primary_PUS": candidate["selectivity"],
        "primary_collateral": candidate["collateral"],
        "deltas": deltas,
        "unavailable_historical_aggregates": unavailable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    destination, pull_record = pull_small_outputs(args.run_id)
    report = build_reports(destination)
    summary = {
        "run_id": args.run_id,
        "output_directory": str(destination),
        "pull": pull_record,
        "report": report,
    }
    (destination / "FINALIZATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
