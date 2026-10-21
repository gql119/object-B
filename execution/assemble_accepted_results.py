"""Build sanitized S4 robustness tables and reports from accepted pair evidence.

This is a local documentation step. It never launches or changes an experiment.
The accepted-pairs JSON contains only metrics, hashes, protocol fields, and a
sanitized server-status snapshot; raw server paths are intentionally excluded.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S4_ROBUSTNESS_20261019"
PAIRS_PATH = DOC / "S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json"
STATUS_PATH = ROOT / "execution" / "PIPELINE_STATUS_LATEST.json"
LOCK_PATH = DOC / "S4_ROBUSTNESS_LOCK.json"
SOURCE = "e17c24c54aae4cdae4bcc4703facf2a57052d2c8"
CONFIG = "00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91"
PARENT = "194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36"
PIPELINE = "s4_robustness_server_pipeline_seed0_20261020_recovery01"
DEFENSE_ORDER = ["jpeg50", "jpeg30", "gray", "bdr", "gauss", "meanf", "medianf", "data_aug"]
PHASE_A = ["jpeg50", "jpeg30", "gray", "bdr"]
PHASE_B = ["gauss", "meanf", "medianf", "data_aug"]
PHASE_BY_DEFENSE = {name: "A" for name in PHASE_A} | {name: "B" for name in PHASE_B}
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
RESULT_FIELDS = [
    "defense", "arm", "status", "run_id", "checkpoint_sha256", "seed", "primary",
    "source_commit", "config_sha256", "parent_S4_manifest_sha256", "defense_parameters",
    "mAP50_target", "mAP50_non_target", "mAP75_target", "mAP50_95_target",
    "Recall_target", "Precision_target", "mAP75_non_target", "mAP50_95_non_target",
    "Recall_non_target", "Precision_non_target", "mAP50_all", "mAP75_all",
    "mAP50_95_all", "Recall_all", "Precision_all", "P", "U", "S", "WCD", "TCD3",
]
PER_CLASS_FIELDS = ["defense", "arm", "status", "class_id", "class_name", "AP50", "AP75", "AP50_95", "Recall", "Precision"]
EXTENDED_FIELDS = ["defense", "arm", "status", "metric", "value", "untransformed_baseline", "defense_minus_baseline"]
METRIC_KEYS = [
    "mAP50_target", "mAP50_non_target", "mAP50_all",
    "mAP75_target", "mAP75_non_target", "mAP75_all",
    "mAP50_95_target", "mAP50_95_non_target", "mAP50_95_all",
    "Recall_target", "Recall_non_target", "Recall_all",
    "Precision_target", "Precision_non_target", "Precision_all",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    if value in (None, "", "NOT_RUN", "IN_PROGRESS", "PENDING"):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def fmt(value: Any, places: int = 9) -> str:
    converted = number(value)
    return "—" if converted is None else f"{converted:.{places}f}"


def pending_run_id(defense: str) -> str:
    return f"s4_robustness_{defense}_paired_seed0_20261019_v1"


def pair_status(defense: str, snapshot: dict[str, Any]) -> tuple[str, dict[str, str]]:
    pairs = snapshot["accepted_pairs"]
    if defense in pairs:
        return "COMPLETE_ACCEPTED", {"clean": "COMPLETE_ACCEPTED", "S4": "COMPLETE_ACCEPTED"}
    if defense == snapshot.get("pipeline_snapshot", {}).get("stage"):
        active = snapshot.get("pipeline_snapshot", {}).get("active_job", "")
        arm = "S4" if ":S4:" in active else "clean" if ":clean:" in active else None
        arms = {"clean": "PENDING", "S4": "PENDING"}
        if arm:
            arms[arm] = "IN_PROGRESS"
        return "IN_PROGRESS", arms
    return "NOT_RUN", {"clean": "NOT_RUN", "S4": "NOT_RUN"}


def accepted_row(defense: str, arm: str, pair: dict[str, Any]) -> dict[str, Any]:
    metrics = pair["metrics"][arm]
    result: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
    result.update(
        defense=defense,
        arm=arm,
        status="COMPLETE_ACCEPTED",
        run_id=pair["run_id"],
        checkpoint_sha256=metrics["checkpoint_sha256"],
        seed=0,
        primary=pair["primary"],
        source_commit=pair["source_commit"],
        config_sha256=pair["config_sha256"],
        parent_S4_manifest_sha256=pair["parent_S4_manifest_sha256"],
        defense_parameters=json.dumps(pair["defense_parameters"], ensure_ascii=False, sort_keys=True),
    )
    for key in METRIC_KEYS:
        result[key] = metrics.get(key, "")
    if arm == "S4":
        for key in ("P", "U", "S", "WCD", "TCD3"):
            result[key] = pair["scores"].get(key, "")
    return result


def blank_row(defense: str, arm: str, status: str, parameters: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
    result.update(
        defense=defense,
        arm=arm,
        status=status,
        run_id=run_id or pending_run_id(defense),
        seed=0,
        primary="last.pt@200",
        source_commit=SOURCE if status != "NOT_AUTHORIZED" else "",
        config_sha256=CONFIG,
        parent_S4_manifest_sha256=PARENT,
        defense_parameters=json.dumps(parameters, ensure_ascii=False, sort_keys=True),
    )
    return result


def class_rows(defense: str, arm: str, pair: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = pair["metrics"][arm]
    arrays = {
        "AP50": metrics["AP50_per_class"],
        "AP75": metrics["AP75_per_class"],
        "AP50_95": metrics["AP50_95_per_class"],
        "Recall": metrics["Recall_per_class"],
        "Precision": metrics["Precision_per_class"],
    }
    names = metrics.get("class_names", CLASS_NAMES)
    assert len(names) == 20 and all(len(values) == 20 for values in arrays.values())
    return [
        {
            "defense": defense,
            "arm": arm,
            "status": "COMPLETE_ACCEPTED",
            "class_id": idx,
            "class_name": names[idx],
            **{key: arrays[key][idx] for key in arrays},
        }
        for idx in range(20)
    ]


def summary_rows(pairs: dict[str, Any], defenses: list[str], pipeline: dict[str, Any], lock_defenses: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for defense in defenses:
        if defense in pairs:
            for arm in ("clean", "S4"):
                rows.append(accepted_row(defense, arm, pairs[defense]))
        else:
            _, arm_states = pair_status(defense, {"accepted_pairs": pairs, "pipeline_snapshot": pipeline})
            for arm in ("clean", "S4"):
                status = arm_states[arm]
                rows.append(blank_row(defense, arm, status, lock_defenses[defense]))
    return rows


def baseline_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = [row for row in rows if row.get("defense") == "none"]
    assert {row["arm"] for row in baseline} == {"clean", "S4"}
    return baseline


def main_table(pairs: dict[str, Any], baseline: list[dict[str, str]], pipeline: dict[str, Any]) -> str:
    clean_base = next(row for row in baseline if row["arm"] == "clean")
    s4_base = next(row for row in baseline if row["arm"] == "S4")
    columns = ["Clean", "Gauss", "JPEG50", "JPEG30", "MeanF", "MedianF", "Gray", "BDR", "AT", "Data Augmentation"]
    mapping = {"Gauss": "gauss", "JPEG50": "jpeg50", "JPEG30": "jpeg30", "MeanF": "meanf", "MedianF": "medianf", "Gray": "gray", "BDR": "bdr", "Data Augmentation": "data_aug"}
    metrics = [
        ("clean Person AP50", "mAP50_target", "clean"),
        ("clean Non-target mAP50", "mAP50_non_target", "clean"),
        ("clean Selectivity S", "S", "clean"),
        ("S4 Person AP50", "mAP50_target", "S4"),
        ("S4 Non-target mAP50", "mAP50_non_target", "S4"),
        ("S4 Selectivity S", "S", "S4"),
    ]
    rows = []
    for label, metric, arm in metrics:
        values = []
        for col in columns:
            if col == "Clean":
                base = clean_base if arm == "clean" else s4_base
                values.append("—" if metric == "S" else fmt(base.get(metric)))
            elif col == "AT":
                values.append("—" if metric == "S" and arm == "clean" else "NOT_AUTHORIZED")
            else:
                defense = mapping[col]
                pair = pairs.get(defense)
                if pair:
                    if metric == "S":
                        values.append("—" if arm == "clean" else fmt(pair["scores"].get("S")))
                    else:
                        values.append(fmt(pair["metrics"][arm].get(metric)))
                else:
                    _, arm_states = pair_status(defense, {"accepted_pairs": pairs, "pipeline_snapshot": pipeline})
                    status = arm_states[arm]
                    values.append(status)
        rows.append((label, values))
    text = [
        "# S4 robustness main table",
        "",
        "Snapshot values are fractions from accepted paired full evaluations; `last.pt@200` is primary. Clean-arm S is not applicable. `IN_PROGRESS` and `NOT_RUN` are statuses, never numeric zeros. AT has no formal run and remains unauthorized.",
        "",
        "| Arm / Metric | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for label, values in rows:
        text.append("| " + label + " | " + " | ".join(values) + " |")
    text += [
        "",
        f"Current server snapshot: `{pipeline.get('stage', 'unknown')}` / `{pipeline.get('state', 'unknown')}` at {pipeline.get('observed_utc', 'unknown')}; see `S4_ROBUSTNESS_EXECUTION_LEDGER_20261021.csv`.",
        "",
    ]
    return "\n".join(text)


def paired_table(pairs: dict[str, Any], defenses: list[str]) -> str:
    columns = ["Defense", "Clean Person AP50", "S4 Person AP50", "Clean NT mAP50", "S4 NT mAP50", "Person Recall Clean/S4", "P", "U", "S", "WCD", "TCD3"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---:|" * len(columns)]
    for defense in defenses:
        pair = pairs.get(defense)
        if pair:
            c, s = pair["metrics"]["clean"], pair["metrics"]["S4"]
            values = [
                defense,
                fmt(c["mAP50_target"]), fmt(s["mAP50_target"]),
                fmt(c["mAP50_non_target"]), fmt(s["mAP50_non_target"]),
                f"{fmt(c['Recall_target'])} / {fmt(s['Recall_target'])}",
                fmt(pair["scores"].get("P")), fmt(pair["scores"].get("U")), fmt(pair["scores"].get("S")),
                fmt(pair["scores"].get("WCD")), fmt(pair["scores"].get("TCD3")),
            ]
        else:
            values = [defense] + ["IN_PROGRESS" if defense == "meanf" else "NOT_RUN"] * (len(columns) - 1)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def phase_a_report(pairs: dict[str, Any], pipeline: dict[str, Any]) -> str:
    lines = [
        "# Phase A report — JPEG50, JPEG30, Gray, and BDR",
        "",
        "Status: `COMPLETE_ACCEPTED` for all four paired defenses. Each Clean/S4 condition has a full 200-epoch evaluation on the unchanged 4,952-image clean VOC validation set; `last.pt@200` is primary and `best.pt` secondary. All values below are fractions and descriptive for seed 0.",
        "",
        "The BDR protocol is the sourced 5-bit, 31-level transform in the frozen protocol lock. Gray uses the locked RGB→gray→three-channel RGB conversion. Transforms affect training data only.",
        "",
        paired_table(pairs, PHASE_A),
        "",
        "JPEG30 is the weakest target-protection result among these four measured defenses: its Person AP50 rises to 0.216120307 and paired P is 0.747344858, versus 0.072199285 / 0.916616354 for JPEG50. Gray and BDR retain lower Person AP50 (0.015809466 and 0.028381203) with paired P of 0.981416571 and 0.967320500. These are single-seed observations; they do not establish significance or mechanism.",
        "",
        "Non-target utility remains measurable for all four pairs (U ranges 0.889623–0.907308). Absolute Clean and S4 metrics, all five per-class measures for all 20 VOC classes, artifact hashes, and paired scores are in `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`, `PHASE_A_ROBUSTNESS_RESULTS.csv`, and `PHASE_A_PER_CLASS_RESULTS.csv`.",
        "",
        "The untransformed S4 reference is Person AP50 0.028158895 and non-target mAP50 0.693787044. Per-defense comparisons must still use the matching Clean condition; these baseline values are not substituted as the P/U denominators.",
        "",
        f"The server-owned Phase B queue was `{pipeline.get('stage', 'unknown')}` at the latest snapshot ({pipeline.get('observed_utc', 'unknown')}); this does not change the accepted Phase A results.",
        "",
        "No formal AT run is included or authorized.",
        "",
    ]
    return "\n".join(lines)


def phase_b_report(pairs: dict[str, Any], pipeline: dict[str, Any]) -> str:
    training = pipeline.get("training", {})
    amp = pipeline.get("AMP", {})
    lines = [
        "# Phase B report — Gaussian, Mean, Median, and Data Augmentation",
        "",
        "Status: `IN_PROGRESS`. The Gauss Clean/S4 pair is complete and accepted. MeanF S4 is the active server job; its Clean historical-control evaluation is pending. MedianF and Data Augmentation have not started. The server pipeline remains the sole stage scheduler.",
        "",
        paired_table(pairs, PHASE_B),
        "",
    ]
    if "gauss" in pairs:
        pair = pairs["gauss"]
        c, s = pair["metrics"]["clean"], pair["metrics"]["S4"]
        lines += [
            f"Gauss uses the audited 3×3 OpenCV Gaussian filter (`sigma=0`, default REFLECT_101 border), not Gaussian noise. Its paired values are P={pair['scores']['P']:.9f}, U={pair['scores']['U']:.9f}, S={pair['scores']['S']:.9f}; Person AP50 is {c['mAP50_target']:.9f} Clean versus {s['mAP50_target']:.9f} S4, and non-target mAP50 is {c['mAP50_non_target']:.9f} versus {s['mAP50_non_target']:.9f}.",
            "",
            "The Clean model is a provenance-checked historical 200-epoch `gaussian3_clean` checkpoint, reevaluated on the fixed clean validation set; it was not retrained for this pair. Its historical checkpoint SHA is retained in the local matching audit; the accepted paired-results file binds the checkpoint and evaluation hashes.",
            "",
        ]
    lines += [
        f"At the {pipeline.get('observed_utc', 'unknown')} RRCTL snapshot, MeanF was `{pipeline.get('state', 'unknown')}` at {training.get('epoch', 'unknown')} completed checkpoint epoch; the live AMP attempt is epoch {amp.get('epoch', 'unknown')}, batch {amp.get('batch_index', 'unknown')}/{amp.get('total_batches', 'unknown')}. It recorded {amp.get('actual_updates_total', 'unknown')} actual updates and {amp.get('skips_total', 'unknown')} legal AMP skips, with {amp.get('consecutive_skips', 'unknown')} consecutive skips and finite model/EMA/optimizer state. These are execution diagnostics, not evaluation results.",
        "",
        "MeanF metrics remain `IN_PROGRESS` until the server reports both arms accepted and the full paired-result audit passes. MedianF and Data Augmentation remain `NOT_RUN`; AT remains excluded.",
        "",
    ]
    return "\n".join(lines)


def final_report(pairs: dict[str, Any], pipeline: dict[str, Any]) -> str:
    rows = {name: pairs[name] for name in pairs}
    lines = [
        "# S4 robustness results — interim, Phase B still running",
        "",
        f"Status: Phase A is complete and accepted; Gauss is accepted in Phase B; MeanF is in progress; MedianF and Data Augmentation have not started. Latest read-only RRCTL snapshot: {pipeline.get('observed_utc', 'unknown')}, pipeline `{PIPELINE}`, stage `{pipeline.get('stage', 'unknown')}`, state `{pipeline.get('state', 'unknown')}`. This is not final Phase A/B acceptance.",
        "",
        "The scientific source, S4 data identity, and training configuration remain frozen. This report describes one seed and does not provide significance tests or causal evidence. It does not use S0/S3/S5 poisoned data and does not include AT training.",
        "",
        "## Accepted paired results",
        "",
        paired_table(rows, ["jpeg50", "jpeg30", "gray", "bdr", "gauss", "meanf", "medianf", "data_aug"]),
        "",
        "## Research questions answered from currently accepted evidence",
        "",
        f"1. JPEG50: paired P/U/S={rows['jpeg50']['scores']['P']:.6f}/{rows['jpeg50']['scores']['U']:.6f}/{rows['jpeg50']['scores']['S']:.6f}; S4 Person AP50 is {rows['jpeg50']['metrics']['S4']['mAP50_target']:.6f} versus Clean {rows['jpeg50']['metrics']['clean']['mAP50_target']:.6f}.",
        f"2. JPEG30 is weaker than JPEG50 on target protection in this seed: P={rows['jpeg30']['scores']['P']:.6f} versus {rows['jpeg50']['scores']['P']:.6f}, while S4 Person AP50 is {rows['jpeg30']['metrics']['S4']['mAP50_target']:.6f} versus {rows['jpeg50']['metrics']['S4']['mAP50_target']:.6f}.",
        f"3. Gray retains low target AP in the paired comparison (Clean/S4 Person AP50 {rows['gray']['metrics']['clean']['mAP50_target']:.6f}/{rows['gray']['metrics']['S4']['mAP50_target']:.6f}; P={rows['gray']['scores']['P']:.6f}). This alone does not prove the background-misdirection mechanism.",
        f"4. BDR retains low target AP in this run (Clean/S4 {rows['bdr']['metrics']['clean']['mAP50_target']:.6f}/{rows['bdr']['metrics']['S4']['mAP50_target']:.6f}; P={rows['bdr']['scores']['P']:.6f}).",
        f"5. Gauss is the only accepted Phase B filter so far (P={rows['gauss']['scores']['P']:.6f}, U={rows['gauss']['scores']['U']:.6f}, S={rows['gauss']['scores']['S']:.6f}); it cannot yet be ranked against MeanF or MedianF.",
        "6. Data Augmentation has not run; no result is available.",
        "7. On absolute non-target mAP50, each accepted S4 arm is below its same-defense Clean arm. The paired U values are 0.889623–0.907308; whether that counts as a practically important loss needs the paper's predeclared interpretation, not a post-hoc threshold.",
        f"8. JPEG30 shows the clearest target recovery among accepted pairs (S4 Person AP50 {rows['jpeg30']['metrics']['S4']['mAP50_target']:.6f}) while retaining paired U={rows['jpeg30']['scores']['U']:.6f}; this is a descriptive trade-off, not proof that one defense is preferable.",
        "9. P/U/S are calculated from target AP50 and non-target mAP50 against the matching defense's Clean arm, using unrounded values; formula inputs and outputs are recorded in the accepted-pairs JSON.",
        "10. Accepted pairs show low-to-moderate Person AP50 with non-target U near 0.89–0.91. A threshold for 'collapse' was not preregistered here, so this report does not label any result a collapse.",
        "11. The measured P values range from about 0.747 to 0.981. The sensitivity to JPEG quality is visible in this seed; robustness across random seeds remains untested.",
        "12. The next informative comparisons are the in-progress MeanF pair and queued MedianF/Data Augmentation pairs; final cross-defense conclusions wait until all Phase B audits pass.",
        "",
        "## Files and provenance",
        "",
        "Full-precision aggregate and 20-class results: `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`, `PHASE_A_PER_CLASS_RESULTS.csv`, and `PHASE_B_PER_CLASS_RESULTS.csv`. Arm status and checkpoint/data-manifest hashes: `S4_ROBUSTNESS_EXECUTION_LEDGER_20261021.csv` and `S4_DEFENSE_DATA_MANIFEST.csv`. The original S4 baseline and config/data lock hashes are retained in the same files.",
        "",
        "AT is protocol-only and remains unauthorized for formal training. Do not treat this interim report as final research acceptance.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    publication = read_json(PAIRS_PATH)
    assert publication["schema"] == "s4.robustness.accepted-results-publication.v1"
    assert publication["scientific_lock"]["source_commit"] == SOURCE
    assert publication["scientific_lock"]["config_sha256"] == CONFIG
    assert publication["scientific_lock"]["parent_S4_manifest_sha256"] == PARENT
    lock = read_json(LOCK_PATH)
    assert lock.get("parent_materialization_manifest_sha256") == PARENT
    pairs = publication["accepted_pairs"]
    assert set(pairs) == {"jpeg50", "jpeg30", "gray", "bdr", "gauss"}

    # Refresh only sanitized status fields from the latest local RRCTL snapshot.
    record = read_json(STATUS_PATH)
    remote = record["remote"]
    server_state = remote["state"]
    progress = remote["progress"]
    assert record["returncode"] == 0 and record["health"]["status"] == "healthy"
    assert server_state["pipeline_run_id"] == PIPELINE and server_state["state"] == "RUNNING"
    own_children = remote.get("own_bound_robustness_training_children", [])
    assert len(own_children) == 1
    checked = record["UTC"]
    training = progress.get("training", {})
    amp = progress.get("AMP", {})
    publication["generated_utc"] = now
    publication["pipeline_snapshot"] = {
        "observed_utc": checked,
        "state": server_state.get("state"),
        "stage": server_state.get("current_stage"),
        "active_job": server_state.get("active_job"),
        "own_training_roots": len(own_children),
        "training": {
            "epoch": training.get("epoch"),
            "AMP": training.get("AMP"),
            "data_loader_workers": training.get("data_loader_workers"),
            "last_sha256": training.get("last_sha256"),
        },
        "AMP": {key: amp.get(key) for key in (
            "epoch", "batch_index", "total_batches", "actual_updates_total", "skips_total",
            "consecutive_skips", "amp_scale_after", "model_finite", "EMA_finite",
            "optimizer_finite", "elapsed_seconds",
        )},
        "resource_blockers": server_state.get("resource_blockers", []),
        "source_pipeline_id": PIPELINE,
    }
    # The generated result artifact is deliberately path-free before publication.
    serialized = json.dumps(publication, ensure_ascii=False, indent=2, allow_nan=False)
    for forbidden in ("/" + "project" + "/", "/" + "data" + "/", "10.15.", "pass" + "word", "private" + "_" + "key"):
        assert forbidden.lower() not in serialized.lower(), f"sensitive marker in publication: {forbidden}"
    PAIRS_PATH.write_text(serialized + "\n", encoding="utf-8")

    all_path = DOC / "S4_ROBUSTNESS_ALL_RESULTS.csv"
    baseline = baseline_rows(all_path)
    lock_defenses = lock["defenses"]
    phase_a_rows = summary_rows(pairs, PHASE_A, publication["pipeline_snapshot"], lock_defenses)
    phase_b_rows = summary_rows(pairs, PHASE_B, publication["pipeline_snapshot"], lock_defenses)
    all_rows = baseline[:]
    for defense in DEFENSE_ORDER:
        if defense in pairs:
            all_rows.extend(accepted_row(defense, arm, pairs[defense]) for arm in ("clean", "S4"))
        else:
            _, statuses = pair_status(defense, publication)
            all_rows.extend(blank_row(defense, arm, statuses[arm], lock_defenses[defense]) for arm in ("clean", "S4"))
    for arm in ("clean", "S4"):
        all_rows.append(blank_row("AT", arm, "NOT_AUTHORIZED", {"scope": "protocol audit only; no formal training"}, "NOT_AUTHORIZED"))
    write_csv(DOC / "PHASE_A_ROBUSTNESS_RESULTS.csv", RESULT_FIELDS, phase_a_rows)
    write_csv(DOC / "PHASE_B_ROBUSTNESS_RESULTS.csv", RESULT_FIELDS, phase_b_rows)
    write_csv(all_path, RESULT_FIELDS, all_rows)

    phase_a_classes = [row for defense in PHASE_A if defense in pairs for arm in ("clean", "S4") for row in class_rows(defense, arm, pairs[defense])]
    phase_b_classes = [row for defense in PHASE_B if defense in pairs for arm in ("clean", "S4") for row in class_rows(defense, arm, pairs[defense])]
    write_csv(DOC / "PHASE_A_PER_CLASS_RESULTS.csv", PER_CLASS_FIELDS, phase_a_classes)
    write_csv(DOC / "PHASE_B_PER_CLASS_RESULTS.csv", PER_CLASS_FIELDS, phase_b_classes)

    baselines = {row["arm"]: row for row in baseline}
    extended = []
    for row in all_rows:
        defense, arm, status = row["defense"], row["arm"], row["status"]
        base = baselines.get(arm, {})
        for metric in METRIC_KEYS:
            value = number(row.get(metric))
            baseline_value = number(base.get(metric)) if base else None
            extended.append({
                "defense": defense,
                "arm": arm,
                "status": status,
                "metric": metric,
                "value": "" if value is None else repr(value),
                "untransformed_baseline": "" if baseline_value is None else base.get(metric),
                "defense_minus_baseline": "" if value is None or baseline_value is None else repr(value - baseline_value),
            })
    write_csv(DOC / "S4_ROBUSTNESS_EXTENDED_METRICS.csv", EXTENDED_FIELDS, extended)

    manifest_rows = []
    for defense in DEFENSE_ORDER:
        if defense in pairs:
            pair = pairs[defense]
            for arm in ("clean", "S4"):
                audit = pair["audits"][arm]
                metrics = pair["metrics"][arm]
                reused = audit.get("reused_clean") is True
                manifest_rows.append({
                    "defense": defense,
                    "arm": arm,
                    "run_id": pair["run_id"],
                    "manifest_sha256": "" if reused else audit.get("manifest_sha256", ""),
                    "parent_S4_manifest_sha256": pair["parent_S4_manifest_sha256"],
                    "checkpoint_sha256": metrics["checkpoint_sha256"],
                    "epoch_rows": 200,
                    "data_audit_status": "HISTORICAL_CLEAN_REUSE_HASH_BOUND" if reused else "PASS_16551_LABELS_GEOMETRY_CLEANVAL4952",
                })
        else:
            _, states = pair_status(defense, publication)
            for arm in ("clean", "S4"):
                manifest_rows.append({
                    "defense": defense,
                    "arm": arm,
                    "run_id": pending_run_id(defense),
                    "manifest_sha256": "",
                    "parent_S4_manifest_sha256": PARENT,
                    "checkpoint_sha256": "",
                    "epoch_rows": "",
                    "data_audit_status": states[arm],
                })
    write_csv(DOC / "S4_DEFENSE_DATA_MANIFEST.csv", [
        "defense", "arm", "run_id", "manifest_sha256", "parent_S4_manifest_sha256",
        "checkpoint_sha256", "epoch_rows", "data_audit_status",
    ], manifest_rows)

    # Phase summaries and a readable main table.
    write_json(DOC / "PHASE_A_TRAINING_STATUS.json", {
        "phase": "A", "state": "COMPLETE_ACCEPTED", "generated_utc": now,
        "pipeline_run_id": PIPELINE,
        "defenses": {name: {"run_id": pairs[name]["run_id"], "state": pairs[name]["state"], "accepted": True,
                            "paired_results_sha256": pairs[name]["evidence"]["paired_results_sha256"],
                            "arms": {arm: {"epoch_rows": 200, "checkpoint_sha256": pairs[name]["metrics"][arm]["checkpoint_sha256"]} for arm in ("clean", "S4")}}
                      for name in PHASE_A},
    })
    _, meanf_arms = pair_status("meanf", publication)
    phase_b_defenses = {}
    for name in PHASE_B:
        if name in pairs:
            phase_b_defenses[name] = {"run_id": pairs[name]["run_id"], "state": pairs[name]["state"], "accepted": True,
                                      "paired_results_sha256": pairs[name]["evidence"]["paired_results_sha256"],
                                      "arms": {arm: {"epoch_rows": 200, "checkpoint_sha256": pairs[name]["metrics"][arm]["checkpoint_sha256"],
                                                     "historical_clean_reuse": pairs[name]["audits"][arm].get("reused_clean", False)} for arm in ("clean", "S4")}}
        else:
            phase_b_defenses[name] = {"run_id": pending_run_id(name), "state": "IN_PROGRESS" if name == publication["pipeline_snapshot"].get("stage") else "NOT_RUN",
                                      "accepted": False, "arms": {"clean": meanf_arms["clean"], "S4": meanf_arms["S4"]} if name == "meanf" else {"clean": "NOT_RUN", "S4": "NOT_RUN"}}
            if name == "meanf":
                phase_b_defenses[name]["progress"] = publication["pipeline_snapshot"]
    write_json(DOC / "PHASE_B_TRAINING_STATUS.json", {
        "phase": "B", "state": "IN_PROGRESS", "generated_utc": now,
        "pipeline_run_id": PIPELINE, "current_server_stage": publication["pipeline_snapshot"].get("stage"),
        "defenses": phase_b_defenses,
    })
    (DOC / "S4_ROBUSTNESS_MAIN_TABLE.md").write_text(main_table(pairs, baseline, publication["pipeline_snapshot"]), encoding="utf-8")
    (DOC / "PHASE_A_REPORT.md").write_text(phase_a_report(pairs, publication["pipeline_snapshot"]), encoding="utf-8")
    (DOC / "PHASE_B_REPORT.md").write_text(phase_b_report(pairs, publication["pipeline_snapshot"]), encoding="utf-8")
    (DOC / "S4_ROBUSTNESS_FINAL_REPORT.md").write_text(final_report(pairs, publication["pipeline_snapshot"]), encoding="utf-8")

    # Compact, path-free arm ledger; active progress checkpoint is not a final result.
    snapshot_time = publication["pipeline_snapshot"].get("observed_utc", now)
    ledger = []
    for arm in ("clean", "S4"):
        base = next(row for row in baseline if row["arm"] == arm)
        ledger.append({"snapshot_utc": snapshot_time, "defense": "none", "arm": arm.upper(), "execution_status": "COMPLETE_ARCHIVED",
                       "run_id_or_source": "archived_S4_or_Clean", "epoch_rows": 200, "checkpoint_sha256": base["checkpoint_sha256"],
                       "manifest_sha256": PARENT, "metric_status": "FULL_AGGREGATE_AVAILABLE", "notes": "Untransformed archived baseline."})
    for defense in DEFENSE_ORDER:
        if defense in pairs:
            pair = pairs[defense]
            for arm in ("clean", "S4"):
                audit, metric = pair["audits"][arm], pair["metrics"][arm]
                ledger.append({"snapshot_utc": snapshot_time, "defense": defense, "arm": arm.upper(), "execution_status": "COMPLETE_ACCEPTED_REUSED" if audit.get("reused_clean") else "COMPLETE_ACCEPTED",
                               "run_id_or_source": pair["run_id"], "epoch_rows": 200, "checkpoint_sha256": metric["checkpoint_sha256"],
                               "manifest_sha256": audit.get("manifest_sha256", ""), "metric_status": "FULL_AGGREGATE_AND_CLASSWISE_AVAILABLE",
                               "notes": "Historical 200-epoch Clean checkpoint reused and reevaluated." if audit.get("reused_clean") else "Accepted full 200-epoch paired evaluation; see accepted-pairs JSON SHA."})
        else:
            _, states = pair_status(defense, publication)
            for arm in ("clean", "S4"):
                status = states[arm]
                progress_note = ""
                ckpt = ""
                epoch_rows: Any = ""
                metric_status = "NOT_RUN"
                if status == "IN_PROGRESS":
                    if arm.upper() == "S4":
                        epoch_rows = publication["pipeline_snapshot"].get("training", {}).get("epoch", "")
                        ckpt = publication["pipeline_snapshot"].get("training", {}).get("last_sha256", "")
                        amp = publication["pipeline_snapshot"].get("AMP", {})
                        progress_note = f"AMP epoch {amp.get('epoch')}, batch {amp.get('batch_index')}/{amp.get('total_batches')}; updates {amp.get('actual_updates_total')}; skips {amp.get('skips_total')}; consecutive {amp.get('consecutive_skips')}; finite={amp.get('model_finite')}/{amp.get('EMA_finite')}/{amp.get('optimizer_finite')}. Progress checkpoint only, not final metric evidence."
                    else:
                        progress_note = "Queued Clean historical-control reevaluation; no paired evaluation accepted yet."
                    metric_status = "IN_PROGRESS_NO_FINAL_METRICS"
                elif status == "PENDING":
                    metric_status = "PENDING"
                ledger.append({"snapshot_utc": snapshot_time, "defense": defense, "arm": arm.upper(), "execution_status": status,
                               "run_id_or_source": pending_run_id(defense), "epoch_rows": epoch_rows, "checkpoint_sha256": ckpt,
                               "manifest_sha256": "", "metric_status": metric_status, "notes": progress_note})
    for arm in ("clean", "S4"):
        ledger.append({"snapshot_utc": snapshot_time, "defense": "AT", "arm": arm.upper(), "execution_status": "NOT_AUTHORIZED",
                       "run_id_or_source": "protocol_only", "epoch_rows": "", "checkpoint_sha256": "", "manifest_sha256": "",
                       "metric_status": "NOT_RUN", "notes": "AT formal run requires a future explicit start instruction."})
    write_csv(DOC / "S4_ROBUSTNESS_EXECUTION_LEDGER_20261021.csv", [
        "snapshot_utc", "defense", "arm", "execution_status", "run_id_or_source", "epoch_rows", "checkpoint_sha256", "manifest_sha256", "metric_status", "notes",
    ], ledger)

    catalog = [
        "# S4 robustness data catalog",
        "",
        f"Snapshot: {snapshot_time}. This is a dated local summary derived from accepted result artifacts and the latest read-only RRCTL snapshot, not a live dashboard.",
        "",
        "## Locked source and protocol",
        "",
        f"- Scientific source: `{SOURCE}`.",
        f"- Config lock SHA-256: `{CONFIG}`.",
        f"- Parent S4 manifest SHA-256: `{PARENT}`.",
        "- Dataset: VOC20, Person target, 16,551 training images and the unchanged 4,952-image clean validation set.",
        "- Victim: random-initialized YOLOv8n, seed 0, 200 epochs, batch 36, workers 16, AMP true; last.pt@200 primary, best.pt secondary.",
        "- Only the original S4 protected training data is used for Poison arms. No S0/S3/S5 poisoned data, new noise, extra seed, S4 method modification, or AT run is included.",
        "",
        "## Current server execution",
        "",
        f"- Pipeline: `{PIPELINE}`; state `{publication['pipeline_snapshot'].get('state')}`; stage `{publication['pipeline_snapshot'].get('stage')}`; active job `{publication['pipeline_snapshot'].get('active_job')}`.",
        f"- Snapshot time: {snapshot_time}; one bound training root; resource blockers: `{publication['pipeline_snapshot'].get('resource_blockers')}`.",
        f"- MeanF checkpoint epoch {publication['pipeline_snapshot'].get('training', {}).get('epoch')}; AMP attempt epoch {publication['pipeline_snapshot'].get('AMP', {}).get('epoch')}, batch {publication['pipeline_snapshot'].get('AMP', {}).get('batch_index')}/{publication['pipeline_snapshot'].get('AMP', {}).get('total_batches')}; actual updates {publication['pipeline_snapshot'].get('AMP', {}).get('actual_updates_total')}, skips {publication['pipeline_snapshot'].get('AMP', {}).get('skips_total')}, all finite {publication['pipeline_snapshot'].get('AMP', {}).get('model_finite')}/{publication['pipeline_snapshot'].get('AMP', {}).get('EMA_finite')}/{publication['pipeline_snapshot'].get('AMP', {}).get('optimizer_finite')}.",
        "",
        "## Accepted pair inventory",
        "",
        "| Defense | State | RunID | Clean Person AP50 | S4 Person AP50 | Clean/S4 NT mAP50 | P | U | S |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for defense in DEFENSE_ORDER:
        if defense in pairs:
            pair = pairs[defense]
            c, s = pair["metrics"]["clean"], pair["metrics"]["S4"]
            catalog.append(f"| {defense} | COMPLETE_ACCEPTED | `{pair['run_id']}` | {c['mAP50_target']:.9f} | {s['mAP50_target']:.9f} | {c['mAP50_non_target']:.9f} / {s['mAP50_non_target']:.9f} | {pair['scores']['P']:.9f} | {pair['scores']['U']:.9f} | {pair['scores']['S']:.9f} |")
        else:
            _, states = pair_status(defense, publication)
            status = "IN_PROGRESS" if "IN_PROGRESS" in states.values() else "NOT_RUN"
            catalog.append(f"| {defense} | {status} | `{pending_run_id(defense)}` | — | — | — | — | — | — |")
    catalog += [
        "",
        "Gray, BDR, and Gauss full per-class metrics were retrieved from accepted server pair JSON and independently checked against each arm's 200-row CSV count, checkpoint SHA, parent S4 manifest, clean validation identity, and paired P/U/S calculations. The BDR local pull manifest hash was checked before normalization.",
        "",
        "## Data manifests and reports",
        "",
        "`S4_DEFENSE_DATA_MANIFEST.csv` lists generated transformed-data manifest hashes and model checkpoint hashes. Gauss Clean is a provenance-checked historical 200-epoch checkpoint reused and reevaluated; no new Clean dataset manifest was produced for that reused arm. Full metrics and pair-artifact SHA-256 values are in `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`.",
        "",
        "`PHASE_A_REPORT.md` records accepted Phase A results. `PHASE_B_REPORT.md` records accepted Gauss results and the active MeanF snapshot. `S4_ROBUSTNESS_FINAL_REPORT.md` remains explicitly interim until Phase B acceptance is complete.",
        "",
        "Raw server artifacts, absolute server paths, weights, datasets, and credentials are not part of the sanitized publication set.",
        "",
    ]
    (DOC / "S4_ROBUSTNESS_DATA_CATALOG_20261021.md").write_text("\n".join(catalog), encoding="utf-8")

    interim = [
        "# S4 robustness interim results inventory",
        "",
        f"Observation: {snapshot_time}. The server pipeline remains active; this report is a timestamped evidence snapshot, not a live query or final Phase A/B acceptance.",
        "",
        f"Pipeline `{PIPELINE}` is `{publication['pipeline_snapshot'].get('state')}` at stage `{publication['pipeline_snapshot'].get('stage')}` (`{publication['pipeline_snapshot'].get('active_job')}`). The current MeanF checkpoint is epoch {publication['pipeline_snapshot'].get('training', {}).get('epoch')}; AMP reports epoch {publication['pipeline_snapshot'].get('AMP', {}).get('epoch')} batch {publication['pipeline_snapshot'].get('AMP', {}).get('batch_index')}/{publication['pipeline_snapshot'].get('AMP', {}).get('total_batches')}, {publication['pipeline_snapshot'].get('AMP', {}).get('actual_updates_total')} actual updates, {publication['pipeline_snapshot'].get('AMP', {}).get('skips_total')} legal skips, and finite model/EMA/optimizer states. This is not a formal evaluation metric.",
        "",
        f"Scientific source `{SOURCE}`, config `{CONFIG}`, parent S4 manifest `{PARENT}`. Dataset and victim protocol are locked: VOC20 Person, 16,551 training / 4,952 clean validation, random-init YOLOv8n seed 0, 200 epochs, batch 36, 16 workers, AMP true; last@200 primary.",
        "",
        "## Accepted Clean/S4 comparisons",
        "",
        paired_table(pairs, DEFENSE_ORDER),
        "",
        "All values are fractions. Clean and S4 rows use the same defense's paired evaluation. P/U/S are derived from full precision Person AP50 and non-target mAP50; Clean rows have no P/U/S. WCD and TCD3 are included only for accepted S4 arms.",
        "",
        "## Pair evidence",
        "",
        "`S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json` contains sanitized per-class AP50/AP75/AP50-95/Recall/Precision, aggregate metrics, scores, data audit summaries, source/config/parent hashes, checkpoint SHAs, and paired artifact SHA-256 values. It omits remote absolute paths. `PHASE_A_PER_CLASS_RESULTS.csv` has 160 accepted rows; `PHASE_B_PER_CLASS_RESULTS.csv` currently has the accepted Gauss classes only.",
        "",
        "The current MeanF progress checkpoint is retained as execution evidence only. It does not populate AP or P/U/S results. MedianF and Data Augmentation remain NOT_RUN. AT remains NOT_AUTHORIZED.",
        "",
    ]
    (DOC / "S4_ROBUSTNESS_INTERIM_RESULTS_20261021.md").write_text("\n".join(interim), encoding="utf-8")

    # Promote the current state in research memory while preserving prior entries as history.
    state_path = ROOT / "research_workspace" / "STATE.md"
    old_state = state_path.read_text(encoding="utf-8")
    marker = "# S4 鲁棒性研究 — 2026-10-19"
    if marker in old_state:
        before, history = old_state.split(marker, 1)
        if before.startswith("LATEST VERIFIED"):
            before = "历史执行快照（2026-10-20；后续状态见下方最新观察）：\n\n" + before.split("\n\n", 1)[-1]
        old_state = before + marker + history
    latest = (
        f"## 最新 S4 鲁棒性发布快照 — {snapshot_time}\n\n"
        f"Phase A（JPEG50/JPEG30/Gray/BDR）均已完成配对训练、完整评价并验收。Phase B 中 Gauss 已验收；MeanF S4 正在运行，最新保存的训练检查点为 epoch {publication['pipeline_snapshot'].get('training', {}).get('epoch')}，AMP 当前报告 epoch {publication['pipeline_snapshot'].get('AMP', {}).get('epoch')} batch {publication['pipeline_snapshot'].get('AMP', {}).get('batch_index')}/{publication['pipeline_snapshot'].get('AMP', {}).get('total_batches')}；MedianF 与 Data Augmentation 尚未开始。正式结果和完整逐类指标见 `docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json` 及对应CSV。\n\n"
        f"当前唯一流水线 `{PIPELINE}` 状态 `{publication['pipeline_snapshot'].get('state')}`，阶段 `{publication['pipeline_snapshot'].get('stage')}`，RRCTL健康且绑定训练根数为 {publication['pipeline_snapshot'].get('own_training_roots')}。科学源码 `{SOURCE}`、配置锁 `{CONFIG}`、父S4 Manifest `{PARENT}` 均未更改。这里只是单seed描述性结果；Phase B与AT仍不能称最终完成。历史记录保留在本文件后续段落。\n\n"
    )
    if "## 最新 S4 鲁棒性发布快照" in old_state:
        start = old_state.index("## 最新 S4 鲁棒性发布快照")
        end = old_state.find("\n# S4 鲁棒性研究", start + 4)
        if end == -1:
            old_state = old_state[:start] + latest
        else:
            old_state = old_state[:start] + latest + old_state[end + 1:]
    else:
        old_state = latest + old_state
    state_path.write_text(old_state, encoding="utf-8")

    print(json.dumps({
        "generated_utc": now,
        "status_snapshot_utc": snapshot_time,
        "pipeline_stage": publication["pipeline_snapshot"].get("stage"),
        "phase_a_accepted_pairs": len(PHASE_A),
        "phase_b_accepted_pairs": sum(name in pairs for name in PHASE_B),
        "phase_a_per_class_rows": len(phase_a_classes),
        "phase_b_per_class_rows": len(phase_b_classes),
        "all_results_rows": len(all_rows),
        "extended_metric_rows": len(extended),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
