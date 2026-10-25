"""Deterministic final report from accepted COCO Experiment A receipts only."""
import argparse
import csv
import json
import math
from pathlib import Path

from .evaluation import paired_metrics
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
from .runtime import DATA_FILES, DATA_ROOT, ROOT, atomic, load, now, safe_relative, sha, verify_release, write_once

STAGES = ("teacher200", "teacher_eval", "calibration", "fourier24", "export25k",
          "clean200", "s4_200", "official_eval")
VOC_ALIASES = {
    "airplane": "aeroplane", "bicycle": "bicycle", "bird": "bird", "boat": "boat",
    "bottle": "bottle", "bus": "bus", "car": "car", "cat": "cat", "chair": "chair",
    "cow": "cow", "dining table": "diningtable", "dog": "dog", "horse": "horse",
    "motorcycle": "motorbike", "person": "person", "potted plant": "pottedplant",
    "sheep": "sheep", "couch": "sofa", "train": "train", "tv": "tvmonitor",
}


def validate_config(cfg):
    expected = dict(schema="s4.coco.final-report-config.v1", phase="final_report",
                    workflow_run_id=WORKFLOW, accepted_stages=list(STAGES),
                    target_class_id=0, class_count=80, seed_count=1,
                    checkpoint_policy="fixed last.pt@200; best.pt secondary only",
                    selection_from_val=False, parameter_updates=False,
                    significance_claims=False, causal_claims=False,
                    LPIPS_policy="null until independently validated", cleanup_policy="retain_all")
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Final report must remain a descriptive summary of accepted fixed checkpoints")


def shared_voc_classes(mapping):
    names = {row["name"].strip().lower() for row in mapping}
    return [alias for coco, alias in VOC_ALIASES.items() if coco in names]


def build_report(teacher_rows, clean_rows, s4_rows, mapping, export_diagnostics, *, provenance):
    if any(len(rows) != 80 for rows in (teacher_rows, clean_rows, s4_rows)):
        raise ValueError("Final report requires full 80-class Teacher/Clean/S4 metrics")
    pair = paired_metrics(clean_rows, s4_rows, target=0)
    by = []
    maps = [{r["yolo_class_id"]: r for r in rows} for rows in (teacher_rows, clean_rows, s4_rows)]
    if any(set(m) != set(range(80)) for m in maps):
        raise ValueError("Metric class IDs are incomplete or duplicated")
    for cls in range(80):
        t, c, s = (m[cls] for m in maps)
        row = dict(yolo_class_id=cls, category_id=c["category_id"], name=c["name"],
                   teacher_AP50=t["AP50"], teacher_AP75=t["AP75"], teacher_AP50_95=t["AP50_95"],
                   teacher_Recall50=t["Recall50"],
                   clean_AP50=c["AP50"], clean_AP75=c["AP75"], clean_AP50_95=c["AP50_95"],
                   clean_Recall50=c["Recall50"], clean_Precision50=c.get("Precision50"),
                   s4_AP50=s["AP50"], s4_AP75=s["AP75"], s4_AP50_95=s["AP50_95"],
                   s4_Recall50=s["Recall50"], s4_Precision50=s.get("Precision50"),
                   absolute_clean_harm_AP50=c["AP50"]-s["AP50"],
                   relative_clean_harm_AP50=(1-s["AP50"]/c["AP50"] if c["AP50"] not in (None, 0) else None))
        for f in ("teacher_AP50", "teacher_AP75", "teacher_AP50_95", "clean_AP50", "s4_AP50"):
            v = row[f]
            if v is not None and (not math.isfinite(v) or not 0 <= v <= 1):
                raise ValueError("Invalid/nonfinite AP in accepted official COCOeval rows")
        by.append(row)
    return dict(schema="s4.coco.experiment-a-final-report.v1", state="ACCEPTED",
                generated_UTC=now(), seed_count=1, analysis_scope="descriptive single-seed matched comparison",
                causal_claims=False, significance_claims=False, checkpoint_selection_from_val=False,
                parameter_updates_during_evaluation=False, target=dict(yolo_class_id=0, name="person"),
                method_protocol=dict(name="Original S4 Safe-Soft05 + TFSS-Utility",
                    scientific_parent="b1e29510e7b1ed5fa0fe361a27132d8790e65c14",
                    victim_component="ab626917f20917e58cf30033ac99df00938cb561",
                    support_values=[1.0, 0.5, 0.0], pixel_budgets=[8, 4, 0],
                    STE="2.01/255", TFSS_weight="1 - 0.15*Rnt*(1-Qt)", TFSS_eta=0.85,
                    selected_dimension=24, Fourier_epochs=24, seed=0),
                official_coco_classes=80, official_val_images=5000, metrics=by,
                paired_summary=pair, shared_voc_class_names=shared_voc_classes(mapping),
                spatial_and_perceptual_diagnostics=export_diagnostics,
                provenance=provenance,
                interpretation="AP and paired metrics are descriptive for this fixed seed and protocol; they do not establish causality or statistical significance.")


def _diagnostics(export_root, export_receipt):
    manifest = Path(export_root) / "poisoned_manifest.jsonl"
    if sha(manifest) != export_receipt.get("manifest_sha256"):
        raise ValueError("Final report export manifest differs from accepted receipt")
    rows = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Malformed retained export diagnostic row")
            rows.append(row)
    if len(rows) != 25000:
        raise ValueError("Spatial diagnostics do not cover all 25K exported images")
    psnr = [float(r["PSNR_db"]) for r in rows if r.get("PSNR_db") is not None]
    if any(not math.isfinite(x) for x in psnr):
        raise ValueError("Nonfinite export PSNR evidence")
    return dict(source="accepted Original S4 export poisoned_manifest.jsonl",
                manifest_sha256=sha(manifest), image_rows=len(rows),
                PSNR_db=dict(mean=(sum(psnr)/len(psnr) if psnr else None),
                             minimum=(min(psnr) if psnr else None), maximum=(max(psnr) if psnr else None),
                             measured_image_count=len(psnr), unchanged_or_zero_mse_count=len(rows)-len(psnr)),
                changed_pixels=sum(int(r["changed_pixels"]) for r in rows),
                person_only_support_pixels=sum(int(r["support_person_only_pixels"]) for r in rows),
                person_non_target_overlap_support_pixels=sum(int(r["support_overlap_pixels"]) for r in rows),
                maximum_decoded_byte_difference=max(int(r["decoded_max_byte_difference"]) for r in rows),
                maximum_overlap_byte_difference=max(int(r["decoded_overlap_max_byte_difference"]) for r in rows),
                outside_nonzero_pixels=sum(int(r["decoded_outside_nonzero_pixels"]) for r in rows),
                LPIPS=dict(value=None, state="not_validated", interpretation="not measured; no value inferred"))


def _accepted_chain(manifest):
    output = WORKFLOW_OUTPUT
    state = load(output / "pipeline_state.json")
    deps = manifest.get("phase_dependencies")
    if not isinstance(deps, dict) or set(deps) != set(STAGES):
        raise ValueError("Final report lacks the exact accepted serial stage chain")
    receipts = {}
    for stage in STAGES:
        path = output / stage / "accepted.json"
        row = load(path)
        digest = sha(path)
        slot = state.get("stages", {}).get(stage, {})
        if (row.get("state") != "ACCEPTED" or slot.get("state") != "ACCEPTED"
                or slot.get("receipt_sha256") != digest or deps[stage].get("accepted_sha256") != digest
                or deps[stage].get("accepted_path") != str(path)):
            raise ValueError("Accepted final report dependency changed: " + stage)
        receipts[stage] = dict(path=str(path), sha256=digest)
    teacher_eval = load(output / "teacher_eval/accepted.json")
    if (sha(output / "teacher_eval/official_metrics.json") != teacher_eval.get("metrics_sha256")
            or sha(output / "teacher_eval/COCO_PER_CLASS_AP.csv") != teacher_eval.get("per_class_csv_sha256")):
        raise ValueError("Fixed-last Teacher assessment files differ from its accepted receipt")
    official = load(output / "official_eval/accepted.json")
    for name in ("clean200", "s4_200"):
        prefix = "clean" if name == "clean200" else "s4"
        if (sha(output / f"official_eval/{name}/official_metrics.json") != official.get(prefix + "_metrics_sha256")
                or sha(output / f"official_eval/{name}/COCO_PER_CLASS_AP.csv") != official.get(prefix + "_per_class_csv_sha256")):
            raise ValueError("Matched-victim official metrics differ from accepted receipt: " + name)
    return receipts


def _formal_gate(registration, manifest, release_path):
    ready_path = Path(registration).resolve()
    expected_ready = (WORKFLOW_OUTPUT / "receipts/final_report_READY.json").resolve()
    if ready_path != expected_ready:
        raise ValueError("Wrong final-report registration path")
    ready = load(ready_path)
    release_relative = safe_relative(ready.get("release_relative", ""))
    if (ready.get("phase") != "final_report" or ready.get("workflow_run_id") != WORKFLOW
            or ready.get("module") != "coco_s4.phase_final_report" or ready.get("pre_run_result") != "pass"
            or ready.get("source_root") != str(Path(release_path).resolve().parent)
            or (Path(ready["source_root"]) / release_relative).resolve() != Path(release_path).resolve()
            or ready.get("release_sha256") != sha(release_path)
            or ready.get("previous_stage_receipt_sha256") != manifest["phase_dependencies"]["official_eval"]["accepted_sha256"]):
        raise ValueError("Final-report READY is not bound to accepted official evaluation")


def run(release_path, output, *, registration=None):
    manifest = verify_release(release_path)
    if manifest.get("phase") != "final_report" or manifest.get("workflow_run_id") != WORKFLOW:
        raise ValueError("Wrong final-report release")
    if not ROOT.resolve().is_relative_to((OWNED / "releases").resolve()):
        raise ValueError("Final-report code must execute from this workflow's owned immutable release root")
    cfg = load(ROOT / manifest["phase_config_path"])
    validate_config(cfg)
    if registration is None or Path(output).resolve() != (WORKFLOW_OUTPUT / "final_report").resolve():
        raise ValueError("Final report must run only through the existing serial controller")
    _formal_gate(registration, manifest, release_path)
    receipts = _accepted_chain(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(phase="final_report", state="RUNNING", source_commit=manifest["source_commit"],
                  release_sha256=sha(release_path), started_UTC=now(), cleanup_policy="retain_all",
                  deleted_paths=[])
    atomic(output / "status.json", status)
    try:
        accepted = _build_and_write(output, manifest, cfg, receipts)
        atomic(output / "status.json", accepted)
        return accepted
    except BaseException as exc:
        status.update(state="FAILED", error=repr(exc), ended_UTC=now())
        atomic(output / "status.json", status)
        raise


def _build_and_write(output, manifest, cfg, receipts):
    teacher = load(WORKFLOW_OUTPUT / "teacher_eval/official_metrics.json")["classes"]
    clean = load(WORKFLOW_OUTPUT / "official_eval/clean200/official_metrics.json")["classes"]
    s4 = load(WORKFLOW_OUTPUT / "official_eval/s4_200/official_metrics.json")["classes"]
    mapping = load(ROOT / "docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json")["mapping"]
    export_receipt = load(WORKFLOW_OUTPUT / "export25k/accepted.json")
    diagnostics = _diagnostics(WORKFLOW_OUTPUT / "export25k", export_receipt)
    checkpoints = {k: load(WORKFLOW_OUTPUT / f"{k}/accepted.json") for k in ("clean200", "s4_200")}
    provenance = dict(source_commit=manifest["source_commit"], release_sha256=sha(release_path),
                      config_sha256=sha(ROOT / manifest["phase_config_path"]), data_files=DATA_FILES,
                      stage_receipts=receipts,
                      checkpoints={k: dict(primary="last.pt@200", sha256=v["primary_checkpoint_sha256"],
                                           best_secondary_sha256=v["best_secondary_checkpoint_sha256"],
                                           initial_state_sha256=v["initial_state_sha256"])
                                   for k, v in checkpoints.items()})
    report = build_report(teacher, clean, s4, mapping, diagnostics, provenance=provenance)
    write_once(output / "final_report.json", report)
    fields = list(report["metrics"][0])
    with (output / "COCO_EXPERIMENT_A_PER_CLASS.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(report["metrics"])
    pair = report["paired_summary"]
    md = ["# Original S4 COCO Experiment A", "", "Fixed-seed descriptive results; no causal or significance claim.", "",
          f"- Target Person AP50 reduction P: {pair['P']}", f"- Non-target AP50 retention U: {pair['U']}",
          f"- Harm-retention harmonic score S: {pair['S']}", f"- WCD: {pair['WCD']}", f"- TCD3: {pair['TCD3']}",
          f"- Official val2017 images/classes: {report['official_val_images']} / {report['official_coco_classes']}",
          f"- LPIPS: not measured (unvalidated); export PSNR mean: {diagnostics['PSNR_db']['mean']}", "",
          "Per-class AP, recall, precision, absolute clean harm, and source receipts are in the adjacent CSV and JSON."]
    (output / "FINAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    accepted = dict(schema="s4.coco.final-report-acceptance.v1", state="ACCEPTED",
                    phase="final_report", source_commit=manifest["source_commit"],
                    release_sha256=sha(release_path), config_sha256=provenance["config_sha256"],
                    predecessor_receipt_sha256=manifest["phase_dependencies"]["official_eval"]["accepted_sha256"],
                    report_sha256=sha(output / "final_report.json"),
                    per_class_csv_sha256=sha(output / "COCO_EXPERIMENT_A_PER_CLASS.csv"),
                    markdown_sha256=sha(output / "FINAL_REPORT.md"),
                    seed_count=1, causal_claims=False, significance_claims=False,
                    parameter_updates=False, checkpoint_selection_allowed=False,
                    LPIPS_measured=False, completed_UTC=now())
    write_once(output / "accepted.json", accepted)
    return accepted


def main():
    p = argparse.ArgumentParser(); p.add_argument("--release", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); p.add_argument("--registration", type=Path)
    a = p.parse_args(); run(a.release, a.output, registration=a.registration)


if __name__ == "__main__":
    main()
