"""Paired official COCO80 evaluation of the fixed matched last@200 victims.

This stage is read-only: it evaluates the accepted clean and Original S4
victims on the untouched 5K val2017 set and never updates or selects weights.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

from .coco_eval import PREDICT_ARGS, official_metrics, predict, raw_signature, raw_tensor_devices
from .evaluation import paired_metrics
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, phase_release
from .runtime import DATA_ROOT, ROOT, atomic, load, now, sha, write_once

EVALUATOR_SHA = "1760c10459dfb4229e7436ae380228428efb0115bbe332a51b72d07fa085d8c0"
STAGES = ("teacher200", "teacher_eval", "calibration", "fourier24", "export25k", "clean200", "s4_200")


def validate_config(cfg):
    expected = dict(
        schema="s4.coco.official-eval-config.v1", phase="official_eval", workflow_run_id=WORKFLOW,
        checkpoint_policy="fixed clean200 and s4_200 native last.pt@200", image_count=5000,
        class_count=80, target_class_id=0, predict_args=PREDICT_ARGS,
        metrics=["AP50", "AP75", "AP50_95", "Recall50", "Precision50"],
        paired_metrics=["P", "U", "S", "WCD", "TCD3", "absolute_clean_harm"],
        parameter_updates=False, checkpoint_selection_allowed=False, method_statistics_allowed=False,
        metric_driven_tuning_allowed=False, evaluator_version="2.0.10",
        evaluator_wheel_sha256=EVALUATOR_SHA, cleanup_policy="retain_all")
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen paired COCO official evaluation contract changed")


def accepted_dependencies(manifest, workflow_output=WORKFLOW_OUTPUT):
    output = Path(workflow_output).resolve()
    deps = manifest.get("phase_dependencies")
    if not isinstance(deps, dict) or set(deps) != set(STAGES):
        raise ValueError("Official evaluation requires the exact accepted serial predecessor chain")
    state = load(output / "pipeline_state.json")
    checkpoints = {}
    receipts = {}
    for name in STAGES:
        receipt_path = output / name / "accepted.json"
        receipt = load(receipt_path)
        actual = sha(receipt_path)
        slot = state.get("stages", {}).get(name, {})
        dep = deps[name]
        if (receipt.get("state") != "ACCEPTED" or slot.get("state") != "ACCEPTED"
                or slot.get("receipt_sha256") != actual
                or dep.get("accepted_path") != str(receipt_path)
                or dep.get("accepted_sha256") != actual):
            raise ValueError("Accepted predecessor receipt changed: " + name)
        receipts[name] = dict(path=str(receipt_path), sha256=actual)
    for name in ("clean200", "s4_200"):
        receipt = load(output / name / "accepted.json")
        checkpoint = output / name / "train" / "weights" / "last.pt"
        full_state = output / name / "train" / "state_snapshots" / "epoch_200_full_state.pt"
        if (receipt.get("completed_epochs") != 200 or receipt.get("primary_checkpoint") != "last.pt@200"
                or receipt.get("zero_validation_forwards") is not True or receipt.get("pretrained") is not False
                or sha(checkpoint) != receipt.get("primary_checkpoint_sha256")
                or sha(full_state) != receipt.get("full_state_sha256")
                or deps[name].get("checkpoint_path") != str(checkpoint)
                or deps[name].get("checkpoint_sha256") != sha(checkpoint)):
            raise ValueError("Victim is not the exact accepted native last@200 checkpoint: " + name)
        checkpoints[name] = dict(path=str(checkpoint), sha256=sha(checkpoint),
                                 full_state_sha256=sha(full_state),
                                 initial_state_sha256=receipt.get("initial_state_sha256"))
    clean, s4 = (load(output / f"{n}/accepted.json") for n in ("clean200", "s4_200"))
    if (not checkpoints["clean200"]["initial_state_sha256"]
            or checkpoints["clean200"]["initial_state_sha256"] != checkpoints["s4_200"]["initial_state_sha256"]
            or s4.get("matched_clean_initial_state") is not True):
        raise ValueError("Clean and Original S4 victims are not a matched initialization pair")
    teacher_eval = load(output / "teacher_eval/accepted.json")
    if (teacher_eval.get("image_count") != 5000 or teacher_eval.get("classes") != 80
            or teacher_eval.get("parameter_updates") is not False
            or teacher_eval.get("checkpoint_selection_allowed") is not False):
        raise ValueError("Separate fixed-last Teacher assessment is not accepted")
    export = load(output / "export25k/accepted.json")
    if export.get("decoded_budget_passed") is not True or export.get("labels_byte_identical") is not True:
        raise ValueError("Poisoned export did not preserve labels and decoded 8/4/0 budgets")
    return dict(receipts=receipts, checkpoints=checkpoints)


def _formal_gate(registration, release_path, manifest, config_sha):
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    registration = Path(registration).resolve()
    if registration != (WORKFLOW_OUTPUT / "receipts/official_eval_READY.json").resolve():
        raise ValueError("Wrong existing-controller registration path")
    ready = load(registration)
    predecessor_sha = manifest["phase_dependencies"]["s4_200"]["accepted_sha256"]
    if (ready.get("phase") != "official_eval" or ready.get("workflow_run_id") != WORKFLOW
            or ready.get("module") != "coco_s4.phase_official_eval" or ready.get("pre_run_result") != "pass"
            or ready.get("source_root") != str(Path(release_path).resolve().parent)
            or ready.get("release_sha256") != sha(release_path)
            or ready.get("previous_stage_receipt_sha256") != predecessor_sha):
        raise ValueError("READY registration is not bound to this exact official-evaluation candidate")
    closure_path = Path(ready.get("scientific_closure_path", "")).resolve()
    if (closure_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or sha(closure_path) != ready.get("scientific_closure_sha256")):
        raise ValueError("Scientific closure path/hash mismatch")
    closure = load(closure_path)
    required = dict(phase="official_eval", source_commit=manifest["source_commit"],
                    release_sha256=sha(release_path), phase_config_sha256=config_sha,
                    reviewer_count=1, pre_run_result="pass", reviewer_independence=True,
                    scientific_verdict="allow_run", blockers=[])
    if any(closure.get(k) != v for k, v in required.items()):
        raise ValueError("One current-candidate independent review and main closure are required")
    review_path = Path(closure.get("review_path", "")).resolve()
    if (review_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or sha(review_path) != closure.get("review_sha256")):
        raise ValueError("Independent review file is not retained in the workflow receipt root")
    review = load(review_path)
    if (review.get("phase") != "official_eval" or review.get("source_commit") != manifest["source_commit"]
            or review.get("release_sha256") != sha(release_path)
            or review.get("smoke_sha256") != closure.get("smoke_sha256")
            or review.get("scientific_verdict") != "allow_run" or review.get("blockers") not in ([], None)):
        raise ValueError("Independent reviewer did not approve this exact official-evaluation candidate")
    smoke_path = Path(closure.get("smoke_path", "")).resolve()
    if not smoke_path.is_relative_to((OWNED / "validation").resolve()) or sha(smoke_path) != closure.get("smoke_sha256"):
        raise ValueError("Official-evaluation smoke must come from the isolated owned validation root")
    smoke = load(smoke_path)
    if (smoke.get("state") != "passed" or smoke.get("phase") != "official_eval"
            or smoke.get("source_commit") != manifest["source_commit"]
            or smoke.get("release_sha256") != sha(release_path)
            or smoke.get("phase_config_sha256") != config_sha
            or smoke.get("official_metrics_disabled") is not True
            or smoke.get("artifact_ingest_disabled") is not True
            or smoke.get("parameter_updates") is not False
            or smoke.get("checkpoint_selection_allowed") is not False
            or set(smoke.get("victim_probes", {})) != {"clean200", "s4_200"}):
        raise ValueError("A current-release paired bounded inference smoke is required")
    for probe in smoke["victim_probes"].values():
        eq = probe.get("baseline_equivalence_probe", {})
        if (probe.get("actual_device") != "cuda:0" or probe.get("finite_outputs") is not True
                or eq.get("reference_actual_device") != probe.get("actual_device")
                or not eq.get("raw_outputs_exact")
                or not eq.get("decoded_predictions_exact") or eq.get("absolute_difference") != 0
                or eq.get("tolerance") != 0):
            raise ValueError("Each matched victim needs exact native reference equivalence")
    if not 1 <= smoke.get("total_raw_forwards_including_warmups", 0) <= 100:
        raise ValueError("Paired smoke exceeded its finite raw-forward budget")
    return closure, smoke


def _native_reference(checkpoint, images, annotation, mapping, batches, candidate):
    import torch
    from ultralytics import YOLO
    entries = sorted(load(annotation)["images"], key=lambda im: im["id"])
    paths = [str(Path(images) / x["file_name"]) for x in entries[:PREDICT_ARGS["batch"] * batches]]
    model = YOLO(str(checkpoint))
    signatures = []; reference_devices = set()
    def reference_hook(module, args, result):
        reference_devices.update(raw_tensor_devices(result))
        signatures.append(raw_signature(result))
    hook = model.model.register_forward_hook(reference_hook)
    table = {r["yolo_class_id"]: r["category_id"] for r in mapping}
    detections = []
    source = Path(candidate).parent / 'reference_images.txt'
    with source.open('x', encoding='utf-8') as f:
        f.write(''.join(p + '\n' for p in paths))
    try:
        for result in model.predict(source=str(source), **PREDICT_ARGS):
            for box in result.boxes.data.detach().cpu().tolist():
                x1, y1, x2, y2, score, cls = box
                detections.append(dict(image_id=int(Path(result.path).stem), category_id=table[int(cls)],
                                       bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
    finally:
        hook.remove()
    audit = load(Path(candidate).parent / "prediction_audit.json")
    if signatures != audit["raw_output_signatures"] or detections != load(candidate):
        raise ValueError("Independent native YOLO.predict reference differs from production sink")
    if len(reference_devices) != 1:
        raise ValueError("Independent reference did not expose one unambiguous forward device")
    return dict(raw_outputs_exact=True, decoded_predictions_exact=True, absolute_difference=0, tolerance=0,
                reference_raw_forward_calls=len(signatures),
                reference_actual_device=next(iter(reference_devices)), weights_sha256=sha(checkpoint))


def _checkpoint_check(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("ema")
    if model is None:
        model = ck.get("model")
    if (ck.get("epoch") != 199 or model is None or getattr(model.model[-1], "nc", None) != 80
            or model.names.get(0) != "person"):
        raise ValueError("Victim checkpoint is not the fixed native COCO80 last@200 model")
    if ck.get("optimizer") is None or ck.get("scaler") is None:
        raise ValueError("Native last@200 did not retain optimizer/scaler state")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite last@200 victim checkpoint")


def run(release_path, output, *, registration=None, smoke_batches=0):
    manifest, cfg, teacher_checkpoint, teacher_receipt = phase_release(release_path, "official_eval")
    validate_config(cfg)
    dep = accepted_dependencies(manifest)
    config_sha = sha(ROOT / manifest["phase_config_path"])
    output = Path(output).resolve()
    if smoke_batches:
        if not 1 <= smoke_batches <= 3 or not output.is_relative_to((OWNED / "validation").resolve()):
            raise ValueError("Paired inference smoke requires 1..3 batches per checkpoint in isolated validation root")
    else:
        if output != (WORKFLOW_OUTPUT / "official_eval").resolve():
            raise ValueError("Wrong existing serial official-evaluation output")
        _formal_gate(registration, release_path, manifest, config_sha)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "yolo_settings")
    os.environ["MPLCONFIGDIR"] = str(output / "mpl")
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    import numpy as np
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("Official paired evaluation requires the serial controller's admitted CUDA device")
    checkpoints = {name: Path(dep["checkpoints"][name]["path"]) for name in ("clean200", "s4_200")}
    for path in checkpoints.values():
        _checkpoint_check(path)
    mapping = load(ROOT / "docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json")["mapping"]
    evaluator = manifest["evaluator"]
    if (Path(evaluator["directory"]).resolve() != (OWNED / "evaluator/pycocotools_2_0_10/addon").resolve()
            or evaluator["wheel_sha256"] != EVALUATOR_SHA):
        raise ValueError("Unexpected pinned isolated official COCO evaluator")
    common = dict(phase="official_eval", source_commit=manifest["source_commit"],
                  release_sha256=sha(release_path), phase_config_sha256=config_sha,
                  previous_stage_receipt_sha256=manifest["phase_dependencies"]["s4_200"]["accepted_sha256"],
                  victim_checkpoint_sha256s={k: sha(v) for k, v in checkpoints.items()},
                  computation_kind="paired-read-only-official-COCOeval", parameter_updates=False,
                  checkpoint_selection_allowed=False, method_statistics_allowed=False,
                  metric_driven_tuning_allowed=False, official_metrics_disabled=bool(smoke_batches),
                  artifact_ingest_disabled=bool(smoke_batches), cleanup_policy="retain_all", deleted_paths=[])
    status = dict(common, state="RUNNING", started_UTC=now())
    atomic(output / "status.json", status)
    write_once(output / "phase_spec.json", dict(common, config=cfg, dependencies=dep, teacher_acceptance=teacher_receipt))
    try:
        audits = {}
        for name, checkpoint in checkpoints.items():
            sub = output / name
            sub.mkdir(exist_ok=False)
            predictions, audit = predict(checkpoint, DATA_ROOT / "images/val",
                                         DATA_ROOT / "metadata/instances_val2017.json",
                                         mapping, sub, smoke_batches=smoke_batches)
            if audit["actual_device"] != "cuda:0":
                raise ValueError("Paired official evaluation did not use CUDA device 0")
            audits[name] = audit
            if smoke_batches:
                reference = _native_reference(checkpoint, DATA_ROOT / "images/val",
                    DATA_ROOT / "metadata/instances_val2017.json",
                    mapping, smoke_batches, predictions)
                if reference["reference_actual_device"] != audit["actual_device"]:
                    raise ValueError("Independent reference and production sink used different devices")
                audits[name]["baseline_equivalence_probe"] = reference
            else:
                rows = official_metrics(DATA_ROOT / "metadata/instances_val2017.json",
                                        predictions, mapping, sub, evaluator["directory"], evaluator["files"])
                if len(rows) != 80 or audit["image_count"] != 5000:
                    raise ValueError("Official COCOval evaluation did not complete all 80 classes and 5000 images")
        if smoke_batches:
            total_raw = sum(a["raw_forward_calls"] + a["baseline_equivalence_probe"]["reference_raw_forward_calls"]
                            for a in audits.values())
            if not 1 <= total_raw <= 100:
                raise ValueError("Paired inference smoke exceeded 100 native raw forwards")
            status.update(state="passed", finite_outputs=all(a["finite_outputs"] for a in audits.values()),
                          victim_probes=audits, actual_inference_batches=sum(a["image_count"] for a in audits.values()),
                          total_raw_forwards_including_warmups=total_raw, completed_UTC=now())
            write_once(output / "smoke_summary.json", status)
        else:
            clean_rows = load(output / "clean200/official_metrics.json")["classes"]
            s4_rows = load(output / "s4_200/official_metrics.json")["classes"]
            paired = paired_metrics(clean_rows, s4_rows, target=0)
            if not all(sha(checkpoints[k]) == common["victim_checkpoint_sha256s"][k] for k in checkpoints):
                raise ValueError("Read-only evaluation changed a victim checkpoint")
            status.update(state="ACCEPTED", finite_outputs=True, image_count=5000, classes=80,
                          per_class_rows=80, paired_metrics=paired,
                          clean_metrics_sha256=sha(output / "clean200/official_metrics.json"),
                          s4_metrics_sha256=sha(output / "s4_200/official_metrics.json"),
                          clean_per_class_csv_sha256=sha(output / "clean200/COCO_PER_CLASS_AP.csv"),
                          s4_per_class_csv_sha256=sha(output / "s4_200/COCO_PER_CLASS_AP.csv"),
                          clean_checkpoint_sha256=common["victim_checkpoint_sha256s"]["clean200"],
                          s4_checkpoint_sha256=common["victim_checkpoint_sha256s"]["s4_200"],
                          completed_UTC=now())
            write_once(output / "paired_metrics.json", status["paired_metrics"])
            write_once(output / "accepted.json", status)
        atomic(output / "status.json", status)
        return status
    except BaseException as exc:
        status.update(state="FAILED", error=repr(exc), ended_UTC=now())
        atomic(output / "status.json", status)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--registration", type=Path)
    p.add_argument("--smoke-batches", type=int, default=0)
    a = p.parse_args()
    run(a.release, a.output, registration=a.registration, smoke_batches=a.smoke_batches)


if __name__ == "__main__":
    main()
