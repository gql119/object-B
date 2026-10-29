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

from .coco_eval import (PREDICT_ARGS, official_metrics, predict, raw_signature,
                        raw_tensor_devices, validate_val2017_image_identity)
from .evaluation import paired_metrics, validate_paired_metric_payload
from .dimension_protocol import PROTOCOL as DIMENSION_PROTOCOL
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, phase_release
from .runtime import DATA_FILES, DATA_ROOT, ROOT, atomic, load, now, sha, write_once
from .workflow_admission import POLICY as RISK_ESTIMATOR_POLICY

EVALUATOR_SHA = "1760c10459dfb4229e7436ae380228428efb0115bbe332a51b72d07fa085d8c0"
VAL_IMAGE_MANIFEST = "metadata/VAL2017_IMAGE_MANIFEST_20261025.jsonl"
ABSOLUTE_CLEAN_HARM_FORMULA = "clean Person AP50 - S4 Person AP50 (unclipped AP fraction)"
ORIGINAL_REVIEW_ID = "COCO_OFFICIAL_EVAL_INDEPENDENT_SCIENTIFIC_REVIEW_20261029"
ORIGINAL_REVIEW_COMMIT = "ed192435cd3bd30cbf412c731288dae472af43ce"
ORIGINAL_BLOCKER_IDS = ("B1", "B2")
RESOLVED_FINDING_IDS = ("B1", "B2", "G1")
STAGES = ("teacher200", "teacher_eval", "calibration", "fourier24", "export25k", "clean200", "s4_200")


def validate_config(cfg):
    expected = dict(
        schema="s4.coco.official-eval-config.v1", phase="official_eval", workflow_run_id=WORKFLOW,
        checkpoint_policy="fixed clean200 and s4_200 native last.pt@200", image_count=5000,
        class_count=80, target_class_id=0, predict_args=PREDICT_ARGS,
        metrics=["AP50", "AP75", "AP50_95", "Recall50", "Precision50"],
        paired_metrics=["P", "U", "S", "WCD", "TCD3", "absolute_clean_harm"],
        paired_metric_formulas={"absolute_clean_harm": ABSOLUTE_CLEAN_HARM_FORMULA},
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


def validate_protocol_identity(manifest):
    expected = dict(risk_estimator_policy=RISK_ESTIMATOR_POLICY,
                    dimension_protocol=DIMENSION_PROTOCOL,
                    strict_Original_S4_equivalence_claim=False)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Official evaluation lacks the accepted COCO protocol attribution")
    return expected


def validate_original_review(review):
    """Accept only the retained single negative review whose findings this phase closes."""
    reviewer = review.get("reviewer", {})
    blockers = review.get("blockers", [])
    blocker_ids = tuple(sorted(item.get("id") for item in blockers
                               if isinstance(item, dict) and item.get("severity") == "blocking"))
    evidence = review.get("review_evidence", {})
    if (review.get("schema_version") != "coco.official-eval-independent-scientific-review.v1"
            or review.get("review_id") != ORIGINAL_REVIEW_ID
            or review.get("review_mode") != "scientific_review"
            or review.get("pre_run_code_commit") != ORIGINAL_REVIEW_COMMIT
            or review.get("scientific_verdict") != "scientifically_incorrect"
            or review.get("decision") != "do_not_run"
            or reviewer.get("mode") != "independent delegated read-only review"
            or reviewer.get("parent_conclusions_used") is not False
            or blocker_ids != ORIGINAL_BLOCKER_IDS
            or evidence.get("candidate_head_matches_packet") is not True
            or evidence.get("local_manifest_sha256_matches_runtime_pin") != DATA_FILES[VAL_IMAGE_MANIFEST]
            or evidence.get("reviewer_reran_tests_metrics_experiments_or_registration") is not False):
        raise ValueError("Only the retained original independent negative review can be closed here")
    return blocker_ids


def _formal_gate(registration, release_path, manifest, config_sha):
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    registration = Path(registration).resolve()
    if registration != (WORKFLOW_OUTPUT / "receipts/official_eval_READY.json").resolve():
        raise ValueError("Wrong existing-controller registration path")
    ready = load(registration)
    predecessor_sha = manifest["phase_dependencies"]["s4_200"]["accepted_sha256"]
    protocol_identity = validate_protocol_identity(manifest)
    if (ready.get("phase") != "official_eval" or ready.get("workflow_run_id") != WORKFLOW
            or ready.get("module") != "coco_s4.phase_official_eval" or ready.get("pre_run_result") != "pass"
            or ready.get("source_root") != str(Path(release_path).resolve().parent)
            or ready.get("release_sha256") != sha(release_path)
            or ready.get("previous_stage_receipt_sha256") != predecessor_sha
            or ready.get("val2017_image_manifest_sha256") != DATA_FILES[VAL_IMAGE_MANIFEST]
            or ready.get("val2017_image_count") != 5000
            or ready.get("independent_scientific_verdict") != "scientifically_incorrect"
            or ready.get("closure_verdict") != "blockers_resolved_by_main_agent"
            or ready.get("resolved_blocker_ids") != list(ORIGINAL_BLOCKER_IDS)
            or any(ready.get(k) != v for k, v in protocol_identity.items())):
        raise ValueError("READY registration is not bound to this exact official-evaluation candidate")
    closure_path = Path(ready.get("scientific_closure_path", "")).resolve()
    if (closure_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or sha(closure_path) != ready.get("scientific_closure_sha256")):
        raise ValueError("Scientific closure path/hash mismatch")
    closure = load(closure_path)
    blocker_path = Path(closure.get("blocker_closure_path", "")).resolve()
    if (blocker_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or sha(blocker_path) != closure.get("blocker_closure_sha256")):
        raise ValueError("Main blocker-closure path/hash mismatch")
    blocker_closure = load(blocker_path)
    smoke_path = Path(closure.get("smoke_path", "")).resolve()
    if (not smoke_path.is_relative_to((OWNED / "validation").resolve())
            or sha(smoke_path) != closure.get("smoke_sha256")):
        raise ValueError("Official-evaluation smoke must come from the isolated owned validation root")
    cpu_path = Path(blocker_closure.get("cpu_validation_path", "")).resolve()
    if (cpu_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or sha(cpu_path) != blocker_closure.get("cpu_validation_sha256")):
        raise ValueError("Blocker-closure CPU evidence path/hash mismatch")
    cpu_validation = load(cpu_path)
    required = dict(phase="official_eval", source_commit=manifest["source_commit"],
                    release_sha256=sha(release_path), phase_config_sha256=config_sha,
                    reviewer_count=1, pre_run_result="pass", reviewer_independence=True,
                    closure_verdict="blockers_resolved_by_main_agent",
                    independent_scientific_verdict="scientifically_incorrect",
                    resolved_blocker_ids=list(ORIGINAL_BLOCKER_IDS),
                    resolved_attribution_gap_ids=["G1"], unresolved_blockers=[],
                    original_review_sha256=sha(WORKFLOW_OUTPUT / "receipts/official_eval_ORIGINAL_NEGATIVE_REVIEW.json"),
                    blocker_closure_sha256=sha(blocker_path), **protocol_identity)
    if any(closure.get(k) != v for k, v in required.items()):
        raise ValueError("One current-candidate independent review and main closure are required")
    review_path = Path(closure.get("review_path", "")).resolve()
    if (review_path.parent != (WORKFLOW_OUTPUT / "receipts").resolve()
            or review_path.name != "official_eval_ORIGINAL_NEGATIVE_REVIEW.json"
            or sha(review_path) != closure.get("review_sha256")):
        raise ValueError("Independent review file is not retained in the workflow receipt root")
    review = load(review_path)
    if (sha(review_path) != closure.get("review_sha256")
            or review.get("scientific_verdict") != "scientifically_incorrect"
            or review.get("decision") != "do_not_run"
            or validate_original_review(review) != ORIGINAL_BLOCKER_IDS):
        raise ValueError("The original negative review must remain intact and be closed with current evidence")
    blocker_findings = blocker_closure.get("findings", {})
    expected_source_paths = {
        "B1": ("coco_s4/evaluation.py", "coco_s4/phase_official_eval.py", "coco_s4/configs/official_eval.json"),
        "B2": ("coco_s4/coco_eval.py", "coco_s4/phase_official_eval.py"),
    }
    expected_source_file_paths = {path for paths in expected_source_paths.values() for path in paths}
    expected_source_sha256 = {path: manifest.get("files", {}).get(path) for path in expected_source_file_paths}
    if (blocker_closure.get("schema") != "s4.coco.official-eval-blocker-closure.v1"
            or blocker_closure.get("state") != "closed"
            or blocker_closure.get("candidate_source_commit") != manifest["source_commit"]
            or blocker_closure.get("release_sha256") != sha(release_path)
            or blocker_closure.get("phase_config_sha256") != config_sha
            or blocker_closure.get("smoke_sha256") != closure.get("smoke_sha256")
            or blocker_closure.get("original_review_sha256") != sha(review_path)
            or blocker_closure.get("closed_blocker_ids") != list(ORIGINAL_BLOCKER_IDS)
            or blocker_closure.get("closed_attribution_gap_ids") != ["G1"]
            or blocker_closure.get("unresolved_blocker_ids") != []
            or blocker_closure.get("cpu_validation_sha256") != sha(cpu_path)
            or blocker_closure.get("val2017_image_manifest_sha256") != DATA_FILES[VAL_IMAGE_MANIFEST]
            or blocker_closure.get("val2017_image_identity_receipt_sha256") != sha(smoke_path.parent / "val2017_image_identity.json")
            or blocker_closure.get("source_file_sha256") != expected_source_sha256
            or set(blocker_findings) != set(RESOLVED_FINDING_IDS)
            or any(item.get("state") != "closed" for item in blocker_findings.values())
            or blocker_findings.get("B1", {}).get("metric") != "absolute_clean_harm"
            or blocker_findings.get("B1", {}).get("formula") != ABSOLUTE_CLEAN_HARM_FORMULA
            or blocker_findings.get("B2", {}).get("identity_image_count") != 5000
            or blocker_findings.get("B2", {}).get("image_manifest_sha256") != DATA_FILES[VAL_IMAGE_MANIFEST]
            or blocker_findings.get("B2", {}).get("smoke_identity_receipt_sha256") != sha(smoke_path.parent / "val2017_image_identity.json")
            or blocker_findings.get("G1", {}).get("risk_estimator_policy") != protocol_identity["risk_estimator_policy"]
            or blocker_findings.get("G1", {}).get("dimension_protocol") != protocol_identity["dimension_protocol"]
            or blocker_findings.get("G1", {}).get("strict_Original_S4_equivalence_claim") is not False):
        raise ValueError("Main blocker-to-fix closure does not bind the current source and smoke")
    if (cpu_validation.get("schema") != "s4.coco.official-eval-blocker-fix-cpu.v1"
            or cpu_validation.get("state") != "passed"
            or cpu_validation.get("source_commit") != manifest["source_commit"]
            or cpu_validation.get("result", {}).get("state") != "passed"
            or cpu_validation.get("result", {}).get("same_name_same_size_substitution_rejected") is not True
            or cpu_validation.get("result", {}).get("missing_metric_rejected") is not True
            or cpu_validation.get("result", {}).get("protocol_identity_contract_passed") is not True
            or cpu_validation.get("source_file_sha256") != expected_source_sha256
            or cpu_validation.get("GPU_units") != 0
            or cpu_validation.get("official_metrics_generated") != 0
            or cpu_validation.get("formal_registration") is not False):
        raise ValueError("Required CPU evidence did not close the official-evaluation contract findings")
    smoke = load(smoke_path)
    image_identity_path = smoke_path.parent / "val2017_image_identity.json"
    image_identity_sha256 = sha(image_identity_path)
    image_identity = load(image_identity_path)
    if (ready.get("val2017_image_identity_receipt_sha256") != image_identity_sha256
            or ready.get("blocker_closure_sha256") != sha(blocker_path)
            or ready.get("cpu_validation_sha256") != sha(cpu_path)
            or ready.get("original_review_sha256") != sha(review_path)
            or smoke.get("state") != "passed" or smoke.get("phase") != "official_eval"
            or smoke.get("source_commit") != manifest["source_commit"]
            or smoke.get("release_sha256") != sha(release_path)
            or smoke.get("phase_config_sha256") != config_sha
            or any(smoke.get(k) != v for k, v in protocol_identity.items())
            or smoke.get("val2017_image_manifest_sha256") != DATA_FILES[VAL_IMAGE_MANIFEST]
            or smoke.get("val2017_image_count") != 5000
            or smoke.get("val2017_image_identity_receipt_sha256") != image_identity_sha256
            or image_identity.get("state") != "passed" or image_identity.get("image_count") != 5000
            or image_identity.get("manifest_sha256") != DATA_FILES[VAL_IMAGE_MANIFEST]
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
    protocol_identity = validate_protocol_identity(manifest)
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
                  **protocol_identity,
                  previous_stage_receipt_sha256=manifest["phase_dependencies"]["s4_200"]["accepted_sha256"],
                  victim_checkpoint_sha256s={k: sha(v) for k, v in checkpoints.items()},
                  val2017_image_manifest_path=(DATA_ROOT / VAL_IMAGE_MANIFEST).as_posix(),
                  val2017_image_manifest_sha256=DATA_FILES[VAL_IMAGE_MANIFEST],
                  val2017_image_count=5000,
                  computation_kind="paired-read-only-official-COCOeval", parameter_updates=False,
                  checkpoint_selection_allowed=False, method_statistics_allowed=False,
                  metric_driven_tuning_allowed=False, official_metrics_disabled=bool(smoke_batches),
                  artifact_ingest_disabled=bool(smoke_batches), cleanup_policy="retain_all", deleted_paths=[])
    status = dict(common, state="RUNNING", started_UTC=now())
    atomic(output / "status.json", status)
    write_once(output / "phase_spec.json", dict(common, config=cfg, dependencies=dep, teacher_acceptance=teacher_receipt))
    try:
        image_identity = validate_val2017_image_identity(
            DATA_ROOT / "metadata/instances_val2017.json", DATA_ROOT / "images/val",
            DATA_ROOT / VAL_IMAGE_MANIFEST, DATA_FILES[VAL_IMAGE_MANIFEST], expected_count=5000)
        image_identity_path = output / "val2017_image_identity.json"
        write_once(image_identity_path, image_identity)
        image_identity_sha256 = sha(image_identity_path)
        ready_identity_sha256 = load(registration)["val2017_image_identity_receipt_sha256"] if not smoke_batches else image_identity_sha256
        if image_identity_sha256 != ready_identity_sha256:
            raise ValueError("Formal val2017 image bytes differ from the reviewed bounded-smoke manifest identity")
        common["val2017_image_identity_receipt_sha256"] = image_identity_sha256
        status.update(val2017_image_identity_state="passed",
                      val2017_image_identity_receipt_sha256=image_identity_sha256,
                      verified_val2017_images=image_identity["image_count"])
        atomic(output / "status.json", status)
        atomic(output / "phase_spec.json", dict(common, config=cfg, dependencies=dep,
                                                 teacher_acceptance=teacher_receipt))
        audits = {}
        for name, checkpoint in checkpoints.items():
            sub = output / name
            sub.mkdir(exist_ok=False)
            predictions, audit = predict(checkpoint, DATA_ROOT / "images/val",
                                         DATA_ROOT / "metadata/instances_val2017.json",
                                         mapping, sub, smoke_batches=smoke_batches,
                                         image_identity_receipt_sha256=image_identity_sha256)
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
                                        predictions, mapping, sub, evaluator["directory"], evaluator["files"],
                                        image_identity_receipt_sha256=image_identity_sha256,
                                        protocol_identity=protocol_identity)
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
            validate_paired_metric_payload(paired, cfg["paired_metrics"])
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
