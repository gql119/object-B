"""CPU-only contracts for paired COCO evaluation and final report derivation."""
import copy
import hashlib
import json
from pathlib import Path

from .coco_eval import PREDICT_ARGS, validate_val2017_image_identity
from .evaluation import PAIRED_METRIC_KEYS, paired_metrics, validate_paired_metric_payload
from .phase_final_report import VOC_ALIASES, build_report, shared_voc_classes, validate_config as validate_report
from .phase_official_eval import (validate_config as validate_official_eval,
                                  validate_protocol_identity)
from .dimension_protocol import PROTOCOL as DIMENSION_PROTOCOL
from .workflow_admission import POLICY as RISK_ESTIMATOR_POLICY
from .runtime import ROOT, load


def rows():
    result = []
    for i in range(80):
        result.append(dict(yolo_class_id=i, category_id=i + 1, name=f"class{i}",
                           AP50=.8 if i == 0 else .5, AP75=.7, AP50_95=.6,
                           Recall50=.75, Precision50=.9))
    return result


def main():
    official = load(ROOT / "coco_s4/configs/official_eval.json")
    report = load(ROOT / "coco_s4/configs/final_report.json")
    validate_official_eval(official); validate_report(report)
    mutations = []
    for key, value in (("checkpoint_selection_allowed", True), ("image_count", 4952),
                       ("target_class_id", 14), ("evaluator_version", "other")):
        bad = copy.deepcopy(official); bad[key] = value
        try: validate_official_eval(bad)
        except ValueError: mutations.append(key)
        else: raise AssertionError("official-evaluation mutation accepted: " + key)
    clean, s4, teacher = rows(), rows(), rows()
    s4[0]["AP50"] = .4
    for row in s4[1:]: row["AP50"] = .4
    pair = paired_metrics(clean, s4, target=0)
    if (pair["P"] != .5 or abs(pair["U"] - .8) > 1e-12
            or pair["absolute_clean_harm"] != .4
            or set(k for k in pair if k in PAIRED_METRIC_KEYS) != set(official["paired_metrics"])):
        raise AssertionError("Signed paired target/non-target metrics changed")
    validate_paired_metric_payload(pair, official["paired_metrics"])
    missing_absolute = dict(pair); missing_absolute.pop("absolute_clean_harm")
    try: validate_paired_metric_payload(missing_absolute, official["paired_metrics"])
    except ValueError: pass
    else: raise AssertionError("Missing declared absolute clean harm was accepted")
    for field, value in (("absolute_clean_harm_formula", "clean minus S4"),
                         ("raw_signed_no_clip", False), ("zero_denominator", 0)):
        bad_pair = dict(pair, **{field: value})
        try: validate_paired_metric_payload(bad_pair, official["paired_metrics"])
        except ValueError: pass
        else: raise AssertionError("Invalid paired metric metadata was accepted: " + field)
    protocol_identity = validate_protocol_identity(dict(
        risk_estimator_policy=RISK_ESTIMATOR_POLICY,
        dimension_protocol=DIMENSION_PROTOCOL,
        strict_Original_S4_equivalence_claim=False))
    bad_protocol = dict(protocol_identity, strict_Original_S4_equivalence_claim=True)
    try: validate_protocol_identity(bad_protocol)
    except ValueError: pass
    else: raise AssertionError("Strict Original S4 equivalence was incorrectly accepted")
    fixture_root = ROOT / "execution/COCO_POST_EVAL_IDENTITY_FIXTURE_20261029"
    annotation_rows = []; manifest_rows = []; payloads = {}
    for image_id, filename, width, height, payload in (
            (1, "000000000001.jpg", 2, 3, b"fixture-image-one"),
            (2, "000000000002.jpg", 4, 5, b"fixture-image-two")):
        payloads[filename] = payload
        annotation_rows.append(dict(id=image_id, file_name=filename, width=width, height=height))
        manifest_rows.append(dict(id=image_id, file_name=filename, width=width, height=height,
                                  image_bytes=len(payload), image_sha256=hashlib.sha256(payload).hexdigest()))
    images = fixture_root / "val"
    annotation_path = fixture_root / "instances.json"
    manifest_path = fixture_root / "images.jsonl"
    expected_annotation = dict(images=annotation_rows)
    expected_manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    if fixture_root.exists():
        if (not images.is_dir() or not annotation_path.is_file() or not manifest_path.is_file()
                or json.loads(annotation_path.read_text(encoding="utf-8")) != expected_annotation
                or manifest_path.read_text(encoding="utf-8") != expected_manifest
                or any((images / name).read_bytes() != data for name, data in payloads.items())):
            raise ValueError("Retained image-identity fixture differs; inspect it without overwriting")
    else:
        images.mkdir(parents=True)
        for name, data in payloads.items(): (images / name).write_bytes(data)
        annotation_path.write_text(json.dumps(expected_annotation), encoding="utf-8")
        manifest_path.write_text(expected_manifest, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    identity = validate_val2017_image_identity(annotation_path, images, manifest_path,
                                              manifest_sha, expected_count=2)
    if identity["state"] != "passed" or identity["image_count"] != 2:
        raise AssertionError("Pinned image identity fixture did not pass")
    altered = images / "000000000001.jpg"
    same_name_same_size_substitution_rejected = False
    try:
        substituted_payload = b"fixture-image-bad"
        if len(substituted_payload) != len(payloads[altered.name]):
            raise AssertionError("Substitution fixture must keep the pinned byte count")
        altered.write_bytes(substituted_payload)
        try: validate_val2017_image_identity(annotation_path, images, manifest_path,
                                             manifest_sha, expected_count=2)
        except ValueError: pass
        else: raise AssertionError("Substituted image bytes passed the pinned identity check")
        same_name_same_size_substitution_rejected = True
    finally:
        altered.write_bytes(payloads[altered.name])
    built = build_report(teacher, clean, s4, [dict(yolo_class_id=i, category_id=i+1, name=n)
                      for i, n in enumerate(["airplane", "person"] + [f"x{i}" for i in range(78)])],
                      dict(PSNR_db=dict(mean=30.), LPIPS=dict(value=None, state="not_validated")),
                      provenance=dict(source_commit="fixture",calibration_dimension=dict(
                          selected_channels=list(range(39)),selected_channel_count=39,
                          total_feature_channels=256,eligible_channel_count=78,
                          dimension_protocol='coco80-strict-tsfl-selected-dimension-v1')))
    if (len(built["metrics"]) != 80 or built["paired_summary"]["P"] != .5
            or built["spatial_and_perceptual_diagnostics"]["LPIPS"]["value"] is not None
            or built["causal_claims"] or built["significance_claims"]):
        raise AssertionError("Final report did not preserve full metrics/null LPIPS/noncausal scope")
    shared = shared_voc_classes([dict(name=x) for x in VOC_ALIASES])
    if len(shared) != 20:
        raise AssertionError("COCO/VOC class-name alignment must cover all 20 shared names")
    print(json.dumps(dict(state="passed", phases=["official_eval", "final_report"],
                          invalid_official_eval_contracts_rejected=len(mutations), paired_metric_cases=1,
                          image_identity_fixture="passed",
                          same_name_same_size_substitution_rejected=same_name_same_size_substitution_rejected,
                          missing_metric_rejected=True, protocol_identity_contract_passed=True,
                          report_rows=len(built["metrics"]), shared_voc_names=len(shared),
                          GPU_units=0, dataset_images_opened=0, official_metrics_generated=0,
                          formal_registration=False), sort_keys=True))


if __name__ == "__main__": main()
