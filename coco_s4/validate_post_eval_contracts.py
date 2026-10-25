"""CPU-only contracts for paired COCO evaluation and final report derivation."""
import copy
import json
from pathlib import Path

from .coco_eval import PREDICT_ARGS
from .evaluation import paired_metrics
from .phase_final_report import VOC_ALIASES, build_report, shared_voc_classes, validate_config as validate_report
from .phase_official_eval import validate_config as validate_official_eval
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
    if pair["P"] != .5 or abs(pair["U"] - .8) > 1e-12:
        raise AssertionError("Signed paired target/non-target metrics changed")
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
                          report_rows=len(built["metrics"]), shared_voc_names=len(shared),
                          GPU_units=0, dataset_images_opened=0, official_metrics_generated=0,
                          formal_registration=False), sort_keys=True))


if __name__ == "__main__": main()
