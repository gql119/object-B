"""Matched clean-val last.pt comparison of Safe-Soft05 and KD-ObjBal."""

import argparse
import csv
import json
from pathlib import Path

from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES, collateral, pus, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-last", required=True)
    parser.add_argument("--safe-last", required=True)
    parser.add_argument("--kd-last", required=True)
    parser.add_argument("--kd-best", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean, safe, kd, kd_best = [read(path) for path in
                              (args.clean_last, args.safe_last, args.kd_last, args.kd_best)]
    names = [tuple(x.get("class_names", NAMES)) for x in (clean, safe, kd, kd_best)]
    if len(set(names)) != 1 or names[0] != NAMES:
        raise ValueError("Comparison class order differs from VOC20")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    def scores(run):
        return {
            "person AP50": run["AP50_per_class"][14],
            "person Recall": run["Recall_per_class"][14],
            "NT mAP50": run["mAP50_non_target"],
            "all mAP50": run["mAP50_all"],
            **{key: pus(clean, run)[key] for key in ("P", "U", "S")},
            **{key: collateral(clean, run)[key] for key in ("WCD", "TCD3")},
        }
    parent, candidate = scores(safe), scores(kd)
    rows = [{"Metric": metric, "Safe-Soft05": parent[metric],
             "Safe-Soft05-KD-ObjBal": candidate[metric],
             "Delta": candidate[metric] - parent[metric]}
            for metric in parent]
    with (output / "safe_soft05_vs_kd_objbal.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_class = [{"class": name, "Safe-Soft05_AP50": safe["AP50_per_class"][i],
                  "KD-ObjBal_AP50": kd["AP50_per_class"][i],
                  "Delta_AP50": kd["AP50_per_class"][i] - safe["AP50_per_class"][i],
                  "Safe-Soft05_Recall": safe["Recall_per_class"][i],
                  "KD-ObjBal_Recall": kd["Recall_per_class"][i],
                  "Delta_Recall": kd["Recall_per_class"][i] - safe["Recall_per_class"][i]}
                 for i, name in enumerate(NAMES)]
    with (output / "voc20_per_class.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    damage = collateral(clean, kd)
    report = {"primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
              "comparison": rows, "per_class": per_class,
              "safe_soft05": parent, "kd_objbal": candidate,
              "kd_best_secondary": scores(kd_best),
              "current_worst_three_classes": [name for name, _ in damage["top_three"]],
              "bicycle_dog_horse": {name: next(r for r in per_class if r["class"] == name)
                                     for name in ("bicycle", "dog", "horse")},
              "interpretation": "Descriptive matched-clean-val comparison; no causal claim."}
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
