"""Compare four last.pt methods on one clean VOC validation copy."""
import argparse
import csv
import json
import math
from pathlib import Path

from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES, collateral, pus, read


def run():
    parser = argparse.ArgumentParser()
    for name in ("clean", "b2to", "safe_lr", "r4", "soft05", "r4_best", "soft05_best"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = {name: read(getattr(args, name)) for name in
              ("clean", "b2to", "safe_lr", "r4", "soft05", "r4_best", "soft05_best")}
    if len({tuple(item["class_names"]) for item in source.values()}) != 1:
        raise ValueError("VOC20 class order differs among comparisons")
    clean = source["clean"]
    rows = []
    definitions = (("B2-TO", "b2to", None, "None", None),
                   ("Safe-LR-R8", "safe_lr", 8, "Hard", 1.0),
                   ("Safe-R4", "r4", 4, "Hard", 1.0),
                   ("Safe-Soft05", "soft05", 8, "Soft", 0.5))
    for method, key, rank, removal, alpha in definitions:
        value = source[key]
        scored = pus(clean, value)
        damage = collateral(clean, value)
        rows.append({"Method": method, "Rank": rank, "Removal": removal, "alpha": alpha,
                     "Target AP50": value["AP50_per_class"][14],
                     "person Recall": value["Recall_per_class"][14],
                     "NT mAP50": value["mAP50_non_target"],
                     "all mAP50": value["mAP50_all"],
                     "P": scored["P"], "U": scored["U"], "S": scored["S"],
                     "WCD": damage["WCD"], "WCD_class": damage["WCD_class"],
                     "TCD3": damage["TCD3"]})
    by_method = {row["Method"]: row for row in rows}
    r8, r4, soft, b2to = (by_method[name] for name in
                             ("Safe-LR-R8", "Safe-R4", "Safe-Soft05", "B2-TO"))
    findings = {
        "R4_target_suppression_stronger_than_R8": r4["Target AP50"] < r8["Target AP50"],
        "R4_NT_utility_above_B2_TO": r4["U"] > b2to["U"],
        "Soft05_target_suppression_stronger_than_R8": soft["Target AP50"] < r8["Target AP50"],
        "Soft05_NT_utility_above_B2_TO": soft["U"] > b2to["U"],
        "highest_selectivity_S": max(rows, key=lambda row: row["S"])["Method"],
        "causal_mechanism_established": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "four_method_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    class_rows = []
    for index, name in enumerate(NAMES):
        item = {"class": name}
        for method, key, *_ in definitions:
            item[f"{method}_AP50"] = source[key]["AP50_per_class"][index]
            item[f"{method}_Recall"] = source[key]["Recall_per_class"][index]
        class_rows.append(item)
    with (output / "voc20_per_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(class_rows[0]))
        writer.writeheader(); writer.writerows(class_rows)
    report = {"primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
              "same_clean_validation_required": True,
              "rows": rows, "per_class": class_rows,
              "best_secondary": {name: source[name] for name in ("r4_best", "soft05_best")},
              "findings": findings,
              "interpretation": "Descriptive matched-clean-val comparison; no causal mechanism inferred."}
    (output / "ablation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "findings": findings}, indent=2))


if __name__ == "__main__":
    run()
