"""Calculate locked last.pt P/U/S and relative collateral metrics."""
import argparse
import csv
import json
import math
from pathlib import Path


NAMES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(value["AP50_per_class"]) != 20 or len(value["Recall_per_class"]) != 20:
        raise ValueError(f"Incomplete VOC20 metrics: {path}")
    return value


def pus(clean, run):
    p = (clean["AP50_per_class"][14] - run["AP50_per_class"][14]) / clean["AP50_per_class"][14]
    u = sum(run["AP50_per_class"][i] for i in range(20) if i != 14) / 19
    u /= sum(clean["AP50_per_class"][i] for i in range(20) if i != 14) / 19
    s = 2 * p * u / (p + u) if p + u else None
    return {"P": p, "P_percent": 100*p, "U": u, "U_percent": 100*u,
            "S": s, "S_percent": 100*s if s is not None else None}


def collateral(clean, run):
    deltas = {}
    for i, name in enumerate(NAMES):
        if i == 14:
            continue
        base = clean["AP50_per_class"][i]
        deltas[name] = ((base - run["AP50_per_class"][i]) / base) if base else None
    valid = sorted(((name, value) for name, value in deltas.items() if value is not None),
                   key=lambda item: (-item[1], item[0]))
    return {"relative_collateral_damage_by_class": deltas,
            "WCD": valid[0][1], "WCD_class": valid[0][0],
            "TCD3": sum(x[1] for x in valid[:3]) / 3,
            "top_three": valid[:3],
            "bicycle_dog_horse": {x: deltas[x] for x in ("bicycle", "dog", "horse")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-last", required=True)
    parser.add_argument("--b2to-last", required=True)
    parser.add_argument("--safe-last", required=True)
    parser.add_argument("--safe-best", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean, b2to, safe_last, safe_best = map(read,
        (args.clean_last, args.b2to_last, args.safe_last, args.safe_best))
    pu = {"clean_vs_clean": pus(clean, clean), "B2-TO": pus(clean, b2to),
          "B2-TO-Safe-LR_last": pus(clean, safe_last),
          "B2-TO-Safe-LR_best_secondary": pus(clean, safe_best)}
    coll = {"B2-TO": collateral(clean, b2to), "B2-TO-Safe-LR_last": collateral(clean, safe_last)}
    restorations = sorted(((NAMES[i], safe_last["AP50_per_class"][i]-b2to["AP50_per_class"][i])
                           for i in range(20) if i != 14), key=lambda x: (-x[1], x[0]))
    output = Path(args.output)
    if output.is_dir():
        report_dir = output
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_dir = output.parent
    comparison = []
    metric_map = {
        "person AP50": lambda m: m["AP50_per_class"][14],
        "person Recall": lambda m: m["Recall_per_class"][14],
        "non-target mAP50": lambda m: sum(m["AP50_per_class"][i] for i in range(20) if i != 14)/19,
        "all-class mAP50": lambda m: m["mAP50_all"],
        "P": lambda m: pus(clean, m)["P"], "U": lambda m: pus(clean, m)["U"],
        "S": lambda m: pus(clean, m)["S"],
        "WCD": lambda m: collateral(clean, m)["WCD"],
        "TCD3": lambda m: collateral(clean, m)["TCD3"],
    }
    for metric, get in metric_map.items():
        a, b = get(b2to), get(safe_last)
        comparison.append({"Metric": metric, "B2-TO": a, "B2-TO-Safe-LR": b, "Delta": b-a})
    with (report_dir / "b2_to_vs_safe_lr_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Metric", "B2-TO", "B2-TO-Safe-LR", "Delta"))
        writer.writeheader(); writer.writerows(comparison)
    per_class = []
    for i, name in enumerate(NAMES):
        per_class.append({"class": name, "clean_last_AP50": clean["AP50_per_class"][i],
            "B2_TO_last_AP50": b2to["AP50_per_class"][i], "Safe_LR_last_AP50": safe_last["AP50_per_class"][i],
            "delta_Safe_LR_minus_B2_TO": safe_last["AP50_per_class"][i]-b2to["AP50_per_class"][i],
            "clean_last_Recall": clean["Recall_per_class"][i], "B2_TO_last_Recall": b2to["Recall_per_class"][i],
            "Safe_LR_last_Recall": safe_last["Recall_per_class"][i]})
    with (report_dir / "per_class_b2to_vs_safe_lr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys()); writer.writeheader(); writer.writerows(per_class)
    chain = []
    for name, data in (("B2-TO", b2to), ("B2-TO-Safe-LR", safe_last)):
        chain.append({"Method": name, "Strict TSFL": "Yes", "Layer Selection": "Yes/P5",
                      "Safe-BG": "Low-rank" if name.endswith("Safe-LR") else "No",
                      "Safe rank": 8 if name.endswith("Safe-LR") else "-",
                      "Target AP50": data["AP50_per_class"][14],
                      "NT mAP50": sum(data["AP50_per_class"][i] for i in range(20) if i != 14)/19,
                      **pus(clean,data)})
    (report_dir / "ablation_metrics.json").write_text(json.dumps({
        "primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
        "clean_baseline_checkpoint": "clean last.pt reevaluated on Safe-LR isolated clean val",
        "B2_TO_checkpoint": "last.pt reevaluated on same Safe-LR isolated clean val",
        "B2_TO_Safe_LR_primary": safe_last, "B2_TO_Safe_LR_best_secondary": safe_best,
        "clean_last": clean, "B2_TO_last": b2to, "PUS": pu, "collateral_damage": coll,
        "relative_collateral_reduced_vs_B2_TO": coll["B2-TO-Safe-LR_last"]["WCD"] < coll["B2-TO"]["WCD"]
          and coll["B2-TO-Safe-LR_last"]["TCD3"] < coll["B2-TO"]["TCD3"],
        "protection_utility_tradeoff_improved_vs_B2_TO": pu["B2-TO-Safe-LR_last"]["S"] > pu["B2-TO"]["S"],
        "per_class_restorations_Safe_LR_minus_B2_TO_best": restorations,
        "most_restored_non_target_classes": restorations[:5],
        "most_declined_non_target_classes": sorted(restorations, key=lambda x: (x[1], x[0]))[:5],
        "bicycle_dog_horse_delta": {name: safe_last["AP50_per_class"][i]-b2to["AP50_per_class"][i]
                                    for i,name in enumerate(NAMES) if name in ("bicycle","dog","horse")},
        "comparison_rows": comparison, "method_chain": chain,
        "interpretation": "Descriptive comparison of matched clean-val metrics; this does not by itself establish a causal mechanism."}, indent=2), encoding="utf-8")
    with (report_dir / "method_chain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=chain[0].keys()); writer.writeheader(); writer.writerows(chain)
    print(json.dumps({"PUS": pu, "WCD_TCD3": coll,
        "most_restored": restorations[:5], "most_declined": sorted(restorations,key=lambda x:(x[1],x[0]))[:5]}, indent=2))


if __name__ == "__main__":
    main()
