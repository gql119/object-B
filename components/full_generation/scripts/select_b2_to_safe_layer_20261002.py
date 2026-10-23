"""Select one B2-TO-Safe feature level from frozen clean-train statistics only."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


LAYERS = ("P3", "P4", "P5")
EPS = 1e-12


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    xx = [v - mx for v in x]
    yy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in xx) * sum(v * v for v in yy))
    if denom <= EPS:
        raise ValueError("Correlation is undefined for a constant sensitivity vector")
    return sum(a * b for a, b in zip(xx, yy)) / denom


def ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for pos in range(start, stop):
            result[ordered[pos]] = rank
        start = stop
    return result


def audit_layer(name, item):
    arrays = item["sensitivity_arrays"]
    target = arrays["target_normalized"]
    worst = arrays["worst_nt_normalized"]
    if len(target) != item["channels"] or len(worst) != len(target):
        raise ValueError(f"{name} has inconsistent channel dimensions")
    if any(not math.isfinite(v) for v in target + worst):
        raise ValueError(f"{name} has nonfinite sensitivity")
    matching = [row for row in item["gate_rows"]
                if row["target_top_fraction"] == .40 and row["worst_nt_quantile"] == .60]
    if len(matching) != 1:
        raise ValueError(f"{name} lacks the common Top40%+Q60 gate")
    gate = matching[0]
    eligible = [i for i, (st, nt) in enumerate(zip(target, worst))
                if st >= gate["target_threshold_normalized"] - 1e-7
                and nt <= gate["worst_nt_threshold_normalized"] + 1e-7]
    if eligible != gate["eligible_indices"] or len(eligible) != gate["eligible_count"]:
        raise ValueError(f"{name} gate membership differs from frozen scan")
    ratio = {i: (target[i] + EPS) / (worst[i] + EPS) for i in eligible}
    values = list(ratio.values())
    n = len(eligible)
    ls = statistics.mean(math.log(v) for v in values) if n >= 4 else None
    mean_ratio = statistics.mean(values) if values else None
    selected_n = min(math.ceil(.15 * len(target)), n)
    ranked = sorted(eligible, key=lambda i: (-ratio[i], i))
    selected = sorted(ranked[:selected_n])
    return {
        "layer": name, "feature_layer": item["layer"], "channels": len(target),
        "candidate_count": n, "mean_target": statistics.mean(target[i] for i in eligible) if n else None,
        "mean_worst_nt": statistics.mean(worst[i] for i in eligible) if n else None,
        "mean_ratio": mean_ratio, "median_ratio": statistics.median(values) if values else None,
        "min_ratio": min(values) if values else None, "max_ratio": max(values) if values else None,
        "pearson_rho": pearson(target, worst),
        "spearman_rho": pearson(ranks(target), ranks(worst)),
        "LS": ls, "valid": bool(n >= 4 and mean_ratio > 1 and ls > 0),
        "eligible_indices": eligible, "selected_indices": selected,
        "selected_mean_target": statistics.mean(target[i] for i in selected) if selected else None,
        "selected_mean_worst_nt": statistics.mean(worst[i] for i in selected) if selected else None,
        "selected_mean_ratio": statistics.mean(ratio[i] for i in selected) if selected else None,
        "selected_count": len(selected), "selected_fraction": len(selected) / len(target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    scan = Path(args.scan).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(scan.read_text(encoding="utf-8"))
    if data["state"] != "complete" or not data["frozen_state_unchanged"] or data["validation_used"]:
        raise ValueError("Layer selection requires a completed frozen clean-train scan")
    if data["noise_updated"] or data["victim_trained"] or tuple(data["layers"]) != LAYERS:
        raise ValueError("Invalid layer scan identity")
    rows = [audit_layer(name, data["layers"][name]) for name in LAYERS]
    valid = [row for row in rows if row["valid"]]
    tie = False
    selected = None
    if valid:
        selected = max(valid, key=lambda row: row["LS"])
        p4 = rows[1]
        p5 = rows[2]
        if p4["valid"] and p5["valid"] and selected["layer"] in ("P4", "P5"):
            tie = abs(p4["LS"] - p5["LS"]) <= .05 * max(abs(p4["LS"]), abs(p5["LS"]), EPS)
            if tie:
                selected = p4
    output.mkdir(parents=True, exist_ok=False)
    fields = ("layer", "channels", "candidate_count", "mean_target", "mean_worst_nt",
              "mean_ratio", "median_ratio", "min_ratio", "max_ratio", "pearson_rho",
              "spearman_rho", "LS", "valid")
    with (output / "strict_tsfl_layer_audit.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    receipt = {
        "state": "selected" if selected else "no_valid_layer",
        "selected_layer": selected["layer"] if selected else None,
        "selected_feature_layer": selected["feature_layer"] if selected else None,
        "selection_reason": "highest_eligible_LS_with_P4_P5_spatial_tie_break" if tie else
                            "highest_eligible_LS" if selected else "no_layer_met_all_three_gates",
        "tie_break_triggered": tie,
        "gate": {"target_top_fraction": .40, "worst_nt_quantile": .60,
                 "sensitivity_worst_k": 3, "selected_fraction_cap": .15,
                 "minimum_eligible_channels": 4, "minimum_mean_ratio_exclusive": 1,
                 "minimum_LS_exclusive": 0},
        "layers": {row["layer"]: row for row in rows},
        "selected_channel_indices": selected["selected_indices"] if selected else None,
        "selected_channel_count": selected["selected_count"] if selected else 0,
        "source_scan_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(),
        "clean_train_manifest_sha256": data["clean_train_manifest_sha256"],
        "selection_uses_victim_metrics": False,
    }
    name = "layer_selection.json" if selected else "layer_selection_failure_report.json"
    (output / name).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"selected_layer": receipt["selected_layer"], "tie_break": tie,
                      "layers": {row["layer"]: {"N": row["candidate_count"],
                           "mean_ratio": row["mean_ratio"], "rho": row["pearson_rho"],
                           "LS": row["LS"], "valid": row["valid"]} for row in rows}}, sort_keys=True))


if __name__ == "__main__":
    main()
