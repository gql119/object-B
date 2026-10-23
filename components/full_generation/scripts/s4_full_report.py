"""S4 vs S4-TFSS-Full reporting; only complete last@200 enters this report."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES, pus, collateral


def spatial_summary(rows):
    """Per-image distributions and area-pooled RMS; never infer AP mediation."""
    import math
    def summarize(group):
        answer={"images":len(group)}
        for key in ("overlap_area_ratio","overlap_mean_abs_delta","overlap_RMS_delta",
                    "overlap_Linf_delta","non_overlap_RMS_delta","person_RMS_delta",
                    "overlap_mean_W_U","overlap_mean_Q_t","overlap_mean_R_nt","PSNR_db"):
            values=sorted(float(r[key]) for r in group if r.get(key) is not None)
            if not values:answer[key]=None;continue
            def quantile(p):
                x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
                return values[lo]+(values[hi]-values[lo])*(x-lo)
            answer[key]=dict(count=len(values),mean=sum(values)/len(values),min=values[0],
                q10=quantile(.1),median=quantile(.5),q90=quantile(.9),max=values[-1])
        mass=sum(r.get("overlap_area_pixels",0) for r in group)
        energy=sum((r.get("overlap_RMS_delta") or 0)**2*r.get("overlap_area_pixels",0) for r in group)
        answer["pooled_overlap_RMS_delta"]=math.sqrt(energy/mass) if mass else None
        return answer
    groups={"all_target_images":rows,"with_GT_overlap":[r for r in rows if r["overlap_area_pixels"]>0],
            "without_GT_overlap":[r for r in rows if r["overlap_area_pixels"]==0]}
    result={k:summarize(v) for k,v in groups.items()}
    result["baseline_S4"]=summarize([r["baseline_S4_spatial"] for r in rows])
    result["NT_class_overlap"]={}
    classes=sorted({k for row in rows for k in row["overlap_by_NT_class"]},key=int)
    for cls in classes:
        values=[r["overlap_by_NT_class"][cls] for r in rows
                if cls in r["overlap_by_NT_class"] and r["overlap_by_NT_class"][cls]["area_pixels"]>0]
        mass=sum(r["area_pixels"] for r in values)
        result["NT_class_overlap"][cls]=dict(images=len(values),area_pixels=mass,
            pooled_RMS_delta=math.sqrt(sum(r["RMS_delta"]**2*r["area_pixels"] for r in values)/mass) if mass else None)
    result["units"]="delta in [0,1] RGB units; image summaries unweighted across images; pooled RMS area-weighted; class masks may overlap"
    return result


def compare(baseline, candidate):
    for record in (baseline["metrics"]["Clean"], baseline["metrics"]["S4"], candidate):
        for key in ("AP50", "AP75", "AP50_95", "Recall", "Precision"):
            values = record[key+"_per_class"]
            if len(values) != 20 or any(not __import__("math").isfinite(float(x)) or not 0 <= x <= 1 for x in values):
                raise ValueError("Incomplete/nonfinite fractional VOC20 metrics")
    clean, old = baseline["metrics"]["Clean"], baseline["metrics"]["S4"]
    scores = {role: dict(pus(clean, record), **collateral(clean, record))
              for role, record in (("S4", old), ("S4_TFSS_FULL_OVERLAP", candidate))}
    deltas = {key: candidate[key]-value for key, value in old.items()
              if key.startswith(("mAP", "Recall_", "Precision_"))
              and isinstance(value, (int, float)) and isinstance(candidate.get(key), (int, float))}
    per_class = []
    for i, name in enumerate(NAMES):
        row = dict(class_id=i, class_name=name)
        for key in ("AP50", "AP75", "AP50_95", "Recall", "Precision"):
            a, b = old[key+"_per_class"][i], candidate[key+"_per_class"][i]
            row.update({key+"_S4": a, key+"_Full": b, key+"_Full_minus_S4": b-a})
        per_class.append(row)
    unfavorable = sorted((r for r in per_class if r["class_id"] != 14),
                         key=lambda r: (r["AP50_Full_minus_S4"], r["class_id"]))
    return dict(method="S4_TFSS_FULL_OVERLAP", primary="last.pt@200", secondary="best.pt",
        scores=scores, aggregate_delta= deltas, per_class=per_class,
        unfavorable_non_target_classes=unfavorable, single_seed_descriptive=True,
        overlap_effect_causal_attribution=False,
        interpretation="Spatial strength changes are measured; final AP differences alone do not identify overlap as the sole causal mediator.")


def finalize(output, root, commit, lock):
    from scripts.run_s4_tfss_pipeline import sha
    output, root = Path(output), Path(root)
    state=json.loads((output/"pipeline_status.json").read_text())
    if state.get("source_commit")!=commit or state.get("support_mode")!="bbox_full_target":
        raise ValueError("Training/report source or support identity mismatch")
    if json.loads((output/"config_lock_snapshot.json").read_text())!=lock:
        raise ValueError("Reporting lock differs from training lock")
    baseline_path = root/lock["baseline"]["comparison_path"]
    if sha(baseline_path) != lock["baseline"]["comparison_sha256"]:
        raise ValueError("Historical S4 comparison changed")
    baseline = json.loads(baseline_path.read_text())
    completion = json.loads((output/"victim_completion.json").read_text())
    if completion["epochs"] != 200 or completion["rows"] != 200:
        raise ValueError("Incomplete victim cannot enter official results")
    metrics = json.loads((output/"S4_TFSS_FULL_OVERLAP_evaluation.json").read_text())
    last = output/"victim_run/victim/weights/last.pt"
    if metrics["checkpoint_sha256"] != sha(last) or completion["last_sha256"] != sha(last):
        raise ValueError("last@200 checkpoint identity mismatch")
    integrity = json.loads((output/"dataset_integrity.json").read_text())
    if integrity.get("state") != "passed" or integrity.get("support_mode") != "bbox_full_target":
        raise ValueError("Missing full-target data audit")
    result = compare(baseline, metrics)
    result.update(source_commit=commit, config_sha256=sha(output/"config_lock_snapshot.json"),
                  checkpoint_sha256=sha(last), dataset_manifest_sha256=sha(output/"materialization_manifest.json"),
                  baseline_comparison_sha256=sha(baseline_path))
    (output/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (output/"S4_vs_Full_VOC20.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(result["per_class"][0]));writer.writeheader();writer.writerows(result["per_class"])
    text=["# S4 vs S4-TFSS-Full", "", "Full seed0/200-epoch last checkpoint; single-seed descriptive comparison.", "",
          "| Metric | S4 | Full | Full minus S4 |", "|---|---:|---:|---:|"]
    for key in ("mAP50_target","mAP50_non_target","mAP75_non_target","mAP50_95_non_target","mAP50_all","Recall_target"):
        old=baseline["metrics"]["S4"][key];new=metrics[key]
        text.append(f"| {key} | {old:.9f} | {new:.9f} | {new-old:+.9f} |")
    for key in ("S", "WCD", "TCD3"):
        a=result["scores"]["S4"][key];b=result["scores"]["S4_TFSS_FULL_OVERLAP"][key]
        text.append(f"| {key} | {a:.9f} | {b:.9f} | {b-a:+.9f} |")
    text.extend(["", "See S4_vs_Full_VOC20.csv for all classes and dataset_integrity.json for GT-geometric overlap strength and class-conditioned diagnostics.",
                 "A lower Person AP with worse NT AP/WCD does not show that TFSS replaces overlap protection. AP changes alone do not prove background absorption or mediation by overlap.", ""])
    (output/"S4_VS_TFSS_FULL_REPORT.md").write_text("\n".join(text))


if __name__ == "__main__":
    import argparse
    import subprocess
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--lock",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    finalize(Path(args.output),root,commit,json.loads(Path(args.lock).read_text()))
