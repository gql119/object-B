# B2-TO E8 half-budget ablation: final report

## Run and validation

- Run: `noise_full_b2_to_eps8_overlap4_seed0_20260927` → `victim_b2_to_eps8_overlap4_seed0_20260927`.
- Epsilon changed from 16/255 to 8/255. Existing `overlap_scale=0.5` gives 8/255 → 4/255 in person/non-target-box overlaps. Config diff permits only this epsilon and the isolated run root to differ from B2-TO.
- Noise optimization: 24 epochs, 9,144 steps, 13,065.945 s (3.6294 h), seed 0; config SHA-256 `597871ef9dc8e8044bd64659feca4efb3506e38575cc3cc18ab44aa20a722790`.
- Materialization: 16,551 train images. The preflight and paired audit completed before victim training; pipeline, noise, materialization, and victim exit codes are all 0.
- Victim: randomly initialized YOLOv8n, `pretrained=false`, seed 0, batch 36, image size 640, 200 epochs; `results.csv` contains header + 200 epoch rows. Clean VOC validation has 4,952 images and was copied byte-for-byte. Best and last checkpoints exist on the server; neither is included in this evidence bundle.
- The output was admitted after the five-step production smoke and one independent `scientifically_correct` review. Official metrics and artifact ingestion were disabled during smoke.

## Clean validation metrics

Values are AP50 and Recall. `Δ AP50` is E8 minus the matched B2-TO run in percentage points; this is an ablation comparison, not a clean-trained baseline comparison.

| Class | E8 AP50 | E8 Recall | Δ AP50 vs B2-TO |
|---|---:|---:|---:|

| aeroplane | 0.840488 | 0.709561 | +0.199 pp |
| bicycle | 0.622913 | 0.501484 | -2.563 pp |
| bird | 0.665063 | 0.581699 | -0.258 pp |
| boat | 0.655314 | 0.558935 | -1.223 pp |
| bottle | 0.497708 | 0.409382 | -1.428 pp |
| bus | 0.821264 | 0.704225 | -0.576 pp |
| car | 0.877980 | 0.752706 | -0.188 pp |
| cat | 0.800026 | 0.751397 | -0.443 pp |
| chair | 0.522573 | 0.421958 | -1.786 pp |
| cow | 0.681318 | 0.594262 | +0.087 pp |
| diningtable | 0.682596 | 0.527277 | +0.457 pp |
| dog | 0.570832 | 0.672802 | -1.252 pp |
| horse | 0.713834 | 0.698276 | -1.921 pp |
| motorbike | 0.733860 | 0.695385 | +3.609 pp |
| person | 0.110228 | 0.000693 | +6.869 pp |
| pottedplant | 0.457201 | 0.320998 | +2.812 pp |
| sheep | 0.734685 | 0.681818 | +1.123 pp |
| sofa | 0.695188 | 0.644351 | -0.334 pp |
| train | 0.847963 | 0.794326 | +0.040 pp |
| tvmonitor | 0.744484 | 0.582329 | +0.361 pp |
## Aggregate metrics and interpretation

| Metric | B2-TO (16/8 budget) | E8 (8/4 budget) | E8 − B2-TO |
|---|---:|---:|---:|
| person AP50 | 0.041540 | 0.110228 | +6.869 pp |
| person Recall | 0.000000 | 0.000693 | +0.069 pp |
| 19-class non-target mean AP50 | 0.694639 | 0.692910 | −0.173 pp |
| all-class mAP50 | 0.661984 | 0.663776 | +0.179 pp |

On the paired 256-image deterministic audit (seed 0; same 256 stems, 45,507,913 spatial pixels), E8 respected decoded uint8 bounds: global/person non-overlap max 8/255, overlap max 4/255, outside-person max 0. The changed spatial-pixel ratio was identical at 36.3014%. Despite halving the allowed budget, mean absolute delta ratios were 0.8513 globally, 0.8528 on person-only pixels, and 0.8389 on overlap pixels; the E8/B2-TO L2 norm ratio was 0.7530. Saturation at the new cap rose 2.869× on person-only pixels and 2.576× on overlaps. Thus the optimizer used more of the smaller permitted range; realized average perturbation did not halve.

Compared per class, bicycle had the largest non-target AP50 drop (−2.563 pp), followed by horse (−1.921 pp) and chair (−1.786 pp). Motorbike (+3.609 pp) and pottedplant (+2.812 pp) had the largest non-target AP50 increases. Person AP50 rose by 6.869 pp, so the half-budget run suppressed person less than B2-TO while leaving aggregate non-target AP50 nearly unchanged (−0.173 pp).

No matched clean-trained, random-init victim metrics were found in the bounded experiment search. Therefore P/U/S and WCD/TCD3 relative to clean training are unavailable; B2-TO is the matched poisoned comparator, not a clean baseline. Results do not establish causal background absorption. This is one seed, so it does not estimate run-to-run variance.

## Provenance

- Candidate branch: `codex/b2-to-e8-ablation-20260927`
- Candidate commit: `0cce7d6e787220fc4a4fd3b30c68bf4c61fc68c5`
- Independent review: `PRERUN-REVIEW-1.md`, verdict `scientifically_correct`.
- Paired audit files: `perturbation_budget_audit.json`, `perturbation_budget_comparison.json`.
- Clean validation metrics: `clean_val_metrics.json`.
- Victim run specification/status/results: `run_spec.json`, `victim_status.json`, `results.csv`.
- No weights, images, or dataset files were copied into this evidence folder.
