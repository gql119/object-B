# B2-T experiment results (2026-09-26)

B2-T adds TSFL-selected semantic channels (48 of 64; fraction 0.75; specificity weight 1.0) to the frozen-reference/adaptive-learner method. Orthogonality weight is 0 for this T run. The final victim is randomly initialized YOLOv8n (`pretrained=false`), seed 0, batch 36, 640-pixel inputs, trained for 200 epochs. Evaluation uses the 4,952-image clean VOC validation split.

## Aggregate clean-validation metrics

AP50 and recall values are percentages.

| Metric | B2 | B2-T | B2-T − B2 (percentage points) |
|---|---:|---:|---:|
| mAP50_target | 2.2018 | 3.3012 | +1.0994 |
| mAP50_non_target | 69.4336 | 69.0503 | -0.3833 |
| mAP50_all | 66.0720 | 65.7629 | -0.3091 |
| Recall_target | 0.0000 | 0.0000 | +0.0000 |

B2-T target (person) AP50 is 3.3012%, above B2 by 1.0994 pp; non-target mAP50 is 69.0503%, lower than B2 by 0.3833 pp. Thus T did not improve the target/non-target trade-off relative to B2 in this single run.

## Per-class clean-validation metrics

AP50 and recall are percentages; B2-T AP50 deltas are percentage points.

| ID | Class | B2-T AP50 | B2-T Recall | B2 AP50 | Δ AP50 |
|---:|---|---:|---:|---:|---:|
| 0 | aeroplane | 83.6290 | 70.1754 | 83.9969 | -0.3680 |
| 1 | bicycle | 66.3479 | 56.0831 | 64.6043 | +1.7436 |
| 2 | bird | 65.1762 | 59.4771 | 65.1033 | +0.0729 |
| 3 | boat | 67.1125 | 57.7947 | 68.6499 | -1.5374 |
| 4 | bottle | 47.1299 | 41.3646 | 47.5374 | -0.4076 |
| 5 | bus | 81.0312 | 72.4775 | 83.2685 | -2.2373 |
| 6 | car | 88.1333 | 77.6020 | 88.6588 | -0.5254 |
| 7 | cat | 80.3870 | 75.4190 | 80.9259 | -0.5389 |
| 8 | chair | 52.4450 | 43.9936 | 53.2546 | -0.8096 |
| 9 | cow | 68.2164 | 65.5422 | 68.7577 | -0.5413 |
| 10 | diningtable | 67.9550 | 56.0388 | 68.0675 | -0.1125 |
| 11 | dog | 57.4977 | 66.8712 | 58.7493 | -1.2516 |
| 12 | horse | 73.1722 | 70.3780 | 72.3732 | +0.7990 |
| 13 | motorbike | 70.4835 | 68.9231 | 72.0110 | -1.5275 |
| 14 | person | 3.3012 | 0.0000 | 2.2018 | +1.0994 |
| 15 | pottedplant | 44.9437 | 32.7083 | 44.1729 | +0.7709 |
| 16 | sheep | 71.7329 | 69.8347 | 71.9600 | -0.2271 |
| 17 | sofa | 70.3734 | 65.8806 | 71.8003 | -1.4269 |
| 18 | train | 82.7400 | 79.0780 | 83.4711 | -0.7311 |
| 19 | tvmonitor | 73.4494 | 60.3354 | 71.8757 | +1.5737 |

## Run verification and provenance

Noise optimization completed 24 epochs / 9144 steps in 3.606 hours; exit code 0. The 16,551-image poisoned training split was materialized and passed formal full-data preflight; all 4,952 clean validation images were copied byte-for-byte. The preflight checked stems and labels, configuration/code/artifact hashes, full person-box pixel coverage, zero perturbation outside support, 16/255 global and 8/255 overlap budgets, and unchanged non-target-only regions. The victim completed 200/200 epochs with exit code 0; both best and last checkpoints were produced, but weights are not included here.

See `metrics.json` for exact metrics, hashes and the B2 comparison. `provenance.json` and `preflight_summary.json` record the run state and preflight receipts; `results.csv` records the 200-epoch training history.

## Limitations

This is one seed and has no protocol-matched clean victim baseline. The comparison to B2 is descriptive, not a controlled causal estimate. AP50 changes alone do not prove background absorption.
