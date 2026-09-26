# B2 experiment results (2026-09-26)

This branch contains the B2 Frozen Reference + Adaptive Learner implementation and its completed VOC person experiment. Noise optimization completed in 3.43 hours (24 epochs, 9,144 steps), within the requested six-hour ceiling. All 16,551 training images were materialized and passed the full-data preflight; validation used the 4,952-image clean VOC split. The victim was randomly initialized YOLOv8n (`pretrained=false`), trained for 200 epochs with seed 0, batch 36 and image size 640; training exited successfully.

## Aggregate metrics

All AP and recall values below are percentages.

| Metric | B0 | B2 | B2 − B0 (percentage points) |
|---|---:|---:|---:|
| mAP50_target | 3.4140 | 2.2018 | -1.2122 |
| mAP50_non_target | 68.9737 | 69.4336 | +0.4599 |
| mAP50_all | 65.6957 | 66.0720 | +0.3763 |
| Recall_target | 0.0000 | 0.0000 | 0.0000 |

B2 person AP50 is 2.2018%, non-target mAP50 is 69.4336%, and all-class mAP50 is 66.0720%. B0 comparison: person 3.4140%, non-target 68.9737%, all 65.6957%.

## Per-class AP50 and recall

Values are percentages. The classes follow VOC20 class-index order.

| ID | Class | B0 AP50 | B2 AP50 | B2 Recall |
|---:|---|---:|---:|---:|
| 0 | aeroplane | 82.3183 | 83.9969 | 71.7839 |
| 1 | bicycle | 64.4438 | 64.6043 | 54.0059 |
| 2 | bird | 65.1847 | 65.1033 | 58.8235 |
| 3 | boat | 66.8142 | 68.6499 | 58.1749 |
| 4 | bottle | 46.0204 | 47.5374 | 38.3795 |
| 5 | bus | 83.3190 | 83.2685 | 74.1784 |
| 6 | car | 87.8662 | 88.6588 | 78.4346 |
| 7 | cat | 80.5109 | 80.9259 | 75.2053 |
| 8 | chair | 51.6457 | 53.2546 | 46.3017 |
| 9 | cow | 68.2928 | 68.7577 | 65.5738 |
| 10 | diningtable | 67.2768 | 68.0675 | 56.7961 |
| 11 | dog | 61.0809 | 58.7493 | 68.7504 |
| 12 | horse | 71.9249 | 72.3732 | 68.9655 |
| 13 | motorbike | 70.4431 | 72.0110 | 68.0373 |
| 14 | person | 3.4140 | 2.2018 | 0.0000 |
| 15 | pottedplant | 43.6969 | 44.1729 | 35.2837 |
| 16 | sheep | 70.5495 | 71.9600 | 67.7686 |
| 17 | sofa | 70.8438 | 71.8003 | 71.1297 |
| 18 | train | 83.6762 | 83.4711 | 79.4326 |
| 19 | tvmonitor | 74.5925 | 71.8757 | 59.5271 |

## Reproducibility and limits

B2 used `η_inner=1e-4`, `λ_adapt=1`, `λ_nt=0.25`; frozen B0 batch size 16; one deterministic adaptive image per step. Perturbation covered every target-box pixel, with half strength in target/non-target overlap; global budget 16/255 and overlap budget 8/255. Noise parameter SHA-256: `1cb2a681748a3993b7bc8b188f77c646b0eabe3cfcf2da0f26b7adaaecdd8679`. B2 config SHA-256: `f08e4abae2f8d749c160c24f4285ef38c53f1d20f578ec3c275f1d5ff9d1feff`.

The B0 and B2 noise runs used 40 and 24 epochs respectively, and each result is from a single seed. The comparison therefore does not isolate the adaptive branch. There is no matched clean baseline in this run, so these numbers do not establish clean-relative degradation or causal background absorption. Target recall is zero at the evaluator operating point, while target AP50 remains nonzero.

The full verification JSON is `metrics.json`. Model weights and datasets are intentionally excluded.
