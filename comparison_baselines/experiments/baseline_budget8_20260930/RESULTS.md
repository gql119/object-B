# Four detection baselines: eight-hour noise-budget variant

The four runs completed in sequence. Each generation used complete passes over the 6,095 person-containing training images and stopped before starting a pass projected to exceed its eight-hour limit. The full 16,551-image training set was materialized and passed the pixel, label, and clean-validation audit. Each independent, randomly initialized YOLOv8n victim completed 200 epochs and was evaluated on the locked 4,952-image clean VOC20 validation set.

| Method | Full cycles | Noise time (h) | Person AP50 | Person Recall | Non-target mean AP50 | All-class mean AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean baseline | — | — | 0.865875 | 0.749406 | 0.771233 | 0.775966 |
| EM-Det | 46 | 7.729 | 0.639893 | 0.424886 | 0.742773 | 0.737629 |
| REM-Det | 2 | 6.267 | 0.672864 | 0.450924 | 0.740400 | 0.737023 |
| EntF-Push-Det | 2 | 6.524 | 0.411415 | 0.178004 | 0.719303 | 0.703908 |
| SEM-Det | 19 | 7.429 | 0.664251 | 0.458260 | 0.739022 | 0.735284 |

The exact VOC20 per-class AP50 and Recall arrays, generation counters, code and data hashes, stage statuses, audit records, and victim specifications are in [formal_results_bundle.json](formal_results_bundle.json). All four pipeline states are `complete`; the sequence and stage exit codes are zero. The generation `stop_reason` is `projected_full_cycle_budget_guard` for each method. EM's training-error threshold was recorded without ending this explicitly designated budget variant. The implementation and result should be described as detection adaptations, not exact official reproductions.

Source provenance: candidate `541e3389317f12e6f5f74d5c010341019c9ddb99`; previously sanitized source publication `0bce675`. No datasets, generated images, checkpoints, or model weights are included here.
