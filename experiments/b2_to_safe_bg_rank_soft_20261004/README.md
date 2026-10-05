# Safe-BG rank and soft-removal ablations (2026-10-04)

Two independent B2-TO-Safe-LR ablations were run under the same VOC20/person protocol. Safe-R4 changes only the Safe-BG projection rank from 8 to 4 (hard removal). Safe-Soft05 retains rank 8 and changes only the removal factor from 1 to 0.5. Both use the same selected P5 channels, seed 0, perturbation budget 8/255 within person boxes and 4/255 in overlapping boxes, and clean 4,952-image validation. The primary evaluation is from **last.pt**; best.pt is secondary.

| Model (last.pt) | Person AP50 | Person recall | Non-target mean AP50 | All mean AP50 | P | U | S |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.867943 | 0.758171 | 0.766462 | 0.771536 | — | — | — |
| B2-TO | 0.036719 | 0.000000 | 0.689359 | 0.656727 | 0.957694 | 0.899405 | 0.927634 |
| Safe-LR-R8 | 0.168829 | 0.019044 | 0.696026 | 0.669666 | 0.805483 | 0.908102 | 0.853720 |
| Safe-R4 | 0.066554 | 0.002574 | 0.690443 | 0.659248 | 0.923320 | 0.900818 | 0.911930 |
| Safe-Soft05 | 0.028993 | 0.000663 | 0.691698 | 0.658563 | 0.966595 | 0.902456 | 0.933425 |

P is normalized target AP50 reduction relative to clean, U is non-target AP50 retention relative to clean, and S is their geometric mean. Safe-Soft05 has the lowest target AP50 and highest S among these four poisoned models; Safe-R4 improves target suppression versus Safe-LR-R8 with a smaller non-target retention. These are descriptive single-seed results, not proof of a background-absorption mechanism.

Both noises completed 24 epochs / 9,144 steps: Safe-R4 in 5.691 hours, Safe-Soft05 in 5.716 hours, each below the 8-hour cap. Both full datasets passed the 16,551 train / 4,952 clean-val label, source fingerprint, person-box change, and pixel-budget audits. Each independent random-init YOLOv8n victim completed 200 epochs (seed 0, batch 36, image size 640, pretrained false). All pipeline stages exited 0. Local verification is in [verification.json](verification.json); source and config hashes, gate evidence, stage statuses, and checkpoint SHA256 values are in the adjacent JSON files. The source fingerprint matched before and after victim training.

Secondary best.pt evaluation: Safe-R4 person AP50 0.081351, recall 0.002650, non-target 0.697181, all 0.666389; Safe-Soft05 person AP50 0.040413, recall 0.000546, non-target 0.694585, all 0.661876. The automatic victim training metrics files are retained for provenance; the explicit last.pt checkpoint evaluations are the primary results.

See [ablation_metrics.json](formal/comparison/ablation_metrics.json) and [VOC20 per-class AP50/Recall](formal/comparison/voc20_per_class.csv) for the complete numerical results; [four_method_comparison.csv](formal/comparison/four_method_comparison.csv) is a compact table. The clean/B2-TO/Safe-LR reference checkpoints were re-evaluated with the same clean validation and last.pt rule. This branch contains code, configs, and small text evidence only; image data and model weights remain on the authorized server.
