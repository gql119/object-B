# Phase A report — JPEG50, JPEG30, Gray, and BDR

Status: `COMPLETE_ACCEPTED` for all four paired defenses. Each Clean/S4 condition has a full 200-epoch evaluation on the unchanged 4,952-image clean VOC validation set; `last.pt@200` is primary and `best.pt` secondary. All values below are fractions and descriptive for seed 0.

The BDR protocol is the sourced 5-bit, 31-level transform in the frozen protocol lock. Gray uses the locked RGB→gray→three-channel RGB conversion. Transforms affect training data only.

| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT mAP50 | S4 NT mAP50 | Person Recall Clean/S4 | P | U | S | WCD | TCD3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jpeg50 | 0.865868647 | 0.072199285 | 0.763383336 | 0.679123317 | 0.772084806 / 0.005962898 | 0.916616354 | 0.889622925 | 0.902917937 | 0.290213226 | 0.241088407 |
| jpeg30 | 0.855396434 | 0.216120307 | 0.757089800 | 0.684108512 | 0.744132005 / 0.039851224 | 0.747344858 | 0.903602864 | 0.818079150 | 0.287642902 | 0.226041942 |
| gray | 0.850729237 | 0.015809466 | 0.759428729 | 0.678860859 | 0.738806758 / 0.001325088 | 0.981416571 | 0.893909900 | 0.935621613 | 0.320781220 | 0.257043844 |
| bdr | 0.868471160 | 0.028381203 | 0.766823981 | 0.695745784 | 0.771234454 / 0.000441696 | 0.967320500 | 0.907308328 | 0.936353834 | 0.263181833 | 0.236967470 |

JPEG30 is the weakest target-protection result among these four measured defenses: its Person AP50 rises to 0.216120307 and paired P is 0.747344858, versus 0.072199285 / 0.916616354 for JPEG50. Gray and BDR retain lower Person AP50 (0.015809466 and 0.028381203) with paired P of 0.981416571 and 0.967320500. These are single-seed observations; they do not establish significance or mechanism.

Non-target utility remains measurable for all four pairs (U ranges 0.889623–0.907308). Absolute Clean and S4 metrics, all five per-class measures for all 20 VOC classes, artifact hashes, and paired scores are in `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`, `PHASE_A_ROBUSTNESS_RESULTS.csv`, and `PHASE_A_PER_CLASS_RESULTS.csv`.

The untransformed S4 reference is Person AP50 0.028158895 and non-target mAP50 0.693787044. Per-defense comparisons must still use the matching Clean condition; these baseline values are not substituted as the P/U denominators.

The server-owned Phase B queue was `meanf` at the latest snapshot (2026-10-21T14:19:19.454649+00:00); this does not change the accepted Phase A results.

No formal AT run is included or authorized.
