# S4 robustness data catalog

Snapshot: 2026-10-21T14:19:19.454649+00:00. This is a dated local summary derived from accepted result artifacts and the latest read-only RRCTL snapshot, not a live dashboard.

## Locked source and protocol

- Scientific source: `e17c24c54aae4cdae4bcc4703facf2a57052d2c8`.
- Config lock SHA-256: `00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91`.
- Parent S4 manifest SHA-256: `194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36`.
- Dataset: VOC20, Person target, 16,551 training images and the unchanged 4,952-image clean validation set.
- Victim: random-initialized YOLOv8n, seed 0, 200 epochs, batch 36, workers 16, AMP true; last.pt@200 primary, best.pt secondary.
- Only the original S4 protected training data is used for Poison arms. No S0/S3/S5 poisoned data, new noise, extra seed, S4 method modification, or AT run is included.

## Current server execution

- Pipeline: `s4_robustness_server_pipeline_seed0_20261020_recovery01`; state `RUNNING`; stage `meanf`; active job `meanf:S4:attempt01`.
- Snapshot time: 2026-10-21T14:19:19.454649+00:00; one bound training root; resource blockers: `[]`.
- MeanF checkpoint epoch 193; AMP attempt epoch 194, batch 148/88929; actual updates 44886, skips 22, all finite True/True/True.

## Accepted pair inventory

| Defense | State | RunID | Clean Person AP50 | S4 Person AP50 | Clean/S4 NT mAP50 | P | U | S |
|---|---|---|---:|---:|---:|---:|---:|---:|
| jpeg50 | COMPLETE_ACCEPTED | `s4_robustness_jpeg50_paired_seed0_20261019_v1` | 0.865868647 | 0.072199285 | 0.763383336 / 0.679123317 | 0.916616354 | 0.889622925 | 0.902917937 |
| jpeg30 | COMPLETE_ACCEPTED | `s4_robustness_jpeg30_paired_seed0_20261019_v1` | 0.855396434 | 0.216120307 | 0.757089800 / 0.684108512 | 0.747344858 | 0.903602864 | 0.818079150 |
| gray | COMPLETE_ACCEPTED | `s4_robustness_gray_paired_seed0_20261019_recovery01` | 0.850729237 | 0.015809466 | 0.759428729 / 0.678860859 | 0.981416571 | 0.893909900 | 0.935621613 |
| bdr | COMPLETE_ACCEPTED | `s4_robustness_bdr_paired_seed0_20261019_v1` | 0.868471160 | 0.028381203 | 0.766823981 / 0.695745784 | 0.967320500 | 0.907308328 | 0.936353834 |
| gauss | COMPLETE_ACCEPTED | `s4_robustness_gauss_paired_seed0_20261019_v1` | 0.851175347 | 0.095012809 | 0.747003404 / 0.678606919 | 0.888374576 | 0.908438857 | 0.898294692 |
| meanf | IN_PROGRESS | `s4_robustness_meanf_paired_seed0_20261019_v1` | — | — | — | — | — | — |
| medianf | NOT_RUN | `s4_robustness_medianf_paired_seed0_20261019_v1` | — | — | — | — | — | — |
| data_aug | NOT_RUN | `s4_robustness_data_aug_paired_seed0_20261019_v1` | — | — | — | — | — | — |

Gray, BDR, and Gauss full per-class metrics were retrieved from accepted server pair JSON and independently checked against each arm's 200-row CSV count, checkpoint SHA, parent S4 manifest, clean validation identity, and paired P/U/S calculations. The BDR local pull manifest hash was checked before normalization.

## Data manifests and reports

`S4_DEFENSE_DATA_MANIFEST.csv` lists generated transformed-data manifest hashes and model checkpoint hashes. Gauss Clean is a provenance-checked historical 200-epoch checkpoint reused and reevaluated; no new Clean dataset manifest was produced for that reused arm. Full metrics and pair-artifact SHA-256 values are in `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`.

`PHASE_A_REPORT.md` records accepted Phase A results. `PHASE_B_REPORT.md` records accepted Gauss results and the active MeanF snapshot. `S4_ROBUSTNESS_FINAL_REPORT.md` remains explicitly interim until Phase B acceptance is complete.

Raw server artifacts, absolute server paths, weights, datasets, and credentials are not part of the sanitized publication set.
