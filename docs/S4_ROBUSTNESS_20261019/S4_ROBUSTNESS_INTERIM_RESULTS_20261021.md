# S4 robustness interim results inventory

Observation: 2026-10-21T14:19:19.454649+00:00. The server pipeline remains active; this report is a timestamped evidence snapshot, not a live query or final Phase A/B acceptance.

Pipeline `s4_robustness_server_pipeline_seed0_20261020_recovery01` is `RUNNING` at stage `meanf` (`meanf:S4:attempt01`). The current MeanF checkpoint is epoch 193; AMP reports epoch 194 batch 148/88929, 44886 actual updates, 22 legal skips, and finite model/EMA/optimizer states. This is not a formal evaluation metric.

Scientific source `e17c24c54aae4cdae4bcc4703facf2a57052d2c8`, config `00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91`, parent S4 manifest `194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36`. Dataset and victim protocol are locked: VOC20 Person, 16,551 training / 4,952 clean validation, random-init YOLOv8n seed 0, 200 epochs, batch 36, 16 workers, AMP true; last@200 primary.

## Accepted Clean/S4 comparisons

| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT mAP50 | S4 NT mAP50 | Person Recall Clean/S4 | P | U | S | WCD | TCD3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jpeg50 | 0.865868647 | 0.072199285 | 0.763383336 | 0.679123317 | 0.772084806 / 0.005962898 | 0.916616354 | 0.889622925 | 0.902917937 | 0.290213226 | 0.241088407 |
| jpeg30 | 0.855396434 | 0.216120307 | 0.757089800 | 0.684108512 | 0.744132005 / 0.039851224 | 0.747344858 | 0.903602864 | 0.818079150 | 0.287642902 | 0.226041942 |
| gray | 0.850729237 | 0.015809466 | 0.759428729 | 0.678860859 | 0.738806758 / 0.001325088 | 0.981416571 | 0.893909900 | 0.935621613 | 0.320781220 | 0.257043844 |
| bdr | 0.868471160 | 0.028381203 | 0.766823981 | 0.695745784 | 0.771234454 / 0.000441696 | 0.967320500 | 0.907308328 | 0.936353834 | 0.263181833 | 0.236967470 |
| gauss | 0.851175347 | 0.095012809 | 0.747003404 | 0.678606919 | 0.736545882 / 0.009275618 | 0.888374576 | 0.908438857 | 0.898294692 | 0.323968407 | 0.240520415 |
| meanf | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS |
| medianf | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| data_aug | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

All values are fractions. Clean and S4 rows use the same defense's paired evaluation. P/U/S are derived from full precision Person AP50 and non-target mAP50; Clean rows have no P/U/S. WCD and TCD3 are included only for accepted S4 arms.

## Pair evidence

`S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json` contains sanitized per-class AP50/AP75/AP50-95/Recall/Precision, aggregate metrics, scores, data audit summaries, source/config/parent hashes, checkpoint SHAs, and paired artifact SHA-256 values. It omits remote absolute paths. `PHASE_A_PER_CLASS_RESULTS.csv` has 160 accepted rows; `PHASE_B_PER_CLASS_RESULTS.csv` currently has the accepted Gauss classes only.

The current MeanF progress checkpoint is retained as execution evidence only. It does not populate AP or P/U/S results. MedianF and Data Augmentation remain NOT_RUN. AT remains NOT_AUTHORIZED.
