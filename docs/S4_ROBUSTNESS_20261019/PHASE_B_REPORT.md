# Phase B report — Gaussian, Mean, Median, and Data Augmentation

Status: `IN_PROGRESS`. The Gauss Clean/S4 pair is complete and accepted. MeanF S4 is the active server job; its Clean historical-control evaluation is pending. MedianF and Data Augmentation have not started. The server pipeline remains the sole stage scheduler.

| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT mAP50 | S4 NT mAP50 | Person Recall Clean/S4 | P | U | S | WCD | TCD3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gauss | 0.851175347 | 0.095012809 | 0.747003404 | 0.678606919 | 0.736545882 / 0.009275618 | 0.888374576 | 0.908438857 | 0.898294692 | 0.323968407 | 0.240520415 |
| meanf | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS | IN_PROGRESS |
| medianf | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| data_aug | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Gauss uses the audited 3×3 OpenCV Gaussian filter (`sigma=0`, default REFLECT_101 border), not Gaussian noise. Its paired values are P=0.888374576, U=0.908438857, S=0.898294692; Person AP50 is 0.851175347 Clean versus 0.095012809 S4, and non-target mAP50 is 0.747003404 versus 0.678606919.

The Clean model is a provenance-checked historical 200-epoch `gaussian3_clean` checkpoint, reevaluated on the fixed clean validation set; it was not retrained for this pair. Its historical checkpoint SHA is retained in the local matching audit; the accepted paired-results file binds the checkpoint and evaluation hashes.

At the 2026-10-21T14:19:19.454649+00:00 RRCTL snapshot, MeanF was `RUNNING` at 193 completed checkpoint epoch; the live AMP attempt is epoch 194, batch 148/88929. It recorded 44886 actual updates and 22 legal AMP skips, with 0 consecutive skips and finite model/EMA/optimizer state. These are execution diagnostics, not evaluation results.

MeanF metrics remain `IN_PROGRESS` until the server reports both arms accepted and the full paired-result audit passes. MedianF and Data Augmentation remain `NOT_RUN`; AT remains excluded.
