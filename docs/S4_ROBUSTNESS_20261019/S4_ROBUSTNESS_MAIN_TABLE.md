# S4 robustness main table

Snapshot values are fractions from accepted paired full evaluations; `last.pt@200` is primary. Clean-arm S is not applicable. `IN_PROGRESS` and `NOT_RUN` are statuses, never numeric zeros. AT has no formal run and remains unauthorized.

| Arm / Metric | Clean | Gauss | JPEG50 | JPEG30 | MeanF | MedianF | Gray | BDR | AT | Data Augmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean Person AP50 | 0.867943219 | 0.851175347 | 0.865868647 | 0.855396434 | PENDING | NOT_RUN | 0.850729237 | 0.868471160 | NOT_AUTHORIZED | NOT_RUN |
| clean Non-target mAP50 | 0.766461905 | 0.747003404 | 0.763383336 | 0.757089800 | PENDING | NOT_RUN | 0.759428729 | 0.766823981 | NOT_AUTHORIZED | NOT_RUN |
| clean Selectivity S | — | — | — | — | PENDING | NOT_RUN | — | — | — | NOT_RUN |
| S4 Person AP50 | 0.028158895 | 0.095012809 | 0.072199285 | 0.216120307 | IN_PROGRESS | NOT_RUN | 0.015809466 | 0.028381203 | NOT_AUTHORIZED | NOT_RUN |
| S4 Non-target mAP50 | 0.693787044 | 0.678606919 | 0.679123317 | 0.684108512 | IN_PROGRESS | NOT_RUN | 0.678860859 | 0.695745784 | NOT_AUTHORIZED | NOT_RUN |
| S4 Selectivity S | — | 0.898294692 | 0.902917937 | 0.818079150 | IN_PROGRESS | NOT_RUN | 0.935621613 | 0.936353834 | NOT_AUTHORIZED | NOT_RUN |

Current server snapshot: `meanf` / `RUNNING` at 2026-10-21T14:19:19.454649+00:00; see `S4_ROBUSTNESS_EXECUTION_LEDGER_20261021.csv`.
