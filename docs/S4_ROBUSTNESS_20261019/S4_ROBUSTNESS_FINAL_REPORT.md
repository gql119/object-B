# S4 robustness results — interim, Phase B still running

Status: Phase A is complete and accepted; Gauss is accepted in Phase B; MeanF is in progress; MedianF and Data Augmentation have not started. Latest read-only RRCTL snapshot: 2026-10-21T14:19:19.454649+00:00, pipeline `s4_robustness_server_pipeline_seed0_20261020_recovery01`, stage `meanf`, state `RUNNING`. This is not final Phase A/B acceptance.

The scientific source, S4 data identity, and training configuration remain frozen. This report describes one seed and does not provide significance tests or causal evidence. It does not use S0/S3/S5 poisoned data and does not include AT training.

## Accepted paired results

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

## Research questions answered from currently accepted evidence

1. JPEG50: paired P/U/S=0.916616/0.889623/0.902918; S4 Person AP50 is 0.072199 versus Clean 0.865869.
2. JPEG30 is weaker than JPEG50 on target protection in this seed: P=0.747345 versus 0.916616, while S4 Person AP50 is 0.216120 versus 0.072199.
3. Gray retains low target AP in the paired comparison (Clean/S4 Person AP50 0.850729/0.015809; P=0.981417). This alone does not prove the background-misdirection mechanism.
4. BDR retains low target AP in this run (Clean/S4 0.868471/0.028381; P=0.967321).
5. Gauss is the only accepted Phase B filter so far (P=0.888375, U=0.908439, S=0.898295); it cannot yet be ranked against MeanF or MedianF.
6. Data Augmentation has not run; no result is available.
7. On absolute non-target mAP50, each accepted S4 arm is below its same-defense Clean arm. The paired U values are 0.889623–0.907308; whether that counts as a practically important loss needs the paper's predeclared interpretation, not a post-hoc threshold.
8. JPEG30 shows the clearest target recovery among accepted pairs (S4 Person AP50 0.216120) while retaining paired U=0.903603; this is a descriptive trade-off, not proof that one defense is preferable.
9. P/U/S are calculated from target AP50 and non-target mAP50 against the matching defense's Clean arm, using unrounded values; formula inputs and outputs are recorded in the accepted-pairs JSON.
10. Accepted pairs show low-to-moderate Person AP50 with non-target U near 0.89–0.91. A threshold for 'collapse' was not preregistered here, so this report does not label any result a collapse.
11. The measured P values range from about 0.747 to 0.981. The sensitivity to JPEG quality is visible in this seed; robustness across random seeds remains untested.
12. The next informative comparisons are the in-progress MeanF pair and queued MedianF/Data Augmentation pairs; final cross-defense conclusions wait until all Phase B audits pass.

## Files and provenance

Full-precision aggregate and 20-class results: `S4_ROBUSTNESS_ACCEPTED_PAIRS_20261021.json`, `PHASE_A_PER_CLASS_RESULTS.csv`, and `PHASE_B_PER_CLASS_RESULTS.csv`. Arm status and checkpoint/data-manifest hashes: `S4_ROBUSTNESS_EXECUTION_LEDGER_20261021.csv` and `S4_DEFENSE_DATA_MANIFEST.csv`. The original S4 baseline and config/data lock hashes are retained in the same files.

AT is protocol-only and remains unauthorized for formal training. Do not treat this interim report as final research acceptance.
