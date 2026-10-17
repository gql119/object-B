# S2/S3 registered schedule and admission status

Change: Parent selected S0 by the preregistered matched last.pt rule. Both methods independently start from original seed0 Fourier initialization and the original Frozen/Adaptive reference, not either victim checkpoint. S3 never starts from S2 perturbations.

| Method | Full Person-image epochs | Outer Adam steps | EOT batches / CNLP paired evaluations | Measured five-step seconds | Linear noise estimate hours |
|---|---:|---:|---:|---:|---:|
| S2 | 24 | 9144 | 18288 | 32.3682 | 16.44 |
| S3 | 24 | 9144 | 18288 | 38.0516 | 19.33 |

Each EOT batch performs one differentiable poisoned virtual SGD update and one detached clean virtual SGD update (J=1), with a one-image support and a disjoint one-image clean TRAIN query; the parent sixteen-image/EOT2 path stays unchanged. Across the registered run that is 18,288 paired paths, or 36,576 virtual SGD updates, in addition to the unchanged Parent computation. These are virtual updates, not persistent surrogate optimizer steps. The training-query sampler draws from its fixed hashed train-only manifest; it does not claim one exhaustive pass of every query image.

Timing is a short-run projection, not a full-cycle measurement, runtime guarantee, convergence criterion, or stop time. The independent full noise schedule completes even if the estimate is exceeded. The five-step measurement includes short-run checkpoint/epoch overhead and unrepresentative image/class complexity. Victim training and audits are excluded. Resource snapshot before review: idle GPU, 1,444,964,995,072 free bytes, 123,465,032 kB available RAM; recheck immediately before launch.

Result: numerical Parent equivalence max difference0; five-step S2/S3 and three-to-five exact recovery checkpoint difference0/diagnostics byte-identical; 12 real matching-edge queries covered51 GT observations including9 recovered native-uncovered objects; decoded smoke PNG budgets and labels passed. Current-source matched proxy completed15 optimizer steps /30EOT; no AP or causal efficacy inference. Separate CUP sink probe changed actual Fourier gradient12/20EOT. All smoke checkpoints retained. The first strict matching failure and the artifact-only packaging failure are retained; the latter used a CPU-only packaging recovery with no GPU rerun.

Finding: admission evidence supports a reviewable implementation, not final scientific efficacy or perturbation convergence. No formal noise or victim has been launched at this report timestamp. One independent review is in progress; both LF config locks are fixed and must be bound to its actual verdict.

Next: after scientific allow_run, launch S2 exactly once; full16551train/4952cleanval/13256PersonGT and 8/4/0 budget/labels/domain audits, independent random-init YOLOv8n200, last.pt primary/best.pt secondary complete VOC20 AP50/AP75/AP50-95/R/P and P/U/S/WCD/TCD3, archive, then independently launch S3. No AP-based retuning, additional seed, data deletion, or early GitHub publication.
