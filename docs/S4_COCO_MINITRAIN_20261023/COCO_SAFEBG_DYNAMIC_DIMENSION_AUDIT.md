# Original S4 — COCO80 Dimension-Compatible Implementation

Protocol: `coco80-strict-tsfl-selected-dimension-v1`. Direct authorization is recorded in USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md. This report describes a CPU-validated implementation component; formal recovery is not yet ready or running.

## Necessary protocol difference

The former COCO compatibility guard required D=24 regardless of the StrictTSFL result. The new explicitly authorized COCO80 protocol takes D from the actual ordered selected indices. VOC20 retains its default24D consumer contract. Both datasets retain the same StrictTSFL rule `min(ceil(.15*C), N_eligible)` and .4 target gate/.6 NT safety quantile/Top3 sensitivity/min4. Current COCO P5 has C=256,D=39; original VOC uses24. There is no truncation, threshold change, layer search or AP tuning.

The actual39 indices, target/NT scores, eligible/global rankings and calibration identities are retained in COCO_SAFEBG_RETAINED_SELECTION_20261025.json. Existing failure/commits and the immutable v3 workflow are preserved.

## Mathematical scope

Only dimension admission/coordinate identity and bounded-cache validation change in the Safe-BG scientific consumers. Ordered SVD, rank8, removal_alpha=.5, active_k3 and norm restoration are unchanged. Candidate SafeBG and frozen-reference VOC24 outputs/diagnostics are bit-identical in measured float32/float64 probes. The eager VOC cache projector hash is identical. COCO selected bases/target/background use the same selected coordinates; a separately versioned identity prevents accidental cross-space reuse. P3 Orth remains independent model.15/64D, with48 selected coordinates under its original .75 rule. P5 SafeBG remains model.21.

The on-demand cache infers D and validates every class basis's dimension, finiteness, full column rank and orthonormality. Each active ordered triplet must support rank8; no omission or rank fallback is permitted. Cached q/spectrum tensors use compact storage and bounded LRU entries. Measured small-cache fixture retains3984 bytes for3 entries; production max8192 is unchanged and scales linearly with D*rank rather than the number of all79-choose3 combinations.

L_BG/L_rel/L_ret/L_orth, Fourier/EOT/Adam/Frozen/Adaptive and all spatial/control sources remain unchanged. Original support8/4/0, STE2.01/255, TFSS eta.85/Q_t/R_nt, official25K/COCO80/cleanval5K are frozen. This is not Full/S5/CUP.

## Retained calibration reuse

A fresh RRCTL CPU-only audit rehashed the full retained raw statistics, support manifest, K8 bank and accepted Teacher. It checked25K images/55096 Person GT, all79 NT classes' finite raw arrays/pools,13630 support images, and exact selection agreement with the original block. No model forward, GPU initialization or data re-scan occurred. P3 has124232 pooled NT features; P5 has89327. The8x256 prototype bank is complete. Hashes and39 score/ranking rows are in the retained selection JSON.

These inputs suffice to reconstruct affected P5 coordinates and subspaces without full calibration or Teacher training. The new builder supports byte-identical import of the retained P3 reference. **The actual retained-data reconstruction/consumer/full-loss tests remain OPEN.** No old failed receipt has been changed to ACCEPTED.

## Regression provenance

The original8-source lock and historical B2 manifest remain preserved. The explicitly authorized source delta records old/new SHA for safebg.py and b2_semantic.py. The historical B2 fixture predated actual Original S4; its exact file set is now checked against the frozen pre-adaptation3a8db4c snapshot, with the explicit dimension delta. One old __new__ support fixture omitted support_mode; the test fixture now sets its intended prebaked mode. Neither repair changes production computation. Related55 regressions pass with0fail/0error/0skip. The two formerly optional native integration tests ran using a local random-YAML VOC20 checkpoint and synthetic inputs; this is regression evidence, not learned VOC efficacy.

## Remaining gates

Production phase/dependency contracts still contain the old24D guards and v3 route. Complete the versioned recovery binding and all downstream dimension checks; reconstruct retained coordinates; verify actual consumers/full-loss CPU and L_BG/L_rel-to-Fourier gradients; publish the final current candidate; execute bounded isolated CUDA production smoke; obtain exactly one new independent scientific review and main closure; verify no conflicting jobs/capacity; then create a unique recovery using the existing serial control mechanism. Do not restart v3 or register OPEN phases.


## Measured reconstruction and first native review: F01 erratum
The earlier OPEN reconstruction/production-contract notes above are superseded by actual retained-data reconstruction and native CPU/CUDA execution. All39 original channels, all79 bases, K8 and byte-identical P348 reference were independently verified. The first current-source learned-Teacher CPU/CUDA probes each executed12 native forwards, EOT2 and one Fourier update. Their per-loss Fourier gradient labels were incorrect: original grad_BG_safe_norm/grad_rel_safe_norm measure poison-image derivatives. The sole independent reviewer rejected that attribution (F01); original reports and negative verdict are retained without alteration. Formal recovery remains CLOSED pending direct coefficient evidence.

The corrective wrapper passively measures autograd.grad of actual weighted BG_sel and rel_sel with respect to trainer.fourier_coeff for each EOT draw, preserves optimizer gradient accumulation exactly, and records per-draw plus mean-EOT derivatives and parameter shape/device. Image derivatives retain their accurate name. Admission rejects old mislabeled evidence. No core loss, parameterization, weights, model, method or hyperparameter change is made. Inherited fractional categorical CSV averages (F02) are preserved; an additive per-EOT JSONL sidecar records actual integer selected IDs, with no ranking/loss change. New current-source CPU/CUDA validation and main-agent closure of the one review are required; no second review.


## First formal batch numerical block (latest actual state)
See COCO_SAFEBG_DIMENSION_NUMERICAL_CONSISTENCY_REPORT_20261025.md and DIMENSION_RECOVERY01_20261025/. Current-source single-image direct coefficient gradients pass, but first formal batch16 fails actual rank8 projector orthogonality: 1.03712e-5 > unchanged1e-5 for ordered classes38,24,26. Both failed controller/child are absent. Bounded diagnosis confirms true Gram violation even evaluating sameq in double precision. No solver/threshold change, no blind restart, no downstream results. Formal readiness remains BLOCKED pending a validated numerical implementation correction.
