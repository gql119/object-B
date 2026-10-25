# COCO dimension-compatible Fourier scientific review

- Review mode: `scientific_review`.
- Reviewer: Codex independent subagent `/root/coco_dynamic_dimension_fourier_scientific_review`; independent read-only review of the supplied packet, committed code, authorization and referenced evidence. The reviewer did not author this implementation. Exact backend model ID is not exposed in this reviewer runtime and is not asserted.
- Reviewed commit: `e89102474b064e1973e57964a19297f117a181a0`.
- Diff base: `3a8db4c4277d2ffbcadef7864240063d84fff6b9`.
- Packet: `execution/COCO_DIMENSION_FOURIER_SCIENTIFIC_REVIEW_PACKET_20261025.json`; canonical UTF-8 JSON SHA256 `2260841e24be9843956d4f06563a43f1ebcaf21afa3aaf9cacf3aa40585b87b4` independently verified. Raw file SHA256 is `c95a85d6f7c5398756b650f5488e422727634705ce89e0ff69b2e98e06c5d271`.
- Result: **`scientifically_incorrect`**.
- Decision: **`do_not_run`**.
- `pre_run_result`: `fail`.
- Review recorded: 2026-10-25, after inspecting current-source GPU evidence completed at 13:54:21 UTC.

The blocking error is in scientific validation and result attribution: the new phase validator reports and admits image-space loss gradients as if they were measured gradients to Fourier coefficients. This review does **not** claim that the real per-loss Fourier gradients are zero. Their actual finite, nonzero values on the learned-Teacher production path have not been measured by the supplied evidence.

## Exact gated run

```text
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B -m coco_s4.phase_fourier --release /project/guoqinglong/projects/s4_coco_experiment_a_20261025/releases/coco_dim_20261025_1350z_gpu/source_release.json --output /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_dimcompat_v1/fourier24 --registration /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_dimcompat_v1/receipts/fourier24_READY.json
```

Only this Fourier24 scientific computation is reviewed. Subsequent export, victim training and evaluation remain separately gated. RRCTL, controller/process/scheduler operations and predictions of scientific effect size are outside this verdict.

## Findings

### F01 — Blocking: image gradients are mislabeled and admitted as per-loss Fourier gradients

**Source:** `ue_framework/methods/b2_semantic.py:408`, `:416`, `:417`; `ue_framework/methods/b2_adaptive.py:38`; `coco_s4/phase_fourier.py:528` through `:546`.

`B2SemanticTrainer._batch_loss` explicitly computes `torch.autograd.grad(terms[key], self._last_poison, retain_graph=True)`. `_last_poison` is the composed poisoned **image**, assigned from composition result index 2. Thus `grad_BG_safe_norm` and `grad_rel_safe_norm` are norms of derivatives with respect to image pixels. The new `phase_fourier` code reads those two fields, raises an error saying that a Fourier gradient is absent when either fails, and publishes them under `SafeBG_loss_Fourier_gradient_norms`.

The actual GPU evidence reproduces that attribution: the values `0.249729186296463` and `0.2502170205116272` exactly equal the image-gradient CSV fields. The CPU summary uses the same source and the same mislabeled route. The independent GPU audit checks these existing fields; it does not take a derivative with respect to Fourier coefficients. `grad_norm_fourier=0.10577037930488586` is computed after backpropagating the **total** loss, including retention, orthogonality, adaptive and other connected terms, rather than separately for BG and rel.

The chain rule requires `dL/dphi = J_poison(phi)^T dL/dpoison`. A nonzero pixel gradient does not establish a finite, nonzero product with the actual Fourier/support/composition Jacobian; a nonzero total Fourier gradient does not identify which loss reached that parameter. The synthetic probe in `coco_s4/validate_dimension_cpu.py:87` genuinely differentiates its two synthetic losses with respect to synthetic Fourier amplitudes, but it uses random features and a synthetic linear feature map, not the learned Teacher and native composition. It cannot close the explicitly required production per-loss sink check.

**Required closure:** retain the existing image-space diagnostics under accurate names. On the actual learned-Teacher production graph, separately measure `autograd.grad` of the BG and rel terms with respect to the actual `trainer.fourier_coeff`, before the graph is freed; record the differentiated variable, coefficient shape, finite status and nonzero norm for each loss. Correct the new admission logic and summary attribution to consume those actual coefficient derivatives. Supply corresponding current-source bounded production evidence and main-agent blocker closure. No loss formula, weight, spatial budget or method tuning is requested. This is the sole blocking finding; a second formal scientific reviewer is not requested.

### F02 — Nonblocking, inherited: categorical active-class IDs are averaged across EOT

**Source:** `ue_framework/methods/b2_semantic.py:267` through `:271`; `ue_framework/methods/legacy_kproto_ret.py:519` through `:521`; actual GPU `diagnostics.csv`.

Per-draw `active_nt_class_*` fields contain class IDs. The parent trainer averages all numeric diagnostics across EOT, except the prototype-ID string. The actual GPU row consequently records `active_nt_class_3=70.5`, which cannot denote a COCO class. The B2 postprocessor repairs `worst_class_*` but does not repair active SafeBG class IDs. The projection computation itself still ranks all 79 classes correctly; this observation does not invalidate that mathematical loop or add a separate Fourier gate. It does prevent interpreting these serialized ID fields as actual selected category identities. Preserve per-draw IDs or aggregate categorical counts/modes explicitly instead of averaging IDs. This issue predates the dimension diff and is recorded without claiming that it was newly introduced.

## Evaluated scientific dimensions

| Dimension | Verdict | Evidence and limits |
| --- | --- | --- |
| Approved intent | correct | Read `USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md` and `USER_AUTOSTART_TASK_20261025.md`. Authorization changes selected dimensionality, preserves original selection/gates and demands actual BG/rel Fourier reachability. |
| Canonical implementation and instantiation | correct | Production entrypoint instantiates `EligibleCOCOB2Trainer(B2SemanticTrainer)`, which instantiates the bounded on-demand cache for COCO80. Independently hashed all 135 release files against both the current checkout and the reviewed Git commit; no mismatches. |
| StrictTSFL selection and coordinates | correct | C=256, target candidates=103, safe candidates=78, selected D=`min(ceil(.15*256),78)=39`. Frozen `.4/.6/top3/.15/min4` rules remain. The 39 ordered indices match the retained selection and reconstruction receipt; selection is not truncated. Target, poison, prototype and selected NT features consume the same `self.channels` indices. |
| SafeBG computation | correct | All class IDs 1..79 are required and ranked. Selected class bases have ranks 12..19, maximum reconstruction Gram error `2.384185791015625e-7`; current GPU projector rank=8 and Gram error `8.225440979003906e-6` is below the unchanged `1e-5` gate. Ordered top-3 concatenation/SVD, alpha=.5, effective-rank rejection and original norm restoration remain. GPU restored/raw direction norms both equal `1.4666229486465454`. No class skipping or silent rank fallback was found. |
| Bounded cache | correct | Exact ordered-triplet key; cloned compact retained tensors, LRU limit=8192. Current report: 2 entries/2656 bytes, upper bound 10,878,976 bytes plus 221,052 bytes of class bases. CPU direct/cache equivalence and eviction evidence are supplied. |
| Independent P3 Orth/Retention | correct | Original `model.15` references and 48 selected channels are byte-reused, distinct from P5 `model.21`/39 channels. Current GPU consumer reports P3=48; original P3 projection/orthogonality code is unchanged. |
| Original losses, Fourier, Frozen/Adaptive and 8/4/0 | correct for evaluated computation | Scientific YAML differs only by the explicit dimension protocol; original Fourier/inner learner/loss sources are retained. Native Adam updates only Fourier coefficients, with the original one-step adaptive virtual update. Support uses 1/.5/0, original STE floor and epsilon; current GPU outside-support maximum=0. No Full/S5 route is selected. TFSS source/config regression preservation is evidenced; later export application is outside this gate. |
| Critical per-loss Fourier sink evidence | **incorrect attribution / not measured** | F01. Actual pixel gradients and a total Fourier update are present, but the required two production coefficient gradients are missing. |
| Baseline/disabled-path equivalence | correct within the stated kernel claim | Independently executed frozen original SafeBG source SHA `9c6dca9eedfd3623f88a70cbfe38edcd32310ac20fe22240871f926d9aeaeaa5` versus candidate `e0bfe9a5c86591c91e2ac1b560923994f1a322e6d34ec567e44bbeb01b673043`. CPU float32/float64 outputs/diagnostics/cache hashes exact. Same-GPU weightless 24D fixture: both norms `9.589117050170898`, difference=0, tolerance=0, every output/diagnostic exact, RNG restored. This supports VOC24 SafeBG kernel equivalence, not full VOC detector/AP equality. |
| Runtime scientific state | correct apart from F01 | Actual current-source CUDA0, 12 native forwards, one support-eligible train image, EOT2, one native Adam update. `L_BG=.4655998945236206`, `L_rel=.6655962467193604`, `L_total=1.204453468322754`, total Fourier norm=.10577037930488586, coefficient change=.0009999996982514858. Frozen Teacher remains unchanged. |
| Experiment identity | correct for this phase | Exact accepted EMA last@200 SHA `9749a7b6ceeb7b322316bdd97dfdde68f11641eef59fe6ba620a5e76a9521e3f`; train-only full25K statistics SHA `8be0361d373f1278075c5e8dff1ae21ae7a792b0488ef7cfeb3b8fd930c1995d`; K8 bank SHA `9aa1a3d75f1b8d76737fb18ddea4af4849cfa7886a443750f9f58ea874560a06`; support manifest SHA `c7ef1f83682428a38bf1ab63bc3c4b96ece6a88d51a8080d91f79cf5979440bf`. 13,630 eligible images imply 20,448 formal updates. Clean val is not consumed by Fourier. Release SHA `b6e562e572502f172853c24437e054526ec76224a9e2c53be98e17dd349a87ca`; config SHA `73cdfb480b2c2d5dcb4c57e351665477b1a218bedc1ac3b7ae5518a6c0900b60`. |

## Evidence read and validation provenance

- `docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_RETAINED_SELECTION_20261025.json`, `COCO_SAFEBG_DIMENSION_CPU_NUMERICS_20261025.json`, `COCO_SAFEBG_DIMENSION_REGRESSION_20261025.json`, `COCO_SAFEBG_PRODUCTION_CONTRACTS_20261025.json`, source delta/protocol and baseline lock; referenced reports record 55 regressions with 0 failures/errors/skips, and 13 rejected invalid contracts. These are prior actual results, not reruns by this reviewer.
- `execution/COCO_DIMENSION_RECONSTRUCTION_INDEPENDENT_AUDIT_20261025.json` and `execution/p/coco_dim_20261025_1340z_schema/result/candidate_receipt.json`: original 39 identities, 79 selected-space bases, reused P3/bank/support and the two diagnostic-only raw-mean aliases repaired in a fresh component.
- `execution/p/coco_dim_20261025_1345z_cpu/result/cpu_probe_summary.json`: actual learned-Teacher CPU full-loss path; F01 affects its gradient attribution too.
- `execution/p/coco_dim_20261025_1350z_gpu/result/{smoke_summary.json,phase_spec.json,diagnostics.csv}`, `execution/coco_dim_20261025_1350z_gpu/source_release.json`, `execution/COCO_DIMENSION_CURRENT_GPU_INDEPENDENT_AUDIT_20261025.json`; read the audit's generating code to check what its assertions actually measured. It verifies integrity and existing fields rather than supplying missing per-loss coefficient derivatives.

No code edits, tests, reruns, remote commands or experiment launches were performed in this review. Only read-only source/evidence inspection and local hash checks were performed; these two verdict artifacts are the only reviewer writes. Passing smoke/readiness was not used as a substitute for source-to-sink scientific correctness. F01 remains open and the exact formal Fourier run remains blocked.
