# COCO NT-risk independent scientific implementation review

- Review mode: `scientific_review`.
- Reviewer: `/root/coco_nt_risk_boundary_scientific_review`, independent subagent inspecting the lean packet, approved sources, committed code and referenced raw evidence. No other reviewer, model forward, remote/GPU run or source edit was performed. Exact model deployment identifier is unavailable; no model override was used.
- Result: **scientifically_incorrect**.
- Decision: **do_not_run** until F01 has reproducible closure.
- Snapshot: `codex/s4-coco-risk-fallback-20261027` / `352425896f31c3e67df0902ec562e67bf88b0a0b`; diff base `49fa2c58aaa1d99e3e3804ec0995a42e3d788141`.
- Packet raw-file SHA256: `18498ca8440fa42e5ad18306e378a609bb5894e55759fc3b7e01b82a7229c3bb`.
- Packet canonical-JSON SHA256: `ed1451e4e5007c34654176143ab148cae473250758d08f8cbe9b0b89a378047e`. Both independently recomputed; their difference is serialization, not packet drift.
- Approved basis: `USER_RISK_FALLBACK_AUTHORIZATION_20261027.md`, `USER_AUTONOMOUS_BUGFIX_AUTHORIZATION_20261027.md` and `COCO_TFSS_RISK_FALLBACK_SPEC.md` under `docs/S4_COCO_MINITRAIN_20261023/`.

All paths below are relative to the reviewed repository root recorded in the companion JSON. This is a **Level C methodological boundary extension**. It is not globally equivalent to strict Original S4, whose complete risk is undefined on affected inputs.

## One scientific blocker

**F01 — Compute the nearest-candidate center and size in float64 before ranking.**

Source: `coco_s4/risk_fallback.py:147-155`. The approved spec's “Candidate detection locations” section requires float64 ranking of the actual GT-size-normalized center distances, with native anchor index breaking a true tie. The code first computes `size` and `center` on the incoming float32 box and only then casts those already-rounded results to double. This can create a false tie and select the wrong native location, changing the fallback detection loss, risk gradient and conditional W.

A read-only IEEE754 arithmetic counterexample, without any detector execution, establishes the discrepancy. Set the geometric mask empty, as explicitly supported by this branch. Use float32 box constructor values `[7.99, 3.99, 8.010001, 4.01]` and native anchors `[[4,4],[12,4]]`:

| Quantity | Current calculation | Specified float64 calculation |
|---|---:|---:|
| Actual x1/x2 endpoints | 7.989999771118164 / 8.010001182556152 | same endpoints |
| Derived center x | 8.0 | 8.000000476837158 |
| Left-anchor x distance squared | 39994.354845637805 | 39994.36438103562 |
| Right-anchor x distance squared | 39994.354845637805 | 39994.34531024112 |
| Selected native index | 0, false tie | 1, strictly nearer |

Both anchors share y, so the y contribution is equal and cannot repair this ordering. The existing symmetric nearest-candidate fixture in `coco_s4/validate_risk_mixed.py` does not expose this case.

Fix the empty-mask branch by promoting the GT **endpoints** first:

```python
box64 = box[0, 0].double()
size64 = box64[2:] - box64[:2]
center64 = (box64[:2] + box64[2:]) / 2
distance = ((anchor_pixels.double() - center64) / size64).square().sum(-1)
```

Keep the existing validity checks, no epsilon, unchanged GT and native-index `argmin` tie policy. Probe the actual helper with the stated box and an explicit empty-mask assigner; it must return `[False, True]`. Retain an exact-center tie fixture, invalid-box tests, and the real five-candidate failed GT. Commit the repair and rerun the relevant bounded current-commit CUDA production smoke because the scientific selection sink changed. The main agent should record blocker → fix → source-to-selection evidence → smoke closure in the original implementation work. **No second or closing scientific reviewer is required.**

This finding does not assert that the eight-image probe or the known backpack took the nearest branch: the actual backpack used five native geometric candidates. The population incidence and effect of the empty-mask branch remain unmeasured.

## Other evaluable dimensions

| Dimension | Verdict | Evidence and scope |
|---|---|---|
| Intent alignment | incorrect | F01 violates the declared float64 candidate-ranking contract |
| Canonical location / instantiation | correct | explicit risk estimator and separate cache selected in canonical training and export |
| Critical data flow | correct | true GT/class, TAL trigger, candidate weights, native losses and original reductions reach their sinks |
| Computation sink effect | incorrect | all observed sinks are active, but F01 can select a different fallback location |
| Runtime scientific state | correct | actual frozen learned Teacher, train-only inputs, finite CPU/CUDA probes, recorded devices |
| Covered / disabled equivalence | correct in measured domain | tolerance-zero numerical probes; no equivalence claim on undefined inputs |
| Experiment identity | correct for validation candidate | policy, source, Teacher, data and cache hashes; formal launch disabled |

The official annotation1827724/image247692 is a legitimate backpack (COCO category27 → YOLO24), not a label or LetterBox bug. All23 converted image GT rows were checked. Actual native and independent same-TAL matching both lack effective supervision; negative unclamped CIoU gives zero native alignment. Evidence: `execution/COCO_RISK_FALLBACK_OFFICIAL_COORDINATE_AUDIT_20261027.json`, `docs/S4_COCO_MINITRAIN_20261023/COCO_S4_NATIVE_TAL_ZERO_EVIDENCE_20261027.json` and `execution/p/coco_dim_20261027_risk_gpu02/result/risk_validation_summary.json`.

Apart from F01, the fallback has ordinary localized detection-supervision meaning: hard true NT class, BCE over the same79 NT logits, original unclamped CIoU/DFL and gains, uniform candidate weights. Candidates are explicitly not native positives. Each NT GT enters once; the original per-present-class GT mean, input-gradient norm, class-map normalization and equal class mean are retained. On the mixed failed image, CPU evidence establishes exact original loss and full input-gradient equality for all13 originally defined NT GT terms; the14th original term stays null. Class-normalized risk exactly matches the native cache. Actual no-Person and Person-only cases are also recorded. Evidence: `execution/p/coco_dim_20261027_risk_mixed_cpu02/diagnostics/a8fb18c6d2be44a8833bc0dd6cc058d0/output/result/mixed_pre_regression.json`.

Covered COCO CPU/CUDA outputs, records and full input gradients match the original with tolerance0. CUDA comparisons use a common deterministic domain and original-repeat controls. VOC20 bounded equivalence passes; the independent regression summary records55 tests,0 failures,0 errors,0 skips. The default-backend VOC comparison and both mixed-regression harness failures remain failures. Their partial evidence is used only for the stages actually completed, never as an overall pass. Sources: component summaries under `execution/p/coco_dim_20261027_risk_cpu01/` and `risk_gpu02/`, the `voceq_diag01` diagnostic and `risk_reg_cpu01/result/regression_summary.json`.

The separate cache reuses original code objects with instance-local estimator bindings and records policy/code/spec/adapter identity. It does not patch the original modules. Real learned-Teacher CPU and CUDA full chains each contain15 native forwards, EOT2 and one isolated Adam update on image247692. TFSS weights enter Frozen/EOT/letterbox and Adaptive paths, and direct BG/rel Fourier-coefficient gradients are nonzero. CPU mean-EOT norms are0.1142069623/0.1140771732; CUDA norms are0.1270332336/0.1258603483. The saved native parameters reproduce the exact final image/export construction; decoded PNG max2 bytes, overlap1 byte and outside0 remain within8/4/0. These are bounded validation facts, not formal poison export or cross-backend equality. Raw sources: `execution/p/coco_dim_20261027_risk_chain_{cpu,gpu}01/result/` summaries, `per_loss_fourier_gradients.json` and `validation_export/export_link_proof.json`.

All154 files in the current statistics release match their local hashes. All152 computation files from validated `a0c32b9` match exactly; the only added files are `risk_population.py` and `validate_risk_mixed.py`. Current-commit statistics CPU/CUDA probes each record11 native forwards and0 updates over eight training images:69 total GT,35 NT GT, one fallback. Teacher state is unchanged. The reviewer independently matched raw summaries against audit copies, recomputed coefficient-proof hashes/norms and checked histogram/per-class totals using standard-library read-only calculations.

`risk_population.py` checks accepted train bytes, every NT GT exactly once, all79 class bins and separate Person/eligible-Person counts. It constructs actual Q/R/W only for affected eligible Person images and reports a **pixel-weighted conditional W distribution on those images**, while leaving the full-training W distribution null. Its constructor chain instantiates no optimizer and the entrypoint calls no training/optimization method. Coverage-only candidates on other images are not represented as validated individual gradients. Current smoke: `execution/p/coco_dim_20261027_risk_stat_gpu01/result/risk_population_summary.json`, source `3524258`, actual cuda:0/index0 outputs,11 forwards, finite outputs,0 updates; CPU counterpart also passes.

## Gated action and limits

The currently blocked command is:

```text
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B -m coco_s4.risk_population --release /project/guoqinglong/projects/s4_coco_experiment_a_20261025/releases/_anchors/coco_dim_20261027_risk_population01/source_release.json --output /project/guoqinglong/projects/s4_coco_experiment_a_20261025/validation/coco_dim_20261027_risk_population01/result --device cuda:0 --limit 0
```

After F01 has reproducible main-agent closure, the permitted scope is **only the complete fixed25K TRAIN incidence / conditional-W validation audit, with no optimizer updates**. Full Fourier24, poison25K, victims and evaluation remain unauthorized. Do not resume the old12-update Fourier checkpoint under a changed risk definition.

There are no full25K population results yet. The eight-image probe deliberately includes the known failed image and cannot estimate population prevalence. Complete original Rnt/W on affected images are undefined/null, so no delta against an imputed zero is valid. The `gradient_validated_fallback_GT_in_eligible_images` count identifies GT included in a validated class-reduced path; it is not a separately measured nonzero derivative for every GT. Nothing here establishes final AP, efficacy, causality or background absorption. rrctl/process/monitor/transport scaffolding was excluded from scientific findings.
