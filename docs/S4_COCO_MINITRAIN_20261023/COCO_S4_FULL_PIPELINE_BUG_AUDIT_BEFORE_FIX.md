# COCO S4 full pipeline audit — BEFORE FIX

Baseline: 433a9dd35198bc5c007b849a34a6f4b42fbb81fc; invalid runtime: dcd91b33d90dfd2472b8eacb843a994a4801d1cf.
No new scientific source edits since emergency task. Candidate433 predates task and is not full-method validated. Confirmed bugs:7.

## Complete coverage

- **Data — No Issue:** Official25K/5K metadata/manifests and current6image-label samples exact; retained full30K acceptance, no repeated fullscan; Person0/COCO80/alllabels/empty/noPerson preserved
- **Teacher — No Issue:** Fixed nativeEMA last200/fullstate/accepted/eval hashes reverified. No retraining. Prior actual CUDA evidence retained; never useTeacherAP as Clean200
- **TSFL/K8 — No Issue:** Full25K raw sensitivity/39indices-scores-ranks/all79pools/K8/eligible13630 rehashed. Later Fourier bug cannot mutate these frozen artifacts
- **Dimension — Intentional COCO Compatibility Change:** 39selectedD vsVOC24; same .4/.6/top3/.15/min4 strict rule; dynamic ordered rank8 SVD precision policy versioned, thresholds retained
- **P3 — No Issue:** Independent48 selected P3 reference files byte exact; not selected P5=39. Original Orth/Retention kernels retained
- **SafeBG — No Issue:** All79 bases rank12..19; top3→rank8/alpha.5/orderedSVD/normrestore/bounded8192cache. Actual safe target passed into kproto BG/rel; currentTFSS full model test pending
- **Loss/Frozen/Adaptive — Confirmed Bug:** S4-001: loss formulas/weights/signs unchanged, but poison source lacked TFSS; canonical candidate real CPU/GPU sink/gradient tests OPEN. inner1/lr1e-4, clean pairing, Teacher freeze retained
- **Support/Fourier — No Issue:** 1/.5/0→8/4/0, STE2.01/255,JND,64candidate/16saved,Adam/EOT2/b16/24epochs preserved; persistence bypass S4-002
- **TFSS maps — No Issue:** Original clean P5-energy Qt/nativeGT-aligned risk Rnt,perPerson max/equalNT mean,canonicalcanvas,detached W formula eta.85. Invalid runtime never used maps; repaired real path pending
- **Export — Confirmed Bug:** S4-003/004. Generator delta is preclamp; W then clamp/png_project. NonPerson bytecopy,alllabels,decoded8/4/0 required; real roundtrip pending
- **Victim — Confirmed Bug:** S4-005 metadata. Bothfresh randomYAML80/nativeSGD/seed0/200/640/b36/w16; stricttrainonly; Teacher not initialization
- **OfficialEval/report — No Issue:** Sparsecategory mapping,original5K,crowdignore,maxDets100,lastprimary/bestsecondary identity,AP50/75/5095/Recall/explicitPrecision50. signedPUS/noClip/zeroNull/WCD/TCD3/LPIPSnullreason; no formal result
- **Lifecycle — Confirmed Bug:** S4-007. Exact ownedSIGTERM once/native-15/controllerFAILEDnoRetry/dead/noGPU/downstream. No formal continuation authorized

## S4-001 / P0

- **id:** S4-001
- **classification:** Confirmed Bug
- **severity:** P0
- **file:** coco_s4/phase_fourier.py
- **function:** EligibleCOCOB2Trainer/run
- **VOC_expected:** Canonical S4Trainer+SoftSpatialProjection+CleanMapCache; TFSS weighted image consumed by Frozen/Adaptive
- **COCO_actual:** dcd runtime instantiated B2SemanticTrainer, no TFSS in optimization; 433 is a toy-tested pretask candidate
- **root_cause:** Wrong inherited B2 entry; finite gradient mistaken for complete S4 fidelity
- **affects_loss_or_gradient:** True
- **affects_poisoned_export:** True
- **AP_interpretation:** Old Fourier invalid for Original S4 AP
- **evidence:** COCO_TFSS_FOURIER_METHOD_FIDELITY_HOLD_20261026.json; COCO_TFSS_WIRING_CPU_COUNTEREXAMPLE_20261026.json
- **minimal_reproduction:** COCO_TFSS_FOURIER_METHOD_FIDELITY_HOLD_20261026.json; COCO_TFSS_WIRING_CPU_COUNTEREXAMPLE_20261026.json
- **repair_plan:** Audit/reuse433 only after real learnedTeacher CPU/CUDA source-to-sink evidence
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-002 / P2

- **id:** S4-002
- **classification:** Confirmed Bug
- **severity:** P2
- **file:** coco_s4/phase_fourier.py
- **function:** inherited train_universal/persistence
- **VOC_expected:** Canonical optimize_s4 saves completed-boundary coefficients/Adam/scheduler/cursor/full RNG
- **COCO_actual:** Stopped invalid run has CSV/sidecar but no global_params or continuation checkpoint; unsaved GPU memory cannot be recovered
- **root_cause:** Wrong optimizer entry bypasses original persistence; wrapper RNG saved only after successful entire loop
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** False
- **AP_interpretation:** No saved learned state or exact resume claim
- **evidence:** COCO_S4_REUSE_REMOTE_AUDIT_20261026.json:invalid_fourier_inventory
- **minimal_reproduction:** COCO_S4_REUSE_REMOTE_AUDIT_20261026.json:invalid_fourier_inventory
- **repair_plan:** Use original optimize_s4 and verify native saved state; future formal run starts fresh
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-003 / P1

- **id:** S4-003
- **classification:** Confirmed Bug
- **severity:** P1
- **file:** coco_s4/phase_export.py
- **function:** run params metadata gate
- **VOC_expected:** Consumer accepts original16x3/coords/eps/target/support/P5/identity pack; class80 bound by actual Teacher/config
- **COCO_actual:** Rejects original pack because num_classes field is absent from original saver
- **root_cause:** Producer/consumer schema mismatch
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** True
- **AP_interpretation:** Would block export; no current victim AP exists
- **evidence:** COCO_S4_BEFORE_FIX_MINIMAL_REPRODUCTIONS_20261026.json
- **minimal_reproduction:** COCO_S4_BEFORE_FIX_MINIMAL_REPRODUCTIONS_20261026.json
- **repair_plan:** Shared strict native-pack validator; class80 from actual bound model/config; optional metadata validated if present
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-004 / P2

- **id:** S4-004
- **classification:** Confirmed Bug
- **severity:** P2
- **file:** coco_s4/phase_export.py
- **function:** forward hook/smoke summary
- **VOC_expected:** Actual raw finite tensor devices/CUDA indices required
- **COCO_actual:** Counter-only hook and hardcoded cuda:0
- **root_cause:** Incomplete device evidence instrumentation
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** False
- **AP_interpretation:** Cannot declare exporter CUDA readiness
- **evidence:** phase_export.py:forward_calls.append(1)/actual_device
- **minimal_reproduction:** phase_export.py:forward_calls.append(1)/actual_device
- **repair_plan:** Capture nested actual forward tensors; fail closed on device/nonfinite mismatch
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-005 / P2

- **id:** S4-005
- **classification:** Confirmed Bug
- **severity:** P2
- **file:** coco_s4/phase_victim.py
- **function:** run matched_clean_initial_state receipt
- **VOC_expected:** Compare canonical model-state SHA with canonical state SHA
- **COCO_actual:** Compares serialized .pt file SHA with state SHA; same initialized models marked unmatched
- **root_cause:** File identity confused with tensor-state identity
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** False
- **AP_interpretation:** Would block correct paired evaluation
- **evidence:** COCO_S4_BEFORE_FIX_MINIMAL_REPRODUCTIONS_20261026.json; phase_official_eval.accepted_dependencies
- **minimal_reproduction:** COCO_S4_BEFORE_FIX_MINIMAL_REPRODUCTIONS_20261026.json; phase_official_eval.accepted_dependencies
- **repair_plan:** Use canonical state digest for matching, retain file SHA separately
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-006 / P2

- **id:** S4-006
- **classification:** Confirmed Bug
- **severity:** P2
- **file:** ue_framework/methods/s4_tfss.py
- **function:** rng_state/restore_rng
- **VOC_expected:** CPU validation creates no CUDA context; formal CUDA generator state preserved
- **COCO_actual:** get_rng_state_all invokes lazy CUDA init even for CPU model on server
- **root_cause:** CUDA-only helper reused for CPU validation
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** False
- **AP_interpretation:** CPU isolation claim unreliable
- **evidence:** COCO_S4_CPU_RNG_BEFORE_FIX_20261026.json
- **minimal_reproduction:** COCO_S4_CPU_RNG_BEFORE_FIX_20261026.json
- **repair_plan:** Capture CUDA RNG only if initialized; restore nonempty CUDA state; verify formal initialized behavior exact
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## S4-007 / P1

- **id:** S4-007
- **classification:** Confirmed Bug
- **severity:** P1
- **file:** coco_s4/phase_contract.py; runtime.py; recovery_control.py
- **function:** release/admission gates
- **VOC_expected:** Superseding scientific invalidation/latest no-formal scope overrides stale native acceptance
- **COCO_actual:** No invalidation consumption guard; repair probes still routed via historical branch/run constants
- **root_cause:** Lifecycle contracts predate emergency cancellation
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** True
- **AP_interpretation:** Native ACCEPTED cannot legalize invalid Fourier or trigger automatic relaunch
- **evidence:** COCO_S4_EMERGENCY_STOP_EVIDENCE_20261026.json; phase_contract.phase_release; dimension_release.PREFLIGHT
- **minimal_reproduction:** COCO_S4_EMERGENCY_STOP_EVIDENCE_20261026.json; phase_contract.phase_release; dimension_release.PREFLIGHT
- **repair_plan:** Explicit emergency-validation-only release; block formal prepare/launch/READY in repair revision and invalidated workflow
- **scientific_change_risk:** Restore original math/identity; CPU RNG guard must preserve initialized CUDA path exact
- **recalibration_required:** False
- **downstream_effect:** Block affected stage until fixed and actually validated
- **status_at_freeze:** OPEN

## Limitations

- Real repaired learnedTeacher CPU/GPU tests remain OPEN
- No formal export/victim/final metrics
- No saved learned coefficients in invalidrun; unsaved memory cannot be archived retroactively
- Further confirmed findings require additive BEFORE_FIX entry before repair

## BEFORE_FIX addendum S4-008

- **id:** S4-008
- **classification:** Confirmed Bug
- **severity:** P2
- **file:** coco_s4/canonical_s4_training.py
- **function:** configure/frozen_input
- **VOC_expected:** Frozen consumes shared-EOT-transformed weighted poison after native letterbox
- **COCO_actual:** 433 prehook counts only tensor identity is _last_poison; original sharedEOT creates a different tensor then letterboxes, so real Frozen count stays0
- **root_cause:** Passive observer targets pre-EOT identity at post-letterbox boundary
- **affects_loss_or_gradient:** False
- **affects_poisoned_export:** False
- **AP_interpretation:** False-negative TFSS admission; toy test failed to cover real source-to-sink geometry
- **evidence:** COCO_S4_EOT_OBSERVER_BEFORE_FIX_20261026.json; SemanticFrozenTrainer._batch_loss; BaseGenerator._forward_raw
- **minimal_reproduction:** {'actual_original_EOT_returns_new_poison_tensor': True, 'candidate433_identity_hook_can_observe_Frozen': False, 'model_forwards': 0, 'GPU_initialized': False}
- **repair_plan:** Observe canonical sharedEOT pair boundary and actual Frozen letterbox forward without altering output/RNG, require both exact links
- **scientific_change_risk:** Instrumentation only; original EOT/Frozen arithmetic retained
- **recalibration_required:** False
- **downstream_effect:** Real TFSS smoke must reject until corrected
- **status_at_freeze:** OPEN
