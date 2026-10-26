# Autonomous bug-fix log

## NTCOV01 — strict NTGT supervision undefined

- Level C: methodological boundary extension candidate; not an equivalent engineering repair.
- Original failed runtime0eb162097ab07ea641a9b5167c0fe3d70d38d349, workflow20261027_tfss_repaired_v1,12 completed optimizer updates. All native outputs retained; no restart/resume.
- Real root cause and official GT coordinates verified; see root-cause report.
- Independent candidate isolates only `R_nt`'s zero-native-supervision branch; original native estimator and all framework method files remain unchanged. Both training/export use explicitly policy-bound adapter caches.
- Level A preventive fix: future release helpers referenced an obsolete Git branch. Candidate helpers now reference the isolated risk branch; no future phase was launched.
- Candidate actual CPU/CUDA/whole-data statistics remain OPEN. No formula, AP or completion claims from component checks.
- Current human scope: execute design, isolated implementation, actual validation and reporting; no full formal Original S4 recovery for a LevelC definition extension.


## Actual follow-up: component and one-image full chain passed

Harness source a0c32b9c2c5a4df8706f726e8e2b8ff45b081228 was normally pushed, ls-remote exact and GitHub API exact. The default VOC reference versus itself had map/input-gradient differences ~1.8e-15. A common deterministic comparison domain gave exact reference/reference, reference/candidate and candidate/candidate outputs and gradients, tolerance zero. Only the validator changed; estimator/cache/original framework bytes are unchanged. The original GPU01 remains FAILED.

GPU02 passed10 native forwards, zero parameter updates, actual raw cuda:0/index0. Covered COCO and VOC records/maps/full input gradients were exact with repeat controls. FailedGT13/class24 retained all14 NTGT once and used5 separately labelled risk candidates; native positive count remained0. Main independently checked152 source files,7 native files and Teacher last/full-state/receipt hashes with no model forward/GPU initialization.

Full-chain CPU01 and CUDA01 each passed15 native forwards, EOT2 and exactly1 isolated Adam update on declared image247692 (not nativebatch16). Direct mean-EOT BG/rel coefficient gradient norms: CPU .1142069623/.1140771732; CUDA .1270332336/.1258603483. These are separate backend measurements, not a claim of CPU/CUDA equality. Both independently audited TFSS -> Frozen/EOT/letterbox and Adaptive inputs, retained native state/RNG,39D/all79/P3=48, saved16x3 parameters and exact native final-image/export construction. Decoded PNG max2 bytes, overlap max1 byte, outside0 satisfy unchanged8/4/0. No formal dataset was exported.

Mixed CPU02 saved a passed pre-regression stage: all13 originally defined NTGT losses AND full input gradients on the mixed failed image matched exactly, zero tolerance; the one original undefined GT remained null in the reference. All14 NTGT were accounted once. Per-class normalized mean exactly matched the native cache; W ranged .869427681..1. Actual noPerson image25 (2GT) passed1 forward, Person-only image29020 (1GT) returned the original empty NT result with0 forwards. Its overall run remains FAILED because the subsequent legacy regression harness lacked the retained-temp cleanup API needed by a lazy Torch import. CPU01 earlier had2 explicit skips due absent integration fixtures. Neither is counted as a complete regression pass; the harness now supplies a retained randomVOC fixture and no-op cleanup, keeping zero-skip acceptance.

Full25K trigger counts/class frequencies/conditional W distributions remain OPEN. A new population-audit wrapper is implemented but not yet executed: eight-image CPU/CUDA validation and one independent scientific review are required before its complete25K validation-only pass. It never instantiates an optimizer and never substitutes zero for undefined original risk. Full formal Fourier/poison/victims/evaluation remain disabled.
