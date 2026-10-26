# Autonomous bug-fix log

## NTCOV01 — strict NTGT supervision undefined

- Level C: methodological boundary extension candidate; not an equivalent engineering repair.
- Original failed runtime0eb162097ab07ea641a9b5167c0fe3d70d38d349, workflow20261027_tfss_repaired_v1,12 completed optimizer updates. All native outputs retained; no restart/resume.
- Real root cause and official GT coordinates verified; see root-cause report.
- Independent candidate isolates only `R_nt`'s zero-native-supervision branch; original native estimator and all framework method files remain unchanged. Both training/export use explicitly policy-bound adapter caches.
- Level A preventive fix: future release helpers referenced an obsolete Git branch. Candidate helpers now reference the isolated risk branch; no future phase was launched.
- Candidate actual CPU/CUDA/whole-data statistics remain OPEN. No formula, AP or completion claims from component checks.
- Current human scope: execute design, isolated implementation, actual validation and reporting; no full formal Original S4 recovery for a LevelC definition extension.
