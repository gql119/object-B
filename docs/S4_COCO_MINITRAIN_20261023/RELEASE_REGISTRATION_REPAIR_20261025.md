# COCO RRCTL phase-release registration repair

Recorded 2026-10-25 01:57 UTC. The active workflow remains the already-bound
`s4_coco_experiment_a_seed0_20261025_v3`; its immutable Teacher runtime is not
modified by this repair.

The first `teacher_eval` smoke candidate reached RRCTL's read-only readiness
check and was rejected before launch because `anchors[0].remote_path` pointed
inside that candidate's `repo_root` lifecycle directory. That unique failed
candidate tag is retained and must not be reused. No GPU child, official
metrics, phase registration, or scientific review resulted from the rejection.

Candidate release files now use a separate owned anchor at
`/project/guoqinglong/projects/s4_coco_experiment_a_20261025/releases/_anchors/<tag>/source_release.json`,
which is outside the candidate `repo_root`, `output_root`, `stage_root`, and
`control_root`. The smoke command reads that exact anchor. After the bounded
smoke and one current-candidate review pass, registration copies the exact
same release bytes into `<candidate repo_root>/source_release.json`, verifies
the SHA by remote readback, uploads the review and closure, then uploads the
`READY` receipt last. The existing serial controller remains the only formal
workflow owner. Existing files can only be reused when their hashes match;
conflicts are refused without overwrite.

The shared registration path is wired for Teacher assessment, calibration,
Fourier, poisoned export, matched victims, and paired official evaluation.
Phase-specific registration checks retain each phase's accepted predecessor
receipt and scientific evidence requirements. This implementation does not
mark any phase ready by itself.

Local validation passed three focused CPU tests: lifecycle-anchor separation,
rejection of an anchor nested in `repo_root`, and exact/idempotent release,
review, closure, and `READY` upload order with a deterministic fake transport.
Python AST parsing and all six release-helper `--help` import paths also pass.
These checks do not establish RRCTL remote readiness or GPU behavior. A fresh
unique `teacher_eval` candidate still needs a real RRCTL bounded CUDA smoke,
exactly one independent review, and closure before the serial workflow can
start the full fixed-last COCO evaluation.
