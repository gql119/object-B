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

The first fresh candidate, `coco_teacher_eval_20261025_0203z`, passed RRCTL
thin readiness and reached the prediction entrypoint, then exited with code 1
before review or formal registration. Its retained status says the production
forward produced finite outputs for 72 images, but the audit recorded `cpu`
from the wrapper model parameters after prediction and rejected the phase. That
post-prediction parameter device does not establish the device of the actual
pre-NMS forward, so this smoke did not prove CUDA reachability and was not
accepted. The candidate is preserved and will not be retried. Exact RRCTL
process identities are now absent; the serial controller is still waiting at
`teacher_eval` with no active job. No official metrics, checkpoint changes, or
cleanup occurred.

The follow-up implementation records tensor devices inside both the production
pre-NMS forward hook and the independent native reference hook. The CUDA0
requirement remains strict, and both paths must report the same actual forward
device. The paired official-evaluation path uses the same forward-device
evidence. A new published commit and unique candidate are required before
another smoke; no review is performed for the failed candidate.

After this telemetry correction, the retained two-image random COCO80 CPU
prediction probe passed with raw and decoded predictions identical, the
production and reference forward tensors both reported `cpu`, nonfinite raw
outputs were rejected, and CUDA remained uninitialized. This is CPU-only
contract evidence, not a CUDA smoke or scientific review; see
`execution/TEACHER_EVAL_NATIVE_CPU_DEVICE_AUDIT_20261025.json`.

The first post-fix CUDA smoke recorded `cuda:0` from the actual pre-NMS
prediction tensors and the independent reference tensors, but its summary did
not carry those fields to the phase registration gate. It remains preserved as
raw CUDA evidence and was not registered. The follow-up contract now records
CUDA indices directly from each forward tensor, propagates the candidate and
reference device evidence into the summary, and cross-checks the pulled raw
prediction audit before registration. This changes telemetry and its gate only;
checkpoint, inference settings, data, metrics policy, and S4 method are fixed.
