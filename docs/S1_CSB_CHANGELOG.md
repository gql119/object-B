# S1-CSB change log

## RECOVERY-001 — parent runtime bytes

Original Design: archive object-B commit0d43da4.

Observed Problem: archive checkout bytes differed from October4 runtime fingerprints (BOM/line endings, one default Kaggle path); active mathematical files matched once runtime bytes were restored.

Modification: recover exactly the 45 explicit Python/YAML runtime files in a new worktree; preserve bytes with .gitattributes. No historical file overwritten. Classification: BUG_FIX (source identity recovery, no mathematical change).

Scientific Justification: the evaluated parent's actual computation must be the reference.

Experimental Evidence: parent_remote_identity.json; numerical equivalence still pending.

Impact on Comparability: restores reference source identity; no CSB yet.

Final Decision: retain; do not claim loss/gradient equivalence before measured probe.

## PROBE-002 — absolute worker config binding

Original Design: parent equivalence subprocesses receive the same config argv.

Observed Problem: v1 failed before model computation because changing worker cwd to the new output made a relative config path unavailable.

Modification: resolve config/prototype paths before spawning workers; fresh probe RunID v2. Classification: BUG_FIX (launcher binding only).

Scientific Justification: both workers must consume identical existing config bytes.

Experimental Evidence: retained v1 rrctl console traceback; no loss or optimization result from v1.

Impact on Comparability: no loss, RNG or optimization setting changes.

Final Decision: retain; v1 failure preserved.

## CSB-003 — shared final poison binding

Original Design: override only final spatial projection, retaining inherited B2 losses.

Observed Problem: mechanism v1 completed native/fixed arms, then failed in CSB at B2's frozen-gradient diagnostic. MRO had cached the parent's old poison before S1 projected it; the frozen branch consumed the new tensor while Adaptive cached the old one. Traceback: autograd tensor unused.

Modification: S1's projection wrapper binds `_last_poison` to its final projected poison. Original B2/Adaptive/Frozen code is unchanged. New mechanism RunID v2. Classification: BUG_FIX, not a CSB parameter/design adjustment.

Scientific Justification: both parent branches must consume the same constrained support image.

Experimental Evidence: v1 console/failed status retained. Numerical closure requires the v2 production probe.

Impact on Comparability: corrects intended shared input; q10/q90, eta.25, gamma1, losses and optimizer unchanged.

Final Decision: retain pending passing production probe; formal blocked.

## RECOVERY-004 — committed blob byte identity

Original Design: restored worktree source bytes preserved by -text attributes.

Observed Problem: the production v3 config hash was133d3c rather than historical80cffe. Worktree bytes matched, but Git's unchanged-file index cache had retained LF blobs for5files despite new attributes (adaptive learner, orth reference, formal config, materializer, victim). Numeric tests passed because differences were line endings, but formal source/config identity checks would correctly refuse to run.

Modification: explicitly renormalize only those5files under -text attributes, preserving exact historical bytes in committed blobs; targeted test now checks both workingtree and Git HEAD bytes. Classification: BUG_FIX, transport/source identity only.

Scientific Justification: the deployed source and formal config must match historical identities.

Experimental Evidence: v3 actual config hash and local git-show SHA comparison;5matching committed blobs are required before the final probe.

Impact on Comparability: no source statements, configuration values, loss, RNG or scientific settings changed.

Final Decision: retain; v3 remains separate evidence and final exact-source admission uses fresh v4.
