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
