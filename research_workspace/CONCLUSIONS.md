# TFSS research conclusions

- C001 | decision | ACTIVE | scope=tfss.short_protocol: Human authorizes isolated code snapshot publication and train-only C/U/pairedUniform short comparisons, after resource/correctness gates. Source docs/tfss/USER_TASK_SOURCE.md. Formal TFSS is not authorized.
- C002 | execution | OBSERVED | scope=tfss.source: Snapshot a5aefbb is pushed/reread; S0 exact45 source restored. Evidence docs/tfss/S0_RESTORATION_AUDIT.json. Historical runtimeGitSHA remains unknown.
- C003 | finding | SUPPORTED | scope=tfss.component_correctness: Eight CPU numeric contracts pass, including actual RMS/image clamp and detached scaling/backward preservation. Evidence docs/tfss/CPU_VALIDATION.json. Bounded component evidence only, not real VOC/GPU mechanism admission.
- C004 | hypothesis | OPEN | scope=tfss.efficacy: C/U may retain S0 target protection while improving NT utility; no 5step or victim data exists. Current recommendation INSUFFICIENT_EVIDENCE, no AP claim.

- C005 | finding | SUPPORTED | scope=tfss.replay: deterministic paired CUDA control repair yields actual canonical/noop maxdiff0 at one update/first16; not a full smoke pass. Evidence execution/tfss_smoke_v2_observation_20261017.json.
- C006 | finding | MIXED | scope=tfss.uniform_numeric_gate: actual GPU Uniform strict RMS gate failed; CPU saved-delta replay found11/16 targets not bracket-matchable within1e-9 in current float32 arithmetic. No matched-control efficacy conclusion. Evidence docs/tfss/RMS_GATE_BLOCKER_20261017.md.
- C007 | decision | PROPOSED | scope=tfss.numerical_protocol: closer-bracket solver plus BOTH abs1e-7/relative1e-4 RMS gates is unapproved. Existing1e-9 stays locked; WAIT_FOR_HUMAN, no automatic retry/reviewer/short/full run.
