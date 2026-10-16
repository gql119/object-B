# TFSS research conclusions

- C001 | decision | ACTIVE | scope=tfss.short_protocol: Human authorizes isolated code snapshot publication and train-only C/U/pairedUniform short comparisons, after resource/correctness gates. Source docs/tfss/USER_TASK_SOURCE.md. Formal TFSS is not authorized.
- C002 | execution | OBSERVED | scope=tfss.source: Snapshot a5aefbb is pushed/reread; S0 exact45 source restored. Evidence docs/tfss/S0_RESTORATION_AUDIT.json. Historical runtimeGitSHA remains unknown.
- C003 | finding | SUPPORTED | scope=tfss.component_correctness: Eight CPU numeric contracts pass, including actual RMS/image clamp and detached scaling/backward preservation. Evidence docs/tfss/CPU_VALIDATION.json. Bounded component evidence only, not real VOC/GPU mechanism admission.
- C004 | hypothesis | OPEN | scope=tfss.efficacy: C/U may retain S0 target protection while improving NT utility; no 5step or victim data exists. Current recommendation INSUFFICIENT_EVIDENCE, no AP claim.
