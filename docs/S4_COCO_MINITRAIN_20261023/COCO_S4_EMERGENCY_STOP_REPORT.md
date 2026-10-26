# COCO S4 emergency stop report

The invalid workflow s4_coco_experiment_a_seed0_20261026_dimcompat_v2 is STOPPED_SCIENTIFIC_INVALID, reason MISSING_TFSS_OPTIMIZATION_WIRING. Before signalling, UID1014, child64595/start357618769, controller63444/start357606197, exact command/executable/cwd, source dcd91b3, release63eb8614, config8228eca8, output and GPU ownership were verified. The GPU query identified only the matched task PID for this stop.

A cancellation request and independent snapshots of original native state/config/release/READY/job identities were saved first under owned emergency_stops/COCO_S4_TFSS_20261026T1231Z. SIGTERM was sent once to exact PID64595 only. The native job recorded returncode -15 at2026-10-26T12:33:40Z (20:33:40 Hong Kong); the existing serial controller recorded FAILED with noautomaticretry and exited. A race-safe follow-up at12:34:47Z confirmed both PIDs absent, no matching GPU process and no remaining workflow process. No group/global kill or SIGKILL was used. The first observer raced /proc/cwd after successful signal delivery; the verifier failure is retained and does not imply failed termination.

No export READY/output or Victim existed. An immutable superseding receipt, outputs/<workflow>/receipts/SCIENTIFIC_INVALIDATION_20261026.json, invalidates any old native acceptance without rewriting old evidence. Current artifacts remain in place and are INVALID_FOR_ORIGINAL_S4; they are prohibited as a resumed initialization, export source or paper result. No cleanup/deletion was performed. An artifact inventory/reuse audit follows; missing in-memory-only state will be reported rather than fabricated.

The native controller exits on nonzero child return and contains no retry loop. Future registered consumers must also reject the scientific invalidation receipt and missing TFSS proof. The same sole heartbeat has been updated to audit/fix/validation ONLY. Formal24epoch Fourier and all formal downstream jobs are disabled pending a later explicit human relaunch instruction.

See COCO_S4_EMERGENCY_STOP_EVIDENCE_20261026.json for exact native exit, original snapshot hashes, source and process evidence.
