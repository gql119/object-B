# S4 server-resident robustness queue

This orchestration change does not modify the scientific implementation. Every
training/data/evaluation invocation imports and verifies the immutable scientific
checkout `e17c24c54aae4cdae4bcc4703facf2a57052d2c8` and frozen configuration
`00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91`.
`PIPELINE_CONFIG.json` hashes the actual computation files and binds the original
S4 data/noise manifest through the existing lock. No S0/S3/S5 poison is consumed.

Order: adopt/audit existing JPEG50/JPEG30, Gray, BDR, Phase A acceptance/reports,
Gauss, MeanF, MedianF, Data Augmentation, Phase B acceptance/final report, archive.
The server runs this order without any client/heartbeat stage-launch commands.
Existing passing real train-batch/AMP probes and G1 full pixel/loader equivalence
are reused and verified, not represented as new tests. BDR is the frozen5-bit
round-to-nearest/ties-even implementation. Historical Clean200 controls are
hash/config checked; unverifiable controls use a new matched Clean200 arm.
An old-metric reevaluation mismatch blocks the stage rather than being hidden.

Single instance: an exclusive Linux flock spans the entire queue. Running the
same command again returns the existing state. Each command has a persisted
launch intent, supervisor PID/start ticks, child PID/start ticks, exact argv,
separate stdout/stderr and an atomic exit receipt. Independent supervisors keep
running if the controller exits. Recovery adopts the same bound child; missing
identity or an ambiguous launch blocks rather than duplicating it.

Accepted stages and completed arms are skipped. A proved200-row native last
checkpoint permits evaluation-only recovery. Incomplete arms cannot claim exact
DataLoader/prefetch RNG restoration: at most two recognized environment failures
may start a new recovery RunID, fresh seed0/full200, preserving the old attempt.
No scientific failure is retried by changing hyperparameters. Unknown failures
stop. Finite raw loss, native AMP skip/backoff handling and all original training
settings are unchanged. No cleanup, deletion, AT or additional seed is scheduled.

Resources: unchanged admission policy keeps RAM32GiB/GPU-free24000MiB/disk150GiB
and three spaced10s I/O-wait windows at30s gaps. CPU/GPU busy/presence/load average
are speed telemetry per the human's existing waiver. Telemetry is retained every
15min. Sustained I/O can stop only the exact own training PID with a stable bound
checkpoint; resume checks the same identity/hash after a fresh clear window.
Idle GPU-free criteria are not applied to active own CUDA usage. No other user's
processes, priorities or files are changed.

RRCTL's detached tmux executor supplies the background lifetime. It survives
client disconnect and Codex exit. Host inspection found user systemd degraded
and Linger=no; this deployment does not change system/user services and DOES NOT
claim automatic restart after server reboot. Persistent state supports manual
reattachment using the same command after verifying the original process is gone;
accepted phases remain skipped. A new RRCTL RunID must never be used to bypass
the singleton queue or an ambiguous live child.

Reattachment automatically adopts a surviving supervisor/child with its durable
identity/exit receipt. A host reboot can remove both without an exit receipt;
that ambiguous incomplete job deliberately BLOCKS pending operator verification
and an explicit recovery record. Re-running a command is not a promise of
automatic post-reboot training recovery. Recognized stage environment failures
have bounded retries; unknown controller/scientific failures are not ignored.

Read-only status (run on the school server; substitute the deployed absolute
script/config paths from execution/PIPELINE_DEPLOYMENT.json):

    <cfbdm-python> -B <repo>/scripts/run_s4_robustness_pipeline.py --config <repo>/docs/S4_ROBUSTNESS_PIPELINE_20261020/PIPELINE_CONFIG.json --status

State/logs live under the configured output_root: pipeline_state.json,
pipeline_progress.json, pipeline_events.jsonl, pipeline.log, jobs/*/stdout.log,
jobs/*/stderr.log, resource_latest.json, resource_admission.json and reports/.
Reports include exact checkpoint/data/source hashes and the complete VOC20 tables.
Only accepted complete pairs enter the tables. AT is NOT_RUN; zero denominators
remain null; signed P/U/S are never clipped. No AP is inferred from smoke/loss.

Validation route is scheduler smoke_validation (not a second scientific review):
local failure/atomic-state/frozen-protocol contracts, then real RRCTL detached
production-supervisor smoke with synthetic CPU stages, crash/adoption, flock
collision and completed-stage idempotence. Synthetic checks are scheduler tests,
not GPU/experimental evidence or official metrics.
