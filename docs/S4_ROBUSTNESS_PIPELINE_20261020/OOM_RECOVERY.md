# OOM scope diagnosis and recovery

The 2026-10-20 19:42:56 HKT user journal confirms systemd-oomd killed52 processes in tmux-spawn-75a2c4d6-7545-4916-90d5-201aae402dc7.scope. Its start record identifies tmux pane2405347, exactly the bound original RRCTL executor. Unit result is oom-kill. SSH remained available; no host reboot. Empty stderr and absent Python traceback do not contradict an external group kill.

Original Gray S4 completed the first training prefix (454 native optimizer updates,6 legal AMP skips, finite model/EMA/optimizer); last stdout reached validation69/69. There is no saved results.csv,checkpoint or exit receipt at the bound attempt. No final Gray metric exists. The historical memory.peak of the whole user slice does not identify the exact allocation responsible for pressure; we do not infer that from CPU utilization or the last finite AMP record.

Original control/supervisor/training shared the tmux pane cgroup even though the supervisor used start_new_session=True. The fix launches each job in a separate ordinary systemd user scope; external supervisor/controller remain outside the training scope. Child PID/startticks/cgroup and memory pressure/peak are recorded. Child exit plus scope Result are retained even on signal termination. No OOM settings, priorities, memory limits, Conda environment, CPU/GPU admission guard, worker count or scientific parameter is changed. This limits the failure blast radius and permits the existing bounded environmental retry policy. It cannot promise that future host/user-wide pressure will never stop another process.

Existing pipeline and Gray attempt remain untouched. Recovery gets a fresh control/output RunID and Gray paired recovery output, reuses accepted JPEG50/JPEG30 by audit, and starts incomplete Gray from random seed0/full200. All other stages remain serial and unchanged. Original scientific e17c24c54aae4cdae4bcc4703facf2a57052d2c8, robustness lock00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91 and original S4 data identities stay frozen. No second reviewer or AT.

Validation: targeted local contracts then production RRCTL CPU-only synthetic scope smoke. The synthetic child self-signal9 tests receipt/controller survival, not a real OOM injection. Do not disable OOM protection or manufacture memory pressure. Actual formal recovery waits for these gates and fresh unchanged resource admission.

Evidence retained locally: execution/PIPELINE_CAUSE_SYSTEM_LOGS_20261020.json; execution/PIPELINE_OOM_SCOPE_AUDIT_20261020.json; execution/PIPELINE_INTERRUPTION_READONLY_20261020.json.
