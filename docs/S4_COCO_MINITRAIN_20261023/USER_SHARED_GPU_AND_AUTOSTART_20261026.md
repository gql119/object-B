# Direct human instruction: shared GPU admission and conditional formal autostart

Date: 2026-10-26, this chat.

> 我觉得当前GPU的占用者占用显存很少，不影响我们的实验与他的实验同步进行，直接开始我们的实验后续复验，复验通过后开始正式训练，如果不通过自行修复BUG然后继续复验直到可以正常开始正式训练

This authorizes coexistence with another user's GPU process when measured capacity passes. Other-user GPU occupancy alone is no longer an exclusive-admission blocker. Retain free GPU memory >=24000 MiB, available RAM >=32 GiB, free disk >=150 GiB and the existing three-window IOwait rule. Keep exactly one own scientific workload/controller. No signals, priority changes, file inspection or modifications to another user's job. Shared compute may change speed; a small memory allocation does not prove zero compute/bandwidth contention.

Continue the last bounded validation, repair reproducible bugs without weakening mathematical/data/numerical checks, and start a fresh uniquely bound formal run once all actual gates pass. No further human confirmation is required within this scope. Preserve all stopped/failed run identities and evidence; do not restart or reuse the invalid v2 workflow. Same sole heartbeat and original RRCTL control mechanism; no duplicate scheduler.

Original S4 8/4/0, TFSS eta=.85, selected coordinate identities, all losses and hyperparameters, fixed official 25K/COCO80/val2017 5K remain unchanged. A substantive mathematical change still requires stopping and reporting; automatic bug repair does not authorize method redesign.
