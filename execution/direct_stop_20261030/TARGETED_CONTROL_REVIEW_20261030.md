# Targeted lifecycle control review — 2026-10-30

- Decision: **allow_stop: false**; verdict: `lifecycle_safety_incorrect`.
- Reviewer: independent delegated reviewer `/root/at_abort_lifecycle_targeted_review_20261030`.
- Reviewed commit: `7e6d1008c229ca2f91cd7ca63a9579918b4db0b7`.
- Diff base: `64daca67003b3eab2321c7bc22775a90a95d9842`.
- CONTROL SHA256: `71f446902103d47c59f1efabeb5ec930474264fc89039465ff6d441ff7da1542`, independently recomputed from committed code.
- Authorization SHA256: `e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8`, independently recomputed.
- Gated command: `python -B execution/direct_stop_20261030/terminate_repaired.py terminate`.
- Scope: destructive lifecycle safety only. No model/scientific review, remote command or real signal was performed.

## Blocker: session termination exceeds the verified ownership scope

The bound `remote_run_control.worker.abort` checks `tmux list-panes -t spec.session.name -F #{pane_pid}`, accepts **any** pane related to the bound executor, then executes `tmux kill-session` for the entire session. It does not reject unrelated panes, and its query does not enumerate every window with session-wide pane listing. The new process-group guard cannot protect unrelated pane process groups from session-wide termination.

This is reproducible in the actual bound abort with the reviewed single comparison replacement. A pure local stub returned both the owned executor and unrelated pane PID `999999`; the ownership predicate was false for the latter. The abort still invoked the fake `kill-session`, then the fake group TERM, and returned `aborted`. No real signal or tmux command was sent. The JSON report preserves the raw probe result.

`20261030T050905Z_ABORT_DIAGNOSTIC.json` records one returned pane, but it uses that same limited query. `20261030T051614Z_PRODUCTION_READONLY_GUARD.json` records descendant-only group ownership, not every session pane. These do not establish the scope of the impending whole-session action. This does **not** claim a foreign pane actually exists; it identifies a reachable unchecked destructive scope and missing production evidence.

Close this blocker by resolving the exact intended session, enumerating every pane in every window, and rejecting any pane/process scope outside the bound AT run immediately before the existing abort. Preserve the original binding/worker bytes and remaining checks. Record fresh production read-only evidence plus a local rejection probe with an unrelated extra pane and another window. Main-agent closure can verify this repair; the single-review workflow does not require another formal reviewer.

## Other evaluated findings

The human authorization explicitly covers immediate queue termination, normal RRCTL escalation and scoped engineering repair without weaker identity checks. No repeated human confirmation is needed.

The one expected cmdline replacement is justified. CONTROL pins the old RunID, source/spec/worker digests, raw binding bytes, UID1014, both PID start ticks, exact canonical queue argv, executor ancestry and group IDs. It accepts only bootstrap/canonical recorded fingerprints, verifies the live canonical fingerprint, pins the original abort AST, and changes exactly one comparison. Production evidence independently observes the actual exec-related mismatch with stable PID/start identity. No binding or worker file is rewritten.

The original abort retains workload PID/start/group, run markers, RunSpec argv, executor PID/start/cmdline/group, ancestry, tmux relationship and invalid-group rejection. CONTROL verifies all observed group members are UID1014 descendants of the executor. Recorded production identities are executor `3338788/389474995`, queue `3338796/389475008`, scientific child `3339007`, and PGID `3338788`. The direct group signals remain confined to this verified group; session collateral is the separate blocker above.

Retention is preserved in code: the frozen production spec has `output_cleanup:null`, so the bound cleanup returns without mutation. The helper writes a separate receipt and compiles the repair in memory. Prestate records 85 checkpoints, the epoch085 checkpoint SHA256, and existing logs. Actual post-stop integrity remains to be checked by the command's VERIFY step; this review does not label unexecuted verification complete.

The original queue at source commit `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb` waits for a successful scientific worker and an ACCEPTED receipt before advancing, with no TERM handler/restart path. The queue supervisor and current worker are both in the targeted group. Recorded prestate is epoch86 of Clean-AT4, with no accepted stages. The command's post-stop verification rejects surviving original processes, own GPU processes and remaining-arm output directories. There is no new scientific-child-only termination path.

CPU02 is real bounded lifecycle evidence: the fixture group was reaped, RRCTL became aborted and original AT remained healthy. Four negative requests rejected UID/start/spec/binding mismatches before signals. Its bootstrap fingerprint branch is explicitly simulated, while actual production mismatch is independently recorded. CPU02 does not exercise unrelated panes/windows or prove production checkpoint retention. CPU01's failure remains recorded. Smoke is not being treated as proof of all safety invariants.

All findings and source/evidence references are retained in `TARGETED_CONTROL_REVIEW_20261030.json`. The stop remains gated by the single session-scope blocker.
