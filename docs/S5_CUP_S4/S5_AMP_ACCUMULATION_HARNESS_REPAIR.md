# S5 smoke v1: native AMP accumulation observer repair

Change: only the isolated synthetic-overflow smoke observer is repaired. The native training algorithm, GradScaler, optimizer accumulation, learning schedule, batch, workers, AMP, S4/TFSS and CUP formulas remain unchanged.

Result: v1 ended exit1 at 2026-10-18T18:48:09Z after all40 noise Adam updates, two exact3-to5 replays and32 decoded PNG checks. S0 controls match; the16-batch S5 PNG prefix observed4 successful native updates. The synthetic batch11 logged optimizer_attempt_observed=false, scale2048 unchanged, parameters/model/EMA/optimizer finite. The console raises `Isolated forced overflow did not take unchanged native skip/backoff` at the observer boundary, before the next optimizer attempt. Thus the assertion incorrectly demanded an AMP decision on an accumulation-only batch. This is a harness failure; actual forced-overflow recovery remains unverified.

Finding: the observer now selects the first actual native optimizer attempt at or after the injected batch, then still requires skip, backoff and unchanged parameters. Subsequent successful update remains mandatory. No assertion/tolerance/step cap is removed. Five CPU contracts cover same-batch and delayed skip, subsequent recovery, and rejection of missing skip/backoff/parameter preservation. v1 console/status/diagnostic17 retained under its unique output; never reused.

Next: unique v2 current-source full bounded smoke, same98-unit cap; no reviewer or formal S5 until passed. Config metadata changes only s5.smoke_run_id v1 to v2, changing lock SHA; all scientific values remain identical. New source commit recorded by the launch binding.
