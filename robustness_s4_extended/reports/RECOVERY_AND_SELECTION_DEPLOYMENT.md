# Recovery and automatic-selection deployment
Observed UTC: 2026-10-22T15:40:24.483615+00:00. Training progress is a snapshot, not a formal result.

Full Victim recovery03 is running: completed25/200, AMPepoch26,6191 actual updates,
6legal AMPskips/consecutive0; model/EMA/optimizer finite. The completed Full
24epoch/9144 noise and new16551train/4952cleanval dataset remain unchanged.
External oomd termination, recovery01 session coupling, and recovery02 bootstrap
guard failure are documented and retained. Recovery03 uses separate ordinary
controller/supervisor/worker scopes plus detached sessions, with no resource
protection or hyperparameter changes. No exact interrupted-batch resume is claimed.

Full science1d10ad620ef01aad20e8c85d49465473d72c9a4e;
lifecycle runtime dc60396d044cefe08fd03ae6b126d33135ff11b2;
scientific protocolSHA2bedd986d44dce40d3815093da76ef3d259aec83fab7f0c6518d0f7795b6675c.
Full branch: codex/s4-tfss-full-overlap-20261021.
Recovery9CPUcontracts and actual source-bound CPUv3 bootstrap/orphan-controller
smoke passed. Original Full independent scientific review count remains1.

## Selected-method successor

Active unique BatchID: s4_robustness_extended_selected_seed0_20261022_v2.
Runtime8644e14cb6804bf9774f8a56c777be155bc64550;
queueSHA97a4ae1ee4788ba1485783ba0b70d2e24ec93464b235c037ce9c953cbe1571e3.
Controller3040375/start330732333/uid1014 is healthy WAITING_PREVIOUS_PIPELINE;
jobs{}/activeNone/newGPUsteps0. Original waiter was normally superseded with no
training jobs; selectedv1 correctly blocked for invalid Full and is retained.
There is one active extended controller and the same sole two-hour monitor.

48local +48actualserver CPU contracts passed0fail/errors/skips, including Full
better/notbetter/invalid, mismatch/undefinedscore/rule drift, selected-data mixing,
existing-decision reuse, predecessor recovery identity and native detection
augmentation cases.129committed source files and12installed native files were
independently byte/SHA-verified. Actual production singleton/scope/completed-job
adoption passed. CPU tests read synthetic images only and perform0GPUsteps.

No method is selected yet. The rule is frozen before Full official results:
S improvement >1e-9, PersonAP50<=S4, NTmAP50>=S4, WCD<=S4, protocols identical.
Valid but not better ->originalS4. Invalid Full ->BLOCKED_FULL_VALIDATION, no fallback.
Actual S4 last@200/CSV/data and Full lineage/200/lastbest/fullVOC20/cleanval/hash
are verified before selection. The one exclusive Decision JSON binds method,
data/manifest/integrity/rule/source/config/checkpoint and BatchID. All later
Clean/Poison arms, GPU smokes, scientific review, acceptance and reports read it.

Server order: Full accepted -> compare/select/lock -> original8defense/source/data
checks -> three bounded GPU smokes -> one extended scientific review service ->
CutoutClean/Poison -> CutMixClean/Poison -> MixUpClean/Poison -> archive partial
results ->OP BLOCKED_PROTOCOL_APPROVAL. No new augmentation GPU run has started.
The server schedules/waits/trains independently. Its scientific-review gate waits
for a real external receipt; the existing sole monitor supplies that once actual
current-source GPU evidence is ready. There is no extra method-choice approval.

Cutout: probability1,n_holes1,nominalarea.25,fill0; anyGT>.5occluded rejects entire
candidate,max8then skip,fullGT retained,all200epochs.
CutMix: nativeDetectp1/Beta1/3areas,primaryGTzero intersection,donor>=.5visible/min2px.
MixUp: nativeDetectp1/Beta32,32,integerinstance concatenation.
Baseline Mosaic/HSV/flip and close_mosaic100 retained; MixUp/CutMix offlast100.
Every pair uses the same random-YAMLseed0/200/batch36/workers16/AMP/nativeSGD/
cacheTrue/640/originalLRscheduler and unchanged clean4952validation, lastprimary,
bestsecondary. OP CPU prototype remains disabled; no approved detection protocol.

Estimated new training budget:6 fresh200epoch Victims, approximately552000 train
batches total plus the locked native epoch-validation and12final checkpoint
evaluations. Wall-clock time is not measured or guaranteed on the shared server.
No OP model training/projection/victim budget is authorized yet.

## Read-only status command

Use the existing Python directly (no PowerShell ExecutionPolicy change):
```powershell
& 'C:\Users\20272\.codex\visualizations\2026\09\17\01a0afba-bca4-7582-815c-b2a5241bad4a/five_method_fidelity_20261013/.venv-five/Scripts/python.exe' -B 'C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_robustness_extended_20261022/execution/query_extended_status.py'
```
This queries server state/counters only, with no GPU, model import, signal or remote
health write. No SwordAgent/VPN test is needed while connection works.

## Actual scoped changed files

- robustness_s4_extended/README.md
- robustness_s4_extended/build_preparation.py
- robustness_s4_extended/common.py
- robustness_s4_extended/configs/FULL_REFERENCE_LOCK.json
- robustness_s4_extended/configs/QUEUE.json
- robustness_s4_extended/configs/SELECTION_RULE.json
- robustness_s4_extended/control.py
- robustness_s4_extended/predecessor.py
- robustness_s4_extended/reports/AUTOMATIC_METHOD_SELECTION_PROTOCOL.md
- robustness_s4_extended/reports/CPU_TEST_RESULT.json
- robustness_s4_extended/reports/SELECTION_USER_AUTHORIZATION.md
- robustness_s4_extended/selection.py
- robustness_s4_extended/successor.py
- robustness_s4_extended/tests/test_defenses.py
- robustness_s4_extended/tests/test_selection.py
- robustness_s4_extended/worker.py
Full recovery additions are scripts/recover_s4_tfss_full_victim.py,
scripts/test_full_victim_recovery.py and
docs/S4_TFSS_FULL_20261021/OOM_VICTIM_RECOVERY_20261022.md.
No old scientific implementation, dataset, weight or credential is published.
Later documentation publication commits are distinct from runtime identities.
