import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
receipt=sorted(HERE.glob('*_OLD_AT_TERMINATED.json'))[-1];stop=json.loads(receipt.read_text());assert stop['state']=='OLD_AT_TERMINATED'
pre=json.loads(Path(stop['prestate']).read_text());p=pre['before']['progress']
state_path=DOC/'CURRENT_STATE.json';s=json.loads(state_path.read_text(encoding='utf8'))
s.update(state='OLD_AT_TERMINATED_CUDA_OPTIMIZATION_PENDING',UTC=stop['UTC'],active_formal_controllers=0,
 current_arm=None,formal_run_allowed=False,AT_results=None,
 next='Latest human direct-stop authorization supersedes natural-boundary/pause/net10 gates. Old run is ABORTED, inspect-only; perform serial published-source finite CUDA equality/ABBA/batch testing, one scientific review, immutable four-arm epoch0 launch or originalBatch36 fallback.')
s['direct_stop']=dict(state='OLD_AT_TERMINATED',old_RunID=stop['run_id'],RRCTL_terminal='aborted',
 evidence=receipt.relative_to(ROOT).as_posix(),checkpoint=stop['checkpoint_retained'],GPU_released=True,
 old_completed=False,old_accepted=False,authorization_sha256='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8',
 previous_natural_boundary_wait_superseded=True,previous_pause_resume_gate_superseded=True,previous_net10_restart_gate_superseded=True)
s['current_progress']={k:p[k] for k in ('epoch','completed_epochs','at_batches','actual_updates','legal_amp_skips','loss','UTC')}
s['current_progress']['terminal']='aborted'
state_path.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
plan=dict(state='AUTHORIZED_FINITE_CUDA_CAMPAIGN_AFTER_VERIFIED_OLD_TERMINATION',old_receipt=str(receipt),
 source_commit='dbdc05f938d1e553fbd62af32b2fa36f908c68d0',published_archive='64daca67003b3eab2321c7bc22775a90a95d9842',
 max_GPU_campaign_seconds=7200,max_native_steps_per_RunID=100,serial_only=True,old_resume_forbidden=True,
 order=['Candidate02/03 cleanAT4 exact CUDA equality at36','Original/Candidate02/Candidate03 ABBA at36 after equality',
 'remaining arm equality and chosen implementation batch36/72 then48/64','four-arm correctness/tail/resume for selected protocol',
 'current final-production-source bounded GPU evidence','exactlyone scientific review and mainclosure',
 'fresh epoch0 four-arm serial launch; originalBatch36 fallback if no reliable valid gain'],
 net10_old_progress_recovery_gate=False,no_posthoc_tolerance_relaxation=True)
with (HERE/'DIRECT_CUDA_CAMPAIGN_PLAN.json').open('x',encoding='utf8') as f:json.dump(plan,f,indent=2)
note=f'''\n\n## OLD AT ACTUALLY TERMINATED — {stop['UTC']}\nLatest direct human authorization supersedes old natural-boundary/pause/net10 gates. Existing old RunID {stop['run_id']} is RRCTL ABORTED sequence6, never COMPLETED/ACCEPTED. Full owned group and GPU processes gone; other3 arms never started. Checkpoint epoch86 SHA {stop['checkpoint_retained']['sha256']} unchanged; old logs retained, no deleted paths, old binding/worker immutable. Scoped canonicalexec guard repaired the initial RRCTL fingerprint refusal; single targeted lifecycle review negative LIFECYCLE-01 preserved and main exact-session guard closure passed with CPU03/live evidence, no second review. See {receipt.relative_to(ROOT).as_posix()}. Stop is DONE: never re-abort/relaunch/resume/health-poll the abandoned old training. Keep original lock/evidence. Next actual work is finite serial CUDA tests from published dbdc source, 120min campaign budget, then one scientific review/new immutable all4arms freshseed0/200epoch or OriginalBatch36 fallback. No waiting for195..199. Same sole monitor stays active and must follow new probes/newRunID.\n'''
with (ROOT/'execution/AT_CONTINUE.md').open('a',encoding='utf8') as f:f.write(note)
report=f'''# Original S4 AT direct stop / CUDA optimization / restart report\n\nState: OLD_AT_TERMINATED. Updated {stop['UTC']}.\n\nOld RunID {stop['run_id']} was explicitly abandoned by human at epoch{p['epoch']}, {p['completed_epochs']} completed epochs / {p['at_batches']} batches / {p['actual_updates']} actual SGD / {p['legal_amp_skips']} legal AMP skips. It is aborted, not accepted. Existing UID1014 full process group was reaped through scoped repaired RRCTL abort; no own GPU PID remains; next arms absent. Epoch86 checkpoint SHA {stop['checkpoint_retained']['sha256']} and logs preserved.\n\nTermination evidence: {receipt.relative_to(ROOT).as_posix()}. Single targeted control review and main repair closure: execution/direct_stop_20261030/TARGETED_CONTROL_REVIEW_20261030.json and TARGETED_CONTROL_MAIN_REPAIR_CLOSURE.json. This is not the subsequent candidate scientific review. Initial fingerprint and CPU01 setup failures remain retained.\n\nCandidate CUDA equality, throughput, batch selection, new scientific review, new formal Release/RunSpec/controller are NOT COMPLETE yet. Actual values remain null until measured. The current authorized fallback is a fresh four-arm OriginalBatch36 protocol if no validated reliable speed gain. All4 seed0 randomYAML200epochs/PGD10/fullGT/640/fullVOC4952 remain fixed. Old checkpoint continuation is forbidden by latest scope.\n'''
(DOC/'S4_AT_DIRECT_STOP_CUDA_OPTIMIZATION_AND_RESTART_REPORT.md').write_text(report,encoding='utf8')
print(json.dumps(dict(state=s['state'],old_terminal='aborted',checkpoint_retained=True,GPU_released=True)))
