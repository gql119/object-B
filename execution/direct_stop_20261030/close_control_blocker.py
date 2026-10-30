import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from guarded_abort_repair import CONTROL
HERE=Path(__file__).resolve().parent
raw=HERE/'TARGETED_CONTROL_REVIEW_20261030.json'
review=json.loads(raw.read_text());assert review['allow_stop'] is False
assert [v['id'] for v in review['findings'] if v['status']=='blocker']==['LIFECYCLE-01']
digest=hashlib.sha256(CONTROL.encode()).hexdigest()
cpu=json.loads((HERE/'s4_at_abort_exec_guard_20261030_cpu03/CPU_ABORT_MAIN_CLOSURE.json').read_text())
local=json.loads((HERE/'ALL_PANES_GUARD_LOCAL_CASES.json').read_text())
guard_path=sorted(HERE.glob('*_PRODUCTION_READONLY_GUARD.json'))[-1]
v=json.loads(guard_path.read_text());assert not v['returncode'];g=json.loads(v['stdout'])
assert cpu['control_source_sha256']==local['control_source_sha256']==digest
assert cpu['state']=='PASS_REAL_RRCTL_CPU_ABORT_WITH_POSTEXEC_GUARD' and cpu['original_AT_healthy']
assert local['valid_single_pane_passed'] and len(local['rejected'])==4
assert g['all_windows_exactly_one_owned_pane_verified'] and g['session_ownership_revalidated_at_termination_sink']
assert g['fixture_bootstrap_capture_simulated'] is False and g['signals_sent'] is False
result=dict(state='PASSED_ALL_REPORTED_BLOCKERS_CLOSED',UTC=datetime.now(timezone.utc).isoformat(),
    raw_review_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),raw_negative_review_preserved=True,
    reviewed_commit=review['reviewed_commit'],repaired_code_commit='5e90706ea295e66928cb282180b3fac2548acfab',
    control_source_sha256=digest,closed_blockers=[dict(id='LIFECYCLE-01',
        fix='Enumerate every window/pane via list-panes -s on exact session; require exactly one pane equal bound executor. Revalidate identity and whole-session scope at kill sink; target captured exact session ID. Preserve all other abort and retention checks.',
        evidence=['ALL_PANES_GUARD_LOCAL_CASES.json','s4_at_abort_exec_guard_20261030_cpu03/CPU_ABORT_MAIN_CLOSURE.json',
                  's4_at_abort_exec_guard_20261030_cpu03/CPU_ABORT_RESULT.json',guard_path.name])],
    scope='main-agent closure of single targeted lifecycle review, not a second review or AT scientific approval',
    original_AT_signals_sent_so_far=0,allow_authorized_old_task_termination_after_fresh_guards=True)
with (HERE/'TARGETED_CONTROL_MAIN_REPAIR_CLOSURE.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
