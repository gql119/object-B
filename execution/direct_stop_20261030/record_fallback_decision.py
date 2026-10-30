"""Choose the human-authorized unchanged fallback from actual retained tests."""
import json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
def load(p):return json.loads(p.read_text(encoding='utf8'))
assessments={k:load(HERE/f'ABBA_CANDIDATE_{k}_B36.json') for k in ('02','03')}
batch=load(HERE/'BATCH_ABBA_Coriginal_B72_ASSESSMENT.json')
assert all(v['state']=='INCONCLUSIVE' for v in assessments.values()) and batch['state']=='INCONCLUSIVE'
cpu=load(HERE/'CURRENT_RUNTIME_CPU_REUSE.json');assert not cpu['intentionally_changed_scientific_files']
now=datetime.now(timezone.utc).isoformat()
decision=dict(state='SELECTED_ORIGINAL_BATCH36_FALLBACK_PENDING_CURRENT_GPU_REVIEW',UTC=now,
    selected_implementation='original',selected_batch=36,source_commit=cpu['source_commit'],
    reason='Neither engineering candidate nor batch72 established a reliable controlled speed gain; retain unchanged accepted scientific computation and use expressly authorized fresh fallback.',
    validated_speedup_percent=None,batch48='NOT_RUN',batch64='NOT_RUN',profile='NOT_RUN',
    optional_cases_not_run_reason='Published campaign dispatch deadline passed; avoid indefinite search. No batch48/64 speed or resource claim is made.',
    old_run='ABORTED_NOT_ACCEPTED',old_checkpoint_resume=False,fresh_epoch0_all_four_arms=True,scientific_ready=False,formal_started=False,
    engineering_assessments={k:dict(state=v['state'],reasons=v['reasons']) for k,v in assessments.items()},batch72_assessment=dict(state=batch['state'],reasons=batch['reasons'],raw_images_per_second=batch['images_per_second']))
(HERE/'FALLBACK_SELECTION_PENDING_GATES.json').write_text(json.dumps(decision,indent=2),encoding='utf8')
s=load(DOC/'CURRENT_STATE.json');s.update(UTC=now,state='CURRENT_SOURCE_FALLBACK_GPU_PREFLIGHT_RUNNING',candidate_runtime_commit=cpu['source_commit'],
    source_commit_url='https://github.com/gql119/object-B/commit/'+cpu['source_commit'],source_publication_API_verified=False,
    source_publication_git_remote_SHA_verified=True,active_formal_controllers=0,AT_server_controller_started=False,
    formal_run_allowed=False,selected_batch=36,selected_implementation='original',new_protocol_scientific_review='NOT_STARTED',
    active_CUDA_probe=dict(run_id='s4_at_selected_native_20261030_gpu01',phase='CURRENT_NATIVE_BASELINE_RESUME_SERIAL_PREFLIGHT',source=cpu['source_commit']),
    next='Finish existing finite fallback preflight client; no duplicate launches. Then one scientific review/main closure/current gate and one fresh four-arm controller.')
s['speed_optimization'].update(decision=decision['state'],GPU_speedup_percent=None)
(DOC/'CURRENT_STATE.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
with (ROOT/'execution/AT_CONTINUE.md').open('a',encoding='utf8') as f:f.write('\n\n## Latest actual fallback selection — '+now+'\n'+
    'Old original stays ABORTED. Candidate02/03 CleanAT4 zero-tolerance CUDA components passed; eight actual implementation ABBA samples and four actual Original36/72 ABBA samples retained. All paired performance assessments INCONCLUSIVE at unchanged thermal/load/clock guards; no reliable gain claim. Batch48/64/profile were NOT RUN before the dispatch deadline, never infer their speed. Explicit human fallback selected: unchanged OriginalBatch36, four fresh seed0 YAML200epoch arms. Current published runtime '+cpu['source_commit']+' has19 scientific files/60OriginalS4 byte-exact51ce, CPU reuse/source audit passed. Actual current-source s4_at_selected_native_20261030_gpu01 is running; finite local finish_fallback_preflight.py client follows it then baseline and phase via RRCTL strictly serial, no formal controller yet. Do not duplicate these IDs or repeat benchmarks. Then exactlyone scientific review/mainclosure/current gate and fresh_serial_queue.py with retained old locks.\n')
print(json.dumps(decision))
