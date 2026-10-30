"""Bookkeeping only: derive current optimization records from real probe receipts."""
import json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];TRACK=ROOT/'execution/cuda_batch_optimization_20261030';DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
now=datetime.now(timezone.utc).isoformat();records=[]
for path in sorted((HERE/'probes').glob('*/summary.json')):
    value=json.loads(path.read_text());spec=json.loads((path.parent/'runspec.json').read_text())
    records.append(dict(label=path.parent.name,run_id=spec['run_id'],case=spec['metadata']['case'],summary=value,path=path.relative_to(ROOT).as_posix()))
eq_path=TRACK/'CUDA_EQUIVALENCE_RESULTS.json';eq=json.loads(eq_path.read_text())
eq.update(status='ACTUAL_CUDA_COMPONENT_VALIDATION_IN_PROGRESS',as_of_UTC=now,old_AT_terminated=True,scientific_review='NOT_STARTED')
cases={}
for r in records:
    if r['case']['kind']=='eq':
        v=r['summary'];key=r['case']['implementation']+'_'+r['case']['arm']+str(r['case']['epsilon'])
        cases[key]=dict(state=v['state'],run_id=r['run_id'],summary=r['path'],native_outer_batches=v['actual_native_outer_batches'],
            SGD_per_side=v['actual_native_SGD_per_side'],AMP_skips_per_side=v['actual_AMP_skips_per_side'],exact_step_and_state_hashes=v['exact_step_and_state_hashes'])
eq['actual_cases']=cases
for impl in ('02','03'):
    component=[v for k,v in cases.items() if k.startswith(impl+'_')]
    eq['candidate'+impl].update(cuda='COMPONENT_PASS' if component else 'NOT_COLLECTED',engineering_equivalent=None,
        completed_component_cases=len(component),full_four_arm_and_resume_complete=False)
eq['window']=dict(old_AT='ABORTED',current_GPU_use='SERIAL_FINITE_VALIDATION',natural_boundary_wait_required=False)
eq_path.write_text(json.dumps(eq,indent=2)+'\n',encoding='utf8')
bench_path=TRACK/'CUDA_BATCH_BENCHMARK_RESULTS.json';bench=json.loads(bench_path.read_text())
bench.update(status='PENDING_ACTUAL_MATCHED_THROUGHPUT',as_of_UTC=now)
measured=[]
for r in records:
    v=r['summary']
    if r['case']['kind'] in ('bench','tail','correctness','profile'):
        measured.append(dict(run_id=r['run_id'],case=r['case'],summary=r['path'],state=v['state'],
            batch=v.get('native_batch'),images_per_second=v.get('timing',{}).get('images_per_second'),
            seconds_per_batch=v.get('timing',{}).get('seconds_per_batch_mean'),peak_allocated_MiB=v.get('cuda_peak_allocated_MiB'),
            paired_assessment_required=True))
bench['actual_samples']=measured
bench_path.write_text(json.dumps(bench,indent=2)+'\n',encoding='utf8')
decision_path=TRACK/'AT_SPEEDUP_SELECTION_AND_RESTART_DECISION.json';decision=json.loads(decision_path.read_text())
decision.update(as_of_UTC=now,decision='OLD_AT_TERMINATED_TESTING_FRESH_FOUR_ARM_PROTOCOL',old_terminal='aborted',
    current_run_id=None,current_arm=None,current_completed_epochs=None,current_remaining_original_training_epochs=None,
    new_formal_started=False,old_checkpoint_resume_forbidden=True,net_remaining_time_saving=None,
    blockers=['Actual candidate/batch CUDA evidence and scientific closure still in progress'],
    thresholds=dict(preferred_stable_throughput_gain=.15,old_progress_net10_gate_applicable=False),
    fallback='Fresh epoch0 four-arm OriginalBatch36 if no reliable scientifically valid speedup')
decision_path.write_text(json.dumps(decision,indent=2)+'\n',encoding='utf8')
sp=DOC/'CURRENT_STATE.json';state=json.loads(sp.read_text())
state['speed_optimization'].update(state='OLD_AT_TERMINATED_REAL_CUDA_TESTING',decision=decision['decision'],old_checkpoint_resume_forbidden=True)
state['new_protocol_scientific_review']='NOT_STARTED'
sp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
report=DOC/'S4_AT_DIRECT_STOP_CUDA_OPTIMIZATION_AND_RESTART_REPORT.md';text=report.read_text();marker='\n## Actual probe receipts\n';base=text.split(marker)[0]
table='\n| Probe | Case | Actual state | Real evidence |\n|---|---|---|---|\n'
for r in records:table+=f"| {r['run_id']} | {r['case']} | {r['summary']['state']} | {r['path']} |\n"
report.write_text(base+marker+f'Updated {now}. These are bounded test components; formal four-arm training has not started.\n'+table,encoding='utf8')
print(json.dumps(dict(collected_probes=len(records),exact_CUDA_component_cases=list(cases),measured_benchmark_samples=len(measured),new_formal_started=False)))
