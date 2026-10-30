"""Refresh status from retained receipts, preserving historical observations."""
import json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
p=DOC/'CURRENT_STATE.json';s=json.loads(p.read_text());s['UTC']=datetime.now(timezone.utc).isoformat()
s['last_monitor_health']='OLD_ABANDONED_ABORTED_BOUNDED_CUDA_TESTING'
s['prior_accepted_runtime']=dict(source='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb',
    historical_CUDA_preflight='NATIVE_PHASE_RESUME_BASELINE_PASSED',
    historical_review='AT-SCI-001_NEGATIVE_PRESERVED_MAIN_CLOSURE_PASSED',old_formal_terminal='aborted')
s.update(AT_server_controller_started=False,active_formal_controllers=0,new_formal_controller_started=False,
    candidate_runtime_commit=None,CUDA_preflight='SELECTED_CURRENT_RUNTIME_NOT_YET_FROZEN',
    scientific_review='NEW_SELECTED_PROTOCOL_REVIEW_NOT_STARTED',formal_preflight='NEW_FORMAL_NOT_READY',
    existing_COCO_state='COMPLETED_AND_ACCEPTED_PUBLISHED',
    pending=['Finish bounded actual CUDA batch comparisons without posthoc tolerance changes',
        'Freeze selected or OriginalBatch36 fallback; current-source native GPU, one scientific review/main closure',
        'Launch one fresh epoch0 four-arm serial controller; never resume abandoned checkpoint'])
s['historical_live_observations_are_not_current']=True
s['early_pause']['superseded_by']='USER_DIRECT_TERMINATION_AND_RESTART_20261030.md'
s['cuda_batch_optimization'].update(GPU_benchmark_executed=True,actual_GPU_validation_started=True,
    candidate_CUDA_equivalence='CLEAN_AT4_COMPONENTS_02_03_PASS_ZERO_TOLERANCE',
    window_usable_now=True,decision='PENDING_FAIR_ABBA_AND_BATCH_TESTS')
proof=HERE/'s4_at_fresh_serial_control_20261030_cpu01/MAIN_CLOSURE.json'
if proof.is_file():
    v=json.loads(proof.read_text());assert v['state']=='PASS_REAL_QUEUE_CONTROL_ONLY'
    s['fresh_serial_queue_CPU']=dict(state=v['state'],source=v['source_commit'],proof=proof.relative_to(ROOT).as_posix(),
        formal_ready=False,formal_launched=False,old_lock_retained=True)
s.setdefault('failed_direct_CUDA_probes',{})['bo1']=dict(state='INVALID_EXTERNAL_GPU_COMPETITION',
    run_id='s4_at_b36_original_r1_20261030_direct01',
    evidence='execution/direct_stop_20261030/probes/bo1/TELEMETRY_FAILURE_DIAGNOSIS.json',speedup_claim=None)
p.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
contract=ROOT/'execution/cuda_batch_optimization_20261030/PENDING_BATCH_CUDA_CONTRACT.json'
c=json.loads(contract.read_text());c.update(state='ACTUAL_CUDA_IN_PROGRESS_AFTER_AUTHORIZED_OLD_TERMINATION',
    latest_authorization_sha256='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8',
    old_natural_boundary_wait_superseded=True,old_pause_resume_gate_superseded=True,
    old_progress_net10_gate_applicable=False,preferred_gain=.15)
contract.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(state=s['state'],actual_CUDA_started=True,fresh_queue_CPU=s.get('fresh_serial_queue_CPU'))))
