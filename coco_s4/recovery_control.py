"""Exact-once repaired relaunch through the existing RRCTL serial controller."""
import argparse,json,os
from pathlib import Path
from .runtime import ROOT,load,now,sha,write_once,assert_formal_launch_authorized
from .phase_contract import OWNED,WORKFLOW,WORKFLOW_OUTPUT,ORIGIN_OUTPUT
from .dimension_protocol import PROTOCOL
from .dimension_release import EX,PYTHON,current_source,services,preflight,check_tag
from .release_transport import ensure_remote_exact,inspect_remote

SPEC=EX/(WORKFLOW+'.json')
CPU_TAG='coco_dim_20261026_2215hk_cpu16'
GPU_TAG='coco_dim_20261026_2230hk_gpu16'

def review_inputs(tag,review_path,cpu_tag):
    assert_formal_launch_authorized()
    tag=check_tag(tag);assert cpu_tag==CPU_TAG
    spec=load(EX/(tag+'.json'));path=EX/tag/'source_release.json';m=load(path)
    assert current_source()==spec['source']['commit']==m['source_commit']
    probe=load(EX/'p'/tag/'result/relaunch_binding_summary.json')
    assert probe['state']=='passed' and probe['source_commit']==m['source_commit'] and probe['release_sha256']==sha(path)
    assert probe['model_forwards']==0 and probe['formal_controller_started'] is False
    summary=EX/'p'/GPU_TAG/'result/smoke_summary.json';cpu_path=EX/'p'/CPU_TAG/'result/cpu_probe_summary.json'
    gpu=load(summary);cpu=load(cpu_path)
    from .fourier_gradient_audit import validate_production_batch16
    for s,device in ((cpu,'cpu'),(gpu,'cuda:0')):
        validate_production_batch16(s)
        assert s['state']=='passed' and s['source_commit']=='0113b9b65b0aba4b0132ff5b1f4f58782756cc53'
        assert s['actual_device']==device and s['actual_fourier_optimizer_updates']==1
        assert s['calibration_receipt_sha256']==m['calibration_dependency']['accepted_sha256']
    main=load(review_path)
    assert main['state']=='CLOSED_ALL_ACTUAL_VALIDATIONS' and main['reviewer_count_remains_one'] is True
    assert len(main['findings'])==9 and all(row['closure'].startswith('CLOSED') for row in main['findings'])
    for t in (CPU_TAG,GPU_TAG):
        audit=load(EX/('COCO_S4_REAL_BATCH16_'+t+'_INDEPENDENT_AUDIT.json'))
        assert audit['state']=='passed' and audit['source_files_verified']==145 and audit['native_files_verified']==7
    assert load(EX/'COCO_S4_EXACT_EXPORT_DEVICE_INDEPENDENT_AUDIT_20261026.json')['state']=='passed'
    review=dict(phase='fourier24',source_commit=m['source_commit'],release_sha256=sha(path),
        phase_config_sha256=sha(ROOT/m['phase_config_path']),smoke_sha256=sha(summary),
        calibration_receipt_sha256=m['calibration_dependency']['accepted_sha256'],reviewer_count=1,
        reviewer_independence=True,pre_run_result='pass',scientifically_correct=True,allow_run=True,
        scientific_verdict='allow_run',blockers=[],independent_review_original_decision='do_not_run',
        independent_review_original_commit='e89102474b064e1973e57964a19297f117a181a0',independent_review_original_verdict_sha256='7fd1c71f7034f0ba3f7693c5292b5aed1e292de64f2b734893293dcac6d583b6',main_agent_findings_closed=main['findings'],
        main_repair_closure_sha256=sha(review_path),independent_review_count_remains_one=True,
        validated_computation_binding=m['validated_computation_binding'])
    bound=EX/tag/'main_repair_review_bound.json'
    if bound.exists():assert load(bound)==review
    else:write_once(bound,review)
    return spec,path,m,summary,review,cpu_path,bound

def prepare(tag,review_path,cpu_tag):
    assert_formal_launch_authorized()
    if SPEC.exists() or (EX/'rrctl_state'/WORKFLOW).exists():raise FileExistsError('Prepared/bound run is inspect-only')
    spec,path,m,summary,review,cpu_path,bound=review_inputs(tag,review_path,cpu_tag)
    _,t=services();observation=preflight(t)
    assert observation['state']=='READY'
    assert observation['failed_dimension_predecessor']==m['failed_dimension_predecessor']
    assert observation['invalidated_predecessor']==m['invalidated_predecessor']
    output=WORKFLOW_OUTPUT.as_posix();receipts=output+'/receipts';binding=(OWNED/'recovery_bindings'/WORKFLOW).as_posix()
    old_gpu=load(EX/(GPU_TAG+'.json'))
    closure=dict(phase='fourier24',source_commit=m['source_commit'],release_sha256=sha(path),
        phase_config_sha256=sha(ROOT/m['phase_config_path']),
        smoke_path=old_gpu['remote']['output_root']+'/result/smoke_summary.json',smoke_sha256=sha(summary),
        review_path=receipts+'/fourier24_SCIENTIFIC_REVIEW.json',review_sha256=sha(bound),
        reviewer_count=1,reviewer_independence=True,pre_run_result='pass',scientific_verdict='allow_run',
        scientifically_correct=True,allow_run=True,blockers=[],
        teacher_checkpoint_sha256=m['teacher_dependency']['checkpoint_sha256'],
        calibration_receipt_sha256=m['calibration_dependency']['accepted_sha256'],
        independent_review_original_decision='do_not_run',independent_review_count_remains_one=True,
        main_agent_findings_closed=review['main_agent_findings_closed'],
        validated_computation_binding=m['validated_computation_binding'],closed_UTC=now())
    closure_path=EX/tag/'scientific_closure.json';write_once(closure_path,closure)
    old_cal=m['dimension_reconstruction_dependency']
    code="import hashlib,json,sys;from pathlib import Path;print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[1:]}))"
    paths=[(ORIGIN_OUTPUT/'pipeline_state.json').as_posix(),(ORIGIN_OUTPUT/'calibration/status.json').as_posix(),old_cal['phase_spec_path']]
    read=t.run([PYTHON,'-B','-c',code,*paths]);assert read.returncode==0,read.stderr.decode(errors='replace')
    hashes=json.loads(read.stdout)
    cfg=dict(schema='s4.coco.dimension-recovery-config.v1',run_id=WORKFLOW,source_commit=m['source_commit'],
        dimension_protocol=PROTOCOL,release_sha256=sha(path),origin_state_sha256=hashes[paths[0]],
        origin_calibration_status_sha256=hashes[paths[1]],failed_dimension_predecessor=m['failed_dimension_predecessor'],
        invalidated_predecessor=m['invalidated_predecessor'],component_receipt_path=old_cal['receipt_path'],
        component_receipt_sha256=old_cal['receipt_sha256'],component_phase_spec_path=old_cal['phase_spec_path'],
        component_phase_spec_sha256=hashes[paths[2]],scientific_closure_path=binding+'/scientific_closure.json',
        scientific_closure_sha256=sha(closure_path),
        CPU_full_loss_path=load(EX/(CPU_TAG+'.json'))['remote']['output_root']+'/result/cpu_probe_summary.json',
        CPU_full_loss_sha256=sha(cpu_path),
        authorization_sha256=sha(ROOT/'docs/S4_COCO_MINITRAIN_20261023/USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md'),
        human_relaunch_authorization_sha256=m['validated_computation_binding']['human_authorization_sha256'],
        no_second_controller=True,OP_enabled=False,retain_all=True)
    cfg_path=EX/WORKFLOW/'workflow.json';write_once(cfg_path,cfg)
    template=load(EX/'s4_coco_experiment_a_seed0_20261025_v3.json')
    remote={k:(OWNED/d/WORKFLOW).as_posix() for k,d in
        (('stage_root','stage'),('repo_root','repos'),('control_root','control'),('output_root','outputs'))}
    template.update(run_id=WORKFLOW,project='s4-coco-tfss-repaired-recovery')
    template['source']=dict(spec['source']);template['remote'].update(remote);template['session']['name']=WORKFLOW
    template['workload']=dict(cwd='.',argv=[PYTHON,'-B','-m','coco_s4.serial_recovery','--config',binding+'/workflow.json',
        '--release',binding+'/source_release.json','--output',remote['output_root']])
    template['anchors']=[dict(name=n,local_path=str(p),remote_path=binding+'/'+dest,sha256=sha(p)) for n,p,dest in
        [('recovery_release',path,'source_release.json'),('recovery_config',cfg_path,'workflow.json'),
         ('reviewed_closure',closure_path,'scientific_closure.json')]]
    template['artifacts']=[dict(path='pipeline_state.json',required=True),dict(path='RECOVERY_PROVENANCE.json',required=True)]
    template['local_pull_root']=str(EX/'p');template['metadata']=dict(execution_purpose='single_serial_tfss_repaired_recovery',
        old_workflow_not_restarted=True,candidate_commit=m['source_commit'],cleanup_policy='retain_all',no_second_controller=True)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template);write_once(SPEC,template)
    print(json.dumps(dict(prepared=str(SPEC),source_commit=m['source_commit'],launched=False)))

def launch():
    assert_formal_launch_authorized()
    if (EX/'rrctl_state'/WORKFLOW).exists() or (EX/WORKFLOW/'launch_intent.json').exists():
        raise FileExistsError('Existing run identity: inspect only')
    s=load(SPEC);assert current_source()==s['source']['commit']
    ctl,t=services();check=preflight(t);assert check['state']=='READY'
    paths=[s['remote'][k] for k in ('repo_root','output_root','stage_root','control_root')]
    r=t.run([PYTHON,'-B','-c','from pathlib import Path;import json,sys;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]))',json.dumps(paths)])
    assert r.returncode==0,'Remote collision: no launch'
    from remote_run_control.models import RunSpec
    ready=ctl.ready(RunSpec.from_path(SPEC));write_once(EX/WORKFLOW/'ready.json',ready)
    assert ready['ready'],ready
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/WORKFLOW/'launch_intent.json',dict(UTC=now(),source_commit=s['source']['commit']))
    value=ctl.launch(RunSpec.from_path(SPEC));write_once(EX/WORKFLOW/'launch.json',value);print(json.dumps(value))

def register_fourier(tag,review_path):
    assert_formal_launch_authorized();tag=check_tag(tag)
    spec=load(EX/(tag+'.json'));release=EX/tag/'source_release.json';m=load(release)
    assert current_source()==m['source_commit']==spec['source']['commit']
    ctl,t=services()
    code="import json,sys;from pathlib import Path;s=json.loads(Path(sys.argv[1]).read_text());assert s['run_id']==sys.argv[2] and s['state']=='WAITING_PHASE_RELEASE' and s['current_stage']=='fourier24' and s['active_job'] is None;print(json.dumps(s))"
    r=t.run([PYTHON,'-B','-c',code,(WORKFLOW_OUTPUT/'pipeline_state.json').as_posix(),WORKFLOW])
    assert r.returncode==0,r.stderr.decode(errors='replace')
    s=json.loads(r.stdout);assert s['stages']['calibration']['receipt_sha256']==m['calibration_dependency']['accepted_sha256']
    closure=EX/tag/'scientific_closure.json';bound=EX/tag/'main_repair_review_bound.json';c=load(closure)
    assert inspect_remote(t,[c['smoke_path']])[c['smoke_path']]==c['smoke_sha256']
    receipt_root=WORKFLOW_OUTPUT/'receipts';remote_source=spec['remote']['repo_root']
    ready=dict(phase='fourier24',workflow_run_id=WORKFLOW,pre_run_result='pass',source_root=remote_source,
        release_sha256=sha(release),release_relative='source_release.json',module='coco_s4.phase_fourier',
        previous_stage_receipt_sha256=m['calibration_dependency']['accepted_sha256'],
        scientific_closure_path=(receipt_root/'fourier24_SCIENTIFIC_CLOSURE.json').as_posix(),
        scientific_closure_sha256=sha(closure),smoke_path=c['smoke_path'],smoke_sha256=c['smoke_sha256'],registered_UTC=now())
    ready_path=EX/tag/'phase_registration.json';write_once(ready_path,ready)
    ensure_remote_exact(t,release,remote_source+'/source_release.json')
    ensure_remote_exact(t,bound,(receipt_root/'fourier24_SCIENTIFIC_REVIEW.json').as_posix())
    ensure_remote_exact(t,closure,ready['scientific_closure_path'])
    ensure_remote_exact(t,ready_path,(receipt_root/'fourier24_READY.json').as_posix())
    result=dict(state='READY_UPLOADED_AND_READBACK',ready_sha256=sha(ready_path),controller_owns_formal_start=True)
    write_once(EX/tag/'registration_verified.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch','register-fourier'))
    p.add_argument('--tag');p.add_argument('--review',type=Path);p.add_argument('--cpu-tag');a=p.parse_args()
    if a.action=='prepare':prepare(a.tag,a.review,a.cpu_tag)
    elif a.action=='launch':launch()
    else:register_fourier(a.tag,a.review)
