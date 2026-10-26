"""Exact-once, reviewed selective recovery through the existing RRCTL mechanism."""
import argparse
import json
import math
import os
from pathlib import Path

from .runtime import ROOT,load,now,sha,write_once,atomic
from .phase_contract import OWNED,WORKFLOW,WORKFLOW_OUTPUT,ORIGIN_OUTPUT
from .dimension_protocol import PROTOCOL
from .dimension_release import EX,PYTHON,current_source,services,preflight,check_tag
from .release_transport import close_and_register

SPEC=EX/(WORKFLOW+'.json')


def component_report(name):
    report=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023'/name)
    if report.get('state')!='passed':raise ValueError('Required component proof did not pass: '+name)
    return report


def review_inputs(tag,review_path,cpu_tag):
    tag=check_tag(tag);cpu_tag=check_tag(cpu_tag)
    spec=load(EX/(tag+'.json'));manifest_path=EX/tag/'source_release.json';manifest=load(manifest_path)
    if current_source()!=spec['source']['commit']:raise ValueError('Recovery review is not the exact current published source')
    summary=Path(spec['local_pull_root'])/tag/'result/smoke_summary.json'
    smoke=load(summary);review=load(review_path)
    from .fourier_gradient_audit import validate_production_batch16
    validate_production_batch16(smoke)
    required=dict(phase='fourier24',source_commit=spec['source']['commit'],release_sha256=sha(manifest_path),
                  phase_config_sha256=sha(ROOT/manifest['phase_config_path']),smoke_sha256=sha(summary),
                  calibration_receipt_sha256=manifest['calibration_dependency']['accepted_sha256'],
                  reviewer_count=1,reviewer_independence=True,pre_run_result='pass',scientifically_correct=True,
                  allow_run=True,scientific_verdict='allow_run',blockers=[])
    if any(review.get(k)!=v for k,v in required.items()):
        raise ValueError('Recovery requires exactly one current-candidate scientific review and main closure')
    expected=dict(state='passed',source_commit=spec['source']['commit'],release_sha256=sha(manifest_path),
        computation_kind='fourier-gradient',actual_device='cuda:0',actual_forward_tensor_devices=['cuda:0'],
        actual_cuda_device_indices=[0],actual_fourier_optimizer_updates=1,finite_outputs=True,
        frozen_teacher_unchanged=True,official_metrics_disabled=True,artifact_ingest_disabled=True)
    if any(smoke.get(k)!=v for k,v in expected.items()):raise ValueError('Actual current CUDA full-loss smoke did not pass')
    if not 1<=smoke.get('production_forward_calls',0)<=100:raise ValueError('Invalid bounded GPU smoke budget')
    if any(not math.isfinite(smoke.get('SafeBG_loss_Fourier_gradient_norms',{}).get(k,0))
           or smoke.get('SafeBG_loss_Fourier_gradient_norms',{}).get(k,0)<=0 for k in ('L_BG','L_rel')):
        raise ValueError('Actual CUDA Fourier gradients are missing')
    cpu_spec=load(EX/(cpu_tag+'.json'))
    cpu_path=Path(cpu_spec['local_pull_root'])/cpu_tag/'result/cpu_probe_summary.json'
    cpu=load(cpu_path)
    validate_production_batch16(cpu)
    if (cpu.get('state')!='passed' or cpu.get('source_commit')!=spec['source']['commit']
            or cpu.get('actual_device')!='cpu' or cpu.get('actual_fourier_optimizer_updates')!=1
            or cpu.get('calibration_receipt_sha256')!=smoke.get('calibration_receipt_sha256')):
        raise ValueError('Current learned-Teacher full-loss CPU gate is not closed')
    numeric=component_report('COCO_SAFEBG_DIMENSION_CPU_NUMERICS_20261025.json')
    regression=component_report('COCO_SAFEBG_DIMENSION_REGRESSION_20261025.json')
    if (not all(row['output_max_absolute_difference']==0 and row['all_diagnostics_exact'] for row in numeric['VOC24_equivalence'])
            or any(regression.get(k)!=v for k,v in dict(tests=55,failures=0,errors=0,skips=0).items())):
        raise ValueError('VOC equivalence/regression evidence is not a pass')
    for name in ('ue_framework/methods/safebg.py','ue_framework/methods/b2_semantic.py'):
        if numeric['sources'][name]!=manifest['files'][name]:raise ValueError('VOC numerical proof does not cover this SafeBG consumer')
    precision=component_report('COCO_SAFEBG_PRECISION_CANDIDATE_CPU_20261025.json')
    if precision['cache_source_sha256']!=manifest['files']['coco_s4/cache.py']:
        raise ValueError('Cache arithmetic evidence does not cover the exact current implementation')
    return spec,manifest_path,manifest,summary,review,cpu_path


def prepare(tag,review_path,cpu_tag):
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    if SPEC.exists() or (EX/'rrctl_state'/WORKFLOW).exists():raise FileExistsError('Prepared/bound recovery identity is inspect-only')
    spec,release_path,m,summary,review,cpu_path=review_inputs(tag,review_path,cpu_tag)
    _,t=services();observation=preflight(t)
    if observation['state']!='READY':print(json.dumps(observation));return
    if observation['failed_dimension_predecessor']!=m['failed_dimension_predecessor']:
        raise ValueError('Failed predecessor evidence changed since isolated validation')
    # Construct exactly the closure that close_and_register will later install
    # beneath the new controller. Its input anchor is outside all lifecycle roots.
    out=(OWNED/'outputs'/WORKFLOW).as_posix();receipt_root=out+'/receipts'
    closure=dict(phase='fourier24',source_commit=m['source_commit'],release_sha256=sha(release_path),
        phase_config_sha256=sha(ROOT/m['phase_config_path']),smoke_path=spec['remote']['output_root']+'/result/smoke_summary.json',
        smoke_sha256=sha(summary),review_path=receipt_root+'/fourier24_SCIENTIFIC_REVIEW.json',
        review_sha256=sha(review_path),reviewer_count=1,reviewer_independence=True,pre_run_result='pass',
        scientific_verdict='allow_run',scientifically_correct=True,allow_run=True,blockers=[],
        teacher_checkpoint_sha256=m['teacher_dependency']['checkpoint_sha256'],
        calibration_receipt_sha256=m['calibration_dependency']['accepted_sha256'],closed_UTC=now(),
        independent_review_original_decision=review.get('independent_review_original_decision'),
        independent_review_original_commit=review.get('independent_review_original_commit'),
        main_agent_findings_closed=review.get('main_agent_findings_closed'),
        independent_review_count_remains_one=True)
    closure_local=EX/tag/'scientific_closure.json';write_once(closure_local,closure)
    binding=(OWNED/'recovery_bindings'/WORKFLOW).as_posix()
    template=load(EX/'s4_coco_experiment_a_seed0_20261025_v3.json')
    remote={k:(OWNED/d/WORKFLOW).as_posix() for k,d in
            (('stage_root','stage'),('repo_root','repos'),('control_root','control'),('output_root','outputs'))}
    probe=m['dimension_reconstruction_dependency'];rebuild_status=probe['phase_spec_path']
    read=t.run([PYTHON,'-B','-c','import hashlib,json,sys;from pathlib import Path;p=Path(sys.argv[1]);q=Path(sys.argv[2]);print(json.dumps(dict(origin=hashlib.sha256(p.read_bytes()).hexdigest(),phase=hashlib.sha256(q.read_bytes()).hexdigest())))',
                (ORIGIN_OUTPUT/'pipeline_state.json').as_posix(),(ORIGIN_OUTPUT/'calibration/status.json').as_posix()])
    if read.returncode:raise RuntimeError(read.stderr.decode(errors='replace')[-2000:])
    origin=json.loads(read.stdout)
    read=t.run([PYTHON,'-B','-c','import hashlib,sys;from pathlib import Path;print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())',rebuild_status])
    if read.returncode:raise RuntimeError(read.stderr.decode(errors='replace')[-2000:])
    cfg=dict(schema='s4.coco.dimension-recovery-config.v1',run_id=WORKFLOW,source_commit=m['source_commit'],
        dimension_protocol=PROTOCOL,release_sha256=sha(release_path),origin_state_sha256=origin['origin'],
        failed_dimension_predecessor=m['failed_dimension_predecessor'],
        origin_calibration_status_sha256=origin['phase'],component_receipt_path=probe['receipt_path'],
        component_receipt_sha256=probe['receipt_sha256'],component_phase_spec_path=rebuild_status,
        component_phase_spec_sha256=read.stdout.decode().strip(),
        scientific_closure_path=binding+'/scientific_closure.json',scientific_closure_sha256=sha(closure_local),
        CPU_full_loss_path=load(EX/(cpu_tag+'.json'))['remote']['output_root']+'/result/cpu_probe_summary.json',
        CPU_full_loss_sha256=sha(cpu_path),authorization_sha256=sha(ROOT/'docs/S4_COCO_MINITRAIN_20261023/USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md'),
        no_second_controller=True,OP_enabled=False,retain_all=True)
    cfg_local=EX/WORKFLOW/'workflow.json';write_once(cfg_local,cfg)
    template.update(run_id=WORKFLOW,project='s4-coco-dimension-compatible-recovery')
    template['source']=dict(spec['source']);template['remote'].update(remote);template['session']['name']=WORKFLOW
    template['workload']=dict(cwd='.',argv=[PYTHON,'-B','-m','coco_s4.serial_recovery','--config',binding+'/workflow.json',
        '--release',binding+'/source_release.json','--output',remote['output_root']])
    template['anchors']=[dict(name=n,local_path=str(p),remote_path=binding+'/'+dest,sha256=sha(p)) for n,p,dest in
        [('recovery_release',release_path,'source_release.json'),('recovery_config',cfg_local,'workflow.json'),
         ('reviewed_closure',closure_local,'scientific_closure.json')]]
    template['artifacts']=[dict(path='pipeline_state.json',required=True),dict(path='RECOVERY_PROVENANCE.json',required=True)]
    template['local_pull_root']=str(EX/'p')
    template['metadata']=dict(execution_purpose='single_serial_dimension_compatible_recovery',
        old_workflow_not_restarted=True,reused_accepted_stages=['teacher200','teacher_eval'],
        candidate_commit=m['source_commit'],cleanup_policy='retain_all',no_second_controller=True)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template);write_once(SPEC,template)
    print(json.dumps(dict(prepared=str(SPEC),source_commit=m['source_commit'],launched=False)))


def launch():
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    if (EX/'rrctl_state'/WORKFLOW).exists() or (EX/WORKFLOW/'launch_intent.json').exists():
        raise FileExistsError('Existing unique recovery binding; no restart')
    s=load(SPEC)
    if current_source()!=s['source']['commit']:raise ValueError('Recovery source changed after preparation')
    ctl,t=services();observation=preflight(t)
    if observation['state']!='READY':print(json.dumps(observation));return
    paths=[s['remote'][k] for k in ('repo_root','output_root','stage_root','control_root')]
    r=t.run([PYTHON,'-B','-c','from pathlib import Path;import json,sys;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]))',json.dumps(paths)])
    if r.returncode:raise ValueError('Remote recovery binding collision')
    from remote_run_control.models import RunSpec
    spec=RunSpec.from_path(SPEC);ready=ctl.ready(spec);write_once(EX/WORKFLOW/'ready.json',ready)
    if not ready.get('ready'):raise ValueError('RRCTL recovery thin readiness failed')
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/WORKFLOW/'launch_intent.json',dict(UTC=now(),source_commit=s['source']['commit']))
    result=ctl.launch(spec);write_once(EX/WORKFLOW/'launch.json',result);print(json.dumps(result))


def register_fourier(tag,review_path):
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    tag=check_tag(tag);spec=load(EX/(tag+'.json'));m_path=EX/tag/'source_release.json';m=load(m_path)
    if current_source()!=spec['source']['commit']:raise ValueError('Fourier registration source drift')
    summary=Path(spec['local_pull_root'])/tag/'result/smoke_summary.json'
    from .fourier_release import read_dependency
    _,t=services();observation=read_dependency(t)
    if (observation.get('state')!='READY_FOR_ISOLATED_PHASE_VALIDATION'
            or observation['calibration_dependency']['accepted_sha256']!=m['calibration_dependency']['accepted_sha256']):
        raise ValueError('New serial recovery is not waiting at the accepted Fourier slot')
    result=close_and_register(t,tag=tag,phase='fourier24',module='coco_s4.phase_fourier',
        spec=spec,release_path=m_path,smoke_path=summary,review_path=review_path,
        previous_stage_receipt_sha256=m['calibration_dependency']['accepted_sha256'],
        closure_extra=dict(calibration_receipt_sha256=m['calibration_dependency']['accepted_sha256']))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch','register-fourier'))
    p.add_argument('--tag');p.add_argument('--review',type=Path);p.add_argument('--cpu-tag')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.tag,a.review,a.cpu_tag)
    elif a.action=='register-fourier':register_fourier(a.tag,a.review)
    else:launch()
