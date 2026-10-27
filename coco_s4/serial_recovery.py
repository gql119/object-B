"""Unique dimension-protocol recovery using the existing serial execution code.

Accepted old Teacher/evaluation stages are referenced, not executed. The old
failed workflow remains immutable. No controller may overlap this one.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time

from .runtime import ROOT,atomic,commit,load,now,sha,safe_relative,verify_release,write_once
from .phase_contract import OWNED,WORKFLOW,WORKFLOW_OUTPUT,ORIGIN_OUTPUT,teacher_dependency,PREVIOUS_DIMENSION_OUTPUT,PREVIOUS_DIMENSION_WORKFLOW
from .dimension_protocol import PROTOCOL,validate_receipt_dimension
from .serial import ORDER,identity,live,capacity,run_job


def validate_closure(closure,manifest,release_sha,component_sha):
    from .risk_formal_contract import validate_adoption
    validate_adoption(closure,manifest)
    expected=dict(source_commit=manifest['source_commit'],release_sha256=release_sha,
        calibration_receipt_sha256=component_sha,pre_run_result='pass',reviewer_count=1,
        reviewer_independence=True,scientifically_correct=True,allow_run=True,blockers=[])
    if any(closure.get(k)!=v for k,v in expected.items()):
        raise ValueError('Recovery requires the actual current-source single-review main closure')
    smoke_path=Path(closure['smoke_path']).resolve()
    if not smoke_path.is_relative_to((OWNED/'validation').resolve()) or sha(smoke_path)!=closure['smoke_sha256']:
        raise ValueError('Recovery smoke binding is not the retained owned candidate')
    s=load(smoke_path)
    from .fourier_gradient_audit import validate_production_batch16
    validate_production_batch16(s)
    from .relaunch_binding import evidence_identity
    evidence=evidence_identity(manifest)
    required=dict(state='passed',source_commit=evidence['source_commit'],release_sha256=evidence['release_sha256'] or release_sha,
                  calibration_receipt_sha256=component_sha,computation_kind='fourier-gradient',
                  actual_device='cuda:0',actual_forward_tensor_devices=['cuda:0'],actual_cuda_device_indices=[0],
                  actual_fourier_optimizer_updates=1,non_target_class_count=79,independent_P3_selected_count=48,
                  finite_outputs=True,frozen_teacher_unchanged=True,official_metrics_disabled=True,
                  artifact_ingest_disabled=True,validation_used=False)
    if any(s.get(k)!=v for k,v in required.items()) or not 1<=s.get('production_forward_calls',0)<=100:
        raise ValueError('Recovery has no actual bounded current-candidate CUDA full-loss evidence')
    if any(not math.isfinite(s.get('SafeBG_loss_Fourier_gradient_norms',{}).get(k,0))
           or s.get('SafeBG_loss_Fourier_gradient_norms',{}).get(k,0)<=0 for k in ('L_BG','L_rel')):
        raise ValueError('Recovery SafeBG gradient gate is not closed')
    baseline=s.get('baseline_equivalence_probe',{})
    if (baseline.get('absolute_difference')!=0 or baseline.get('tolerance')!=0
            or baseline.get('all_outputs_and_diagnostics_exact') is not True
            or baseline.get('RNG_restored') is not True):
        raise ValueError('Actual frozen24D current-candidate equivalence is not closed')
    return s


def origin_preconditions(cfg,manifest):
    origin=load(ORIGIN_OUTPUT/'pipeline_state.json')
    if (origin['state']!='FAILED' or origin['current_stage']!='calibration'
            or sha(ORIGIN_OUTPUT/'pipeline_state.json')!=cfg['origin_state_sha256']
            or live(origin['controller']) or live(origin.get('active_child'))):
        raise ValueError('Old workflow is not the bound dead failed predecessor')
    old_phase=ORIGIN_OUTPUT/'calibration/status.json'
    if sha(old_phase)!=cfg['origin_calibration_status_sha256'] or load(old_phase)['state']!='BLOCKED_PROTOCOL_COMPATIBILITY':
        raise ValueError('Old calibration block record changed')
    teacher_dependency(manifest['teacher_dependency'])
    for name,binding in (('teacher200',manifest['teacher_dependency']),('teacher_eval',manifest['teacher_eval_dependency'])):
        receipt=ORIGIN_OUTPUT/name/'accepted.json'
        if (origin['stages'][name]['state']!='ACCEPTED' or sha(receipt)!=binding['accepted_sha256']
                or origin['stages'][name]['receipt_sha256']!=sha(receipt) or load(receipt)['state']!='ACCEPTED'):
            raise ValueError('Already accepted predecessor identity changed: '+name)
    return origin


def previous_dimension_preconditions(cfg,manifest):
    proof=cfg.get('failed_dimension_predecessor')
    if not proof or proof!=manifest.get('failed_dimension_predecessor') or proof.get('run_id')!=PREVIOUS_DIMENSION_WORKFLOW:
        raise ValueError('Missing immutable failed dimension-v1 predecessor binding')
    expected_files={'pipeline_state.json','fourier24/status.json','jobs/fourier24/exit.json'}
    if set(proof.get('files',{}))!=expected_files:
        raise ValueError('Incomplete failed dimension-v1 evidence')
    for name,digest in proof['files'].items():
        if sha(PREVIOUS_DIMENSION_OUTPUT/name)!=digest:
            raise ValueError('Failed dimension-v1 evidence changed: '+name)
    state=load(PREVIOUS_DIMENSION_OUTPUT/'pipeline_state.json')
    if (state.get('state')!='FAILED' or state.get('current_stage')!='fourier24'
            or state.get('source_commit')!=proof['source_commit']
            or live(state.get('controller')) or live(state.get('active_child'))
            or load(PREVIOUS_DIMENSION_OUTPUT/'fourier24/status.json').get('state')!='FAILED'
            or load(PREVIOUS_DIMENSION_OUTPUT/'jobs/fourier24/exit.json').get('returncode',0)==0):
        raise ValueError('Dimension-v1 is not a dead, retained failed predecessor')
    return proof


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--release',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();m=verify_release(a.release);cfg=load(a.config)
    if (cfg['run_id']!=WORKFLOW or cfg['source_commit']!=commit() or cfg['release_sha256']!=sha(a.release)
            or cfg['dimension_protocol']!=PROTOCOL or m.get('workflow_run_id')!=WORKFLOW):
        raise ValueError('Wrong unique versioned recovery/source binding')
    out=a.output.resolve()
    if out!=WORKFLOW_OUTPUT.resolve() or (out/'pipeline_state.json').exists():
        raise FileExistsError('Recovery output collision; old/new bound workflows are inspect-only')
    if os.getuid()!=1014:raise ValueError('Wrong recovery owner')
    lock=(OWNED/'single_dimension_recovery_controller.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    origin=origin_preconditions(cfg,m)
    dimension_history=previous_dimension_preconditions(cfg,m)
    from .relaunch_binding import invalidated_predecessor,evidence_identity
    invalid_history=invalidated_predecessor(cfg)
    from .risk_formal_contract import failed_repaired_predecessor
    repaired_history=failed_repaired_predecessor(cfg,m)
    evidence=evidence_identity(m)
    component_path=Path(cfg['component_receipt_path']).resolve()
    if (not component_path.is_relative_to((OWNED/'validation').resolve())
            or sha(component_path)!=cfg['component_receipt_sha256']):
        raise ValueError('Retained reconstruction component binding changed')
    component=load(component_path);validate_receipt_dimension(component)
    if component.get('acceptance_scope')!='CPU_RECONSTRUCTION_COMPONENT' or component.get('formal_use_requires_current_review') is not True:
        raise ValueError('Wrong new-protocol reconstruction receipt')
    closure_path=Path(cfg['scientific_closure_path']).resolve()
    if not closure_path.is_relative_to(OWNED.resolve()) or sha(closure_path)!=cfg['scientific_closure_sha256']:
        raise ValueError('Recovery scientific closure route/hash drift')
    closure=load(closure_path);smoke=validate_closure(closure,m,sha(a.release),sha(component_path))
    if smoke.get('selected_channel_count')!=validate_receipt_dimension(component):
        raise ValueError('The reviewed GPU consumer used a different selected coordinate space')
    cpu_path=Path(cfg['CPU_full_loss_path']).resolve()
    if not cpu_path.is_relative_to((OWNED/'validation').resolve()) or sha(cpu_path)!=cfg['CPU_full_loss_sha256']:
        raise ValueError('Learned-Teacher native CPU evidence path/hash drift')
    cpu=load(cpu_path)
    from .fourier_gradient_audit import validate_production_batch16
    validate_production_batch16(cpu)
    if any(cpu.get(k)!=v for k,v in dict(state='passed',source_commit=evidence['source_commit'],actual_device='cpu',
        actual_forward_tensor_devices=['cpu'],actual_fourier_optimizer_updates=1,finite_outputs=True,
        calibration_receipt_sha256=sha(component_path)).items()):
        raise ValueError('Current learned-Teacher native CPU loss chain did not pass')
    numeric=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_DIMENSION_CPU_NUMERICS_20261025.json')
    regression=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_DIMENSION_REGRESSION_20261025.json')
    if (numeric['state']!='passed' or regression['state']!='passed'
            or any(regression[k]!=v for k,v in dict(tests=55,failures=0,errors=0,skips=0).items())
            or any(row['output_max_absolute_difference']!=0 for row in numeric['VOC24_equivalence'])):
        raise ValueError('Frozen24D numerical/regression gate is not closed')
    for name in ('ue_framework/methods/safebg.py','ue_framework/methods/b2_semantic.py'):
        if numeric['sources'][name]!=m['files'][name]:raise ValueError('VOC24 evidence consumer source drift')
    precision=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_PRECISION_CANDIDATE_CPU_20261025.json')
    if (precision.get('state')!='passed' or precision.get('cache_source_sha256')!=m['files']['coco_s4/cache.py']
            or not all(row.get('VOC24_all_cache_outputs_and_hash_exact') is True for row in precision['VOC24'])):
        raise ValueError('Current cache precision candidate lacks exact legacy24 arithmetic evidence')
    if sha(ROOT/'docs/S4_COCO_MINITRAIN_20261023/USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md')!=cfg['authorization_sha256']:
        raise ValueError('Human dimension-only authorization drift')
    out.mkdir(parents=True,exist_ok=True)
    (out/'receipts').mkdir(exist_ok=False)
    for name in ('teacher200','teacher_eval'):
        # Existing owned checkpoints stay in place. Later consumers resolve
        # these links to the exact already accepted old fixed dependencies.
        (out/name).symlink_to(ORIGIN_OUTPUT/name,target_is_directory=True)
    cal=out/'calibration';cal.mkdir(exist_ok=False)
    shutil.copyfile(component_path,cal/'accepted.json')
    spec_path=Path(cfg['component_phase_spec_path']).resolve()
    if not spec_path.is_relative_to((OWNED/'validation').resolve()) or sha(spec_path)!=cfg['component_phase_spec_sha256']:
        raise ValueError('Reconstruction source/root specification drift')
    shutil.copyfile(spec_path,cal/'phase_spec.json')
    write_once(cal/'FORMAL_USE_AUTHORIZATION.json',dict(state='FORMAL_USE_GATES_CLOSED',
        component_receipt_sha256=sha(component_path),scientific_closure_sha256=sha(closure_path),
        protocol=PROTOCOL,old_failed_calibration_unchanged=True,UTC=now()))
    write_once(out/'RECOVERY_PROVENANCE.json',dict(origin_run_id=origin['run_id'],origin_state_sha256=cfg['origin_state_sha256'],
        failed_dimension_predecessor=dimension_history,invalidated_TFSS_predecessor=invalid_history,
        failed12step_predecessor=repaired_history,risk_estimator_policy=m['risk_estimator_policy'],
        strict_Original_S4_equivalence_claim=False,Fourier_restart='original initialization; no12step state loaded',
        validated_computation_binding=m.get('validated_computation_binding'),
        origin_calibration_status_sha256=cfg['origin_calibration_status_sha256'],reused_without_execution=['teacher200','teacher_eval'],
        reconstructed_from_retained25K_statistics=True,component_receipt_sha256=sha(component_path),
        scientific_closure_sha256=sha(closure_path),exact_worker_prefetch_resume_claim=False,UTC=now()))
    stages={n:dict(state='ACCEPTED',receipt_sha256=sha(out/n/'accepted.json'),reused=True)
            for n in ('teacher200','teacher_eval','calibration')}
    status=dict(schema='s4.coco.serial-recovery.v1',state='WAITING_PHASE_RELEASE',run_id=WORKFLOW,
        source_commit=commit(),config_sha256=sha(a.config),release_sha256=sha(a.release),controller=identity(),
        stages=stages,current_stage='fourier24',active_job=None,active_child=None,OP_enabled=False,
        phase_order=list(ORDER),dimension_protocol=PROTOCOL,created_UTC=now())
    status['risk_estimator_policy']=m['risk_estimator_policy']
    status['strict_Original_S4_equivalence_claim']=False
    atomic(out/'pipeline_state.json',status)
    try:
        # Same phase order, job supervisor, capacities and immutable READY
        # semantics as the existing serial controller. No accepted job reruns.
        for phase in ORDER[3:]:
            status.update(state='WAITING_PHASE_RELEASE',current_stage=phase,updated_UTC=now())
            atomic(out/'pipeline_state.json',status)
            registration=out/'receipts'/(phase+'_READY.json')
            while not registration.exists():time.sleep(60)
            req=load(registration)
            source=Path(req['source_root']).resolve()
            relative=safe_relative(req['release_relative'])
            previous=ORDER[ORDER.index(phase)-1]
            module={'fourier24':'phase_fourier','export25k':'phase_export','clean200':'phase_victim',
                    's4_200':'phase_victim','official_eval':'phase_official_eval','final_report':'phase_final_report'}[phase]
            if (req.get('phase')!=phase or req.get('workflow_run_id')!=WORKFLOW or req.get('pre_run_result')!='pass'
                    or not source.is_relative_to((OWNED/'releases').resolve())
                    or req['previous_stage_receipt_sha256']!=status['stages'][previous]['receipt_sha256']
                    or sha(source/relative)!=req['release_sha256']
                    or req['module']!='coco_s4.'+module):
                raise ValueError('Invalid next-phase immutable recovery registration')
            while True:
                resources=capacity(out);atomic(out/'capacity_latest.json',resources)
                if resources['admitted']:break
                time.sleep(300)
            argv=[sys.executable,'-B','-m',req['module'],'--release',str(source/req['release_relative']),
                  '--output',str(out/phase),'--registration',str(registration)]
            run_job(out,phase,argv,source,status)
            accepted=out/phase/'accepted.json'
            if load(accepted)['state']!='ACCEPTED':raise ValueError('Recovery phase did not accept: '+phase)
            status['stages'][phase]=dict(state='ACCEPTED',receipt_sha256=sha(accepted))
        status.update(state='COMPLETED_AND_ACCEPTED',current_stage=None,updated_UTC=now())
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),ended_UTC=now());raise
    finally:atomic(out/'pipeline_state.json',status)


if __name__=='__main__':main()
