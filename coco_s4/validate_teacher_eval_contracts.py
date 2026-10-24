"""Adversarial dependency/closure checks; synthetic receipts, no GPU or metrics."""
import copy
from pathlib import Path
import json
from unittest.mock import patch

from .runtime import ROOT, now, sha, write_once
from . import phase_contract as pc
from .phase_teacher_eval import validate_config


def main():
    # Short synthetic paths avoid Windows MAX_PATH without changing production routes.
    out=ROOT.parent/('evc_'+now().replace(':','').replace('+','_'))
    out.mkdir(parents=True,exist_ok=False)
    workflow=out/'w'
    accepted=workflow/'teacher200/accepted.json'
    last=workflow/'teacher200/train/weights/last.pt'
    full=workflow/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
    last.parent.mkdir(parents=True);full.parent.mkdir(parents=True)
    last.write_bytes(b'synthetic EMA checkpoint identity only')
    full.write_bytes(b'synthetic full native state identity only')
    a=dict(state='ACCEPTED',source_commit=pc.TEACHER_COMMIT,release_sha256=pc.TEACHER_RELEASE,
           completed_epochs=200,finite_loss=True,zero_validation_forwards=True,
           last_checkpoint_sha256=sha(last),full_state_sha256=sha(full))
    write_once(accepted,a)
    binding=dict(accepted_path=str(accepted.resolve()),accepted_sha256=sha(accepted),
                 checkpoint_path=str(last.resolve()),checkpoint_sha256=sha(last),
                 full_state_path=str(full.resolve()),full_state_sha256=sha(full))
    assert pc.teacher_dependency(binding,workflow)[0]==last.resolve()
    rejected=[]
    def refuse(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError):rejected.append(name)
        else:raise AssertionError('Guard accepted invalid case: '+name)
    for field,value in [('checkpoint_path',str(last.with_name('best.pt'))),('accepted_sha256','0'*64),
                        ('checkpoint_sha256','0'*64),('full_state_sha256','0'*64)]:
        b=dict(binding);b[field]=value
        refuse(field,lambda b=b:pc.teacher_dependency(b,workflow))
    for field,value in [('completed_epochs',199),('source_commit','b1e29510e7b1ed5fa0fe361a27132d8790e65c14'),
                        ('zero_validation_forwards',False),('finite_loss',False),('state','RUNNING')]:
        changed=dict(a);changed[field]=value
        accepted.write_text(json.dumps(changed),encoding='utf-8')
        b=dict(binding,accepted_sha256=sha(accepted))
        refuse(field,lambda b=b:pc.teacher_dependency(b,workflow))
    accepted.write_text(json.dumps(a),encoding='utf-8');binding['accepted_sha256']=sha(accepted)
    source=out/'releases'/'eval_candidate';source.mkdir(parents=True)
    release=source/'release.json';write_once(release,{'synthetic':'release identity only'})
    config=ROOT/'coco_s4/configs/teacher_eval.json';cfg=json.loads(config.read_text())
    validate_config(cfg)
    for field,value in [('image_count',4999),('target_class_id',14),('parameter_updates',True),
                        ('checkpoint_selection_allowed',True),('method_statistics_allowed',True),
                        ('metric_driven_tuning_allowed',True)]:
        c=copy.deepcopy(cfg);c[field]=value
        refuse('config_'+field,lambda c=c:validate_config(c))
    manifest=dict(source_commit='1'*40,teacher_dependency=binding)
    smoke=out/'validation'/'smoke'/'smoke_summary.json'
    s=dict(state='passed',phase='teacher_eval',source_commit=manifest['source_commit'],
           release_sha256=sha(release),phase_config_sha256=sha(config),
           teacher_checkpoint_sha256=binding['checkpoint_sha256'],official_metrics_disabled=True,
           artifact_ingest_disabled=True,computation_kind='inference',finite_outputs=True,
           production_entrypoint_reached=True,actual_inference_batches=4,total_raw_forwards_including_warmup=6,
           baseline_equivalence_probe=dict(raw_outputs_exact=True,decoded_predictions_exact=True,
                                            absolute_difference=0,tolerance=0))
    write_once(smoke,s)
    closure=workflow/'receipts/teacher_eval_SCIENTIFIC_CLOSURE.json'
    c=dict(phase='teacher_eval',source_commit=manifest['source_commit'],release_sha256=sha(release),
           phase_config_sha256=sha(config),pre_run_result='pass',reviewer_count=1,
           teacher_checkpoint_sha256=binding['checkpoint_sha256'],smoke_path=str(smoke.resolve()),smoke_sha256=sha(smoke))
    write_once(closure,c)
    registration=workflow/'receipts/teacher_eval_READY.json'
    r=dict(phase='teacher_eval',workflow_run_id=pc.WORKFLOW,pre_run_result='pass',source_root=str(source),
           release_relative='release.json',module='coco_s4.phase_teacher_eval',
           release_sha256=sha(release),previous_stage_receipt_sha256=binding['accepted_sha256'],
           scientific_closure_path=str(closure.resolve()),scientific_closure_sha256=sha(closure))
    write_once(registration,r)
    with patch.multiple(pc,ROOT=source,OWNED=out,WORKFLOW_OUTPUT=workflow):
        pc.formal_phase_gate(registration,'teacher_eval',release,manifest,sha(config))
        for field,value in [('reviewer_count',0),('reviewer_count',2),('source_commit','2'*40),
                            ('release_sha256','0'*64),('teacher_checkpoint_sha256','0'*64),('pre_run_result','OPEN')]:
            cc=dict(c);cc[field]=value;closure.write_text(json.dumps(cc))
            rr=dict(r,scientific_closure_sha256=sha(closure));registration.write_text(json.dumps(rr))
            refuse('closure_'+field+'_'+str(value),lambda:pc.formal_phase_gate(registration,'teacher_eval',release,manifest,sha(config)))
        for field,value in [('finite_outputs',False),('official_metrics_disabled',False),
                            ('actual_inference_batches',0),('actual_inference_batches',101),
                            ('total_raw_forwards_including_warmup',101),
                            ('source_commit','2'*40)]:
            ss=dict(s);ss[field]=value;smoke.write_text(json.dumps(ss))
            cc=dict(c,smoke_sha256=sha(smoke));closure.write_text(json.dumps(cc))
            rr=dict(r,scientific_closure_sha256=sha(closure));registration.write_text(json.dumps(rr))
            refuse('smoke_'+field,lambda:pc.formal_phase_gate(registration,'teacher_eval',release,manifest,sha(config)))
        ss=copy.deepcopy(s);ss['baseline_equivalence_probe']['raw_outputs_exact']=False
        smoke.write_text(json.dumps(ss));cc=dict(c,smoke_sha256=sha(smoke));closure.write_text(json.dumps(cc))
        rr=dict(r,scientific_closure_sha256=sha(closure));registration.write_text(json.dumps(rr))
        refuse('baseline_raw_mismatch',lambda:pc.formal_phase_gate(registration,'teacher_eval',release,manifest,sha(config)))
    proof=dict(state='passed',UTC=now(),synthetic_dependency_and_closure_only=True,
               rejected_cases=rejected,count=len(rejected),valid_fixture_routes_passed=True,
               GPU_initialized=False,scientific_review_passed=False,formal_phase_allowed=False,
               source_sha256={n:sha(ROOT/n) for n in ['coco_s4/phase_contract.py','coco_s4/phase_teacher_eval.py',
                                                     'coco_s4/configs/teacher_eval.json']},output=str(out),deleted_paths=[])
    (ROOT/'execution/TEACHER_EVAL_CONTRACT_VALIDATION_20261025.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))


if __name__=='__main__':main()
