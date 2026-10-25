"""Adversarial recovery admission contracts; fixtures are not scientific evidence."""
import argparse
import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from .runtime import now,sha,write_once
from .runtime import ROOT
from .serial_recovery import validate_closure


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    smoke_path=output/'validation/fixture/smoke.json'
    smoke_path.parent.mkdir(parents=True)
    source='a'*40;release='b'*64;component='c'*64
    s=dict(state='passed',source_commit=source,release_sha256=release,calibration_receipt_sha256=component,
           computation_kind='fourier-gradient',actual_device='cuda:0',actual_forward_tensor_devices=['cuda:0'],
           actual_cuda_device_indices=[0],actual_fourier_optimizer_updates=1,non_target_class_count=79,
           independent_P3_selected_count=48,finite_outputs=True,frozen_teacher_unchanged=True,
           official_metrics_disabled=True,artifact_ingest_disabled=True,validation_used=False,
           production_forward_calls=8,SafeBG_loss_Fourier_gradient_norms=dict(L_BG=1.,L_rel=1.),
           per_loss_fourier_gradient_measurement=dict(measurement='direct_autograd_fourier_coeff_v1',
               target='trainer.fourier_coeff',parameter_shape=[64,3],parameter_device='cuda:0',EOT_draws=2,
               optimizer_gradient_accumulation_unchanged=True),
           baseline_equivalence_probe=dict(absolute_difference=0,tolerance=0,
                                          all_outputs_and_diagnostics_exact=True,RNG_restored=True))
    smoke_path.write_text(json.dumps(s))
    c=dict(source_commit=source,release_sha256=release,calibration_receipt_sha256=component,
           pre_run_result='pass',reviewer_count=1,reviewer_independence=True,scientifically_correct=True,
           allow_run=True,blockers=[],smoke_path=str(smoke_path.resolve()),smoke_sha256=sha(smoke_path))
    rejected=[]
    with patch('coco_s4.serial_recovery.OWNED',output):
        validate_closure(c,dict(source_commit=source),release,component)
        for field,value in (('source_commit','d'*40),('calibration_receipt_sha256','d'*64),
                            ('reviewer_count',2),('allow_run',False),('blockers',['block']),('pre_run_result','OPEN')):
            invalid=dict(c);invalid[field]=value
            try:validate_closure(invalid,dict(source_commit=source),release,component)
            except ValueError:rejected.append('closure.'+field)
            else:raise AssertionError('Invalid recovery closure accepted: '+field)
        for field,value in (('actual_forward_tensor_devices',['cpu']),('actual_cuda_device_indices',[]),
                            ('production_forward_calls',101),('non_target_class_count',78),
                            ('independent_P3_selected_count',39),('actual_fourier_optimizer_updates',0),
                            ('SafeBG_loss_Fourier_gradient_norms',dict(L_BG=float('nan'),L_rel=1.)),
                            ('SafeBG_loss_Fourier_gradient_norms',dict(L_BG=1.,L_rel=0.)),
                            ('baseline_equivalence_probe',dict(absolute_difference=1e-12,tolerance=1e-6))):
            invalid=copy.deepcopy(s);invalid[field]=value;smoke_path.write_text(json.dumps(invalid))
            changed=dict(c,smoke_sha256=sha(smoke_path))
            try:validate_closure(changed,dict(source_commit=source),release,component)
            except ValueError:rejected.append('smoke.'+field)
            else:raise AssertionError('Invalid actual-runtime admission accepted: '+field)
    report=dict(state='passed',UTC=now(),adversarial_recovery_contracts_rejected=len(rejected),
                test_fixture_only=True,actual_GPU_evidence=False,model_forwards=0,
                no_remote_write=True,no_controller_created=True,retained_temporary_artifacts=True)
    source=output/'releases/original/safebg.py';source.parent.mkdir(parents=True)
    source.write_bytes(subprocess.check_output(['git','-C',str(ROOT),'show',
                      '3a8db4c4277d2ffbcadef7864240063d84fff6b9:ue_framework/methods/safebg.py']))
    import torch
    from .dimension_equivalence import probe
    torch.set_num_threads(1)
    with patch('coco_s4.dimension_equivalence.OWNED',output):
        numeric=probe(dict(path=str(source),sha256=sha(source)),torch.device('cpu'),'weightless test fixture; no checkpoint used')
    assert numeric['absolute_difference']==0 and numeric['RNG_restored']
    report['new_baseline_helper_actual_CPU_exact_and_RNG_restored']=True
    write_once(output/'report.json',report);print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
