"""No-model simulation of the downstream CPU-evidence consumer; no formal receipt."""
import copy
import json
from unittest.mock import patch
from .runtime import ROOT,sha,write_once,now
from .phase_export import cpu_export_dependency


def main():
    out=ROOT/'execution'/('export_cpu_guard_'+now().replace(':','').replace('+','_'))
    path=out/'validation/cpu/result/cpu_probe_summary.json'
    source='a'*40
    manifest=dict(source_commit=source,phase_config_path='coco_s4/configs/export25k.json',
        fourier_dependency=dict(accepted_sha256='b'*64,params_sha256='c'*64),
        teacher_dependency=dict(checkpoint_sha256='d'*64))
    good=dict(state='passed',phase='export25k',source_commit=source,actual_device='cpu',cpu_probe=True,
        actual_training_images=1,Fourier_receipt_sha256='b'*64,Fourier_params_sha256='c'*64,
        teacher_checkpoint_sha256='d'*64,phase_config_sha256=sha(ROOT/manifest['phase_config_path']),
        production_entrypoint_reached=True,finite_outputs=True,decoded_budget_passed=True,
        labels_byte_identical=True,parameter_updates=False,validation_used=False,
        official_metrics_disabled=True,artifact_ingest_disabled=True,CUDA_initialized=False,
        production_forward_calls=1)
    write_once(path,good)
    manifest['cpu_export_validation']=dict(path=str(path),sha256=sha(path))
    rejected=[]
    with patch('coco_s4.phase_contract.OWNED',out):
        assert cpu_export_dependency(manifest)==good
        for key,value in (('source_commit','e'*40),('Fourier_params_sha256','f'*64),
                          ('CUDA_initialized',True),('decoded_budget_passed',False),
                          ('production_forward_calls',0),('parameter_updates',True)):
            bad=copy.deepcopy(good);bad[key]=value
            other=path.parent/(key+'.json');write_once(other,bad)
            current=copy.deepcopy(manifest)
            current['cpu_export_validation']=dict(path=str(other),sha256=sha(other))
            try:cpu_export_dependency(current)
            except ValueError:rejected.append(key)
            else:raise AssertionError('Invalid real-CPU binding accepted: '+key)
        missing=copy.deepcopy(manifest);missing.pop('cpu_export_validation')
        try:cpu_export_dependency(missing)
        except ValueError:rejected.append('missing_CPU')
        else:raise AssertionError('Missing CPU evidence accepted')
    report=dict(state='passed',scope='SIMULATION_ONLY CPU consumer guards',rejected=rejected,
        positive=1,negative=len(rejected),GPU_initialized=False,model_forwards=0,
        production_READY_written=False,fixture=str(out))
    write_once(out/'report.json',report);print(json.dumps(report))


if __name__=='__main__':main()
