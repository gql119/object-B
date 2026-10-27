"""SIMULATION_ONLY positive/negative admission tests with retained fixtures.

Uses real passed CPU/CUDA summaries as inputs, no models, GPU or controller.
Only the fixture owned-root is patched; all production guard code executes.
Never emits a production READY or ACCEPTED receipt.
"""
import argparse,copy,json
from pathlib import Path
from unittest.mock import patch
from . import runtime,phase_contract

def run(cpu,gpu,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    scope=runtime.load(runtime.ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    CPU=runtime.load(cpu);GPU=runtime.load(gpu);assert CPU['source_commit']==GPU['source_commit']
    head=GPU['source_commit'];owned=out/'fixture_owned';folder=owned/'recovery_bindings'/phase_contract.WORKFLOW
    folder.mkdir(parents=True);cases=[]
    receipt=dict(state='ALL_FORMAL_PRECHECKS_PASSED',workflow_run_id=phase_contract.WORKFLOW,
        source_commit=head,risk_estimator_policy=scope['risk_estimator_policy'],
        human_authorization_sha256=scope['human_relaunch_authorization_sha256'],
        native_CPU16='passed',native_CUDA16='passed',read_only_main_audits='passed',consumer_simulation='passed',
        full25K_risk_audit='accepted',fresh_Fourier_initialization=True,old12step_resume_allowed=False,
        strict_Original_S4_equivalence_claim=False,CPU=dict(path=str(Path(cpu).resolve()),sha256=runtime.sha(cpu)),
        GPU=dict(path=str(Path(gpu).resolve()),sha256=runtime.sha(gpu)))
    def case(name,value=None,manifest_change=None,expected=True):
        p=folder/(name+'.json');runtime.write_once(p,receipt if value is None else value)
        m=dict(source_commit=head,formal_precheck_receipt=dict(path=str(p.resolve()),sha256=runtime.sha(p)))
        if manifest_change:manifest_change(m)
        with patch.object(phase_contract,'OWNED',owned):
            try:runtime.assert_formal_launch_authorized(m);passed=True
            except (ValueError,FileNotFoundError):passed=False
        assert passed==expected,name
        cases.append(dict(name=name,expected=expected,observed=passed))
    runtime.assert_formal_launch_authorized()
    case('valid_bound_actual_native_inputs')
    case('wrong_hash',manifest_change=lambda m:m['formal_precheck_receipt'].update(sha256='0'*64),expected=False)
    case('wrong_source',manifest_change=lambda m:m.update(source_commit='0'*40),expected=False)
    for field,value in [('native_CPU16','OPEN'),('native_CUDA16','OPEN'),('old12step_resume_allowed',True),
                        ('strict_Original_S4_equivalence_claim',True),('risk_estimator_policy','wrong'),('consumer_simulation','OPEN')]:
        bad=copy.deepcopy(receipt);bad[field]=value;case('bad_'+field,bad,expected=False)
    bad=copy.deepcopy(receipt);bad['GPU']['sha256']='f'*64;case('bad_GPU_evidence_hash',bad,expected=False)
    assert 'torch' not in __import__('sys').modules
    result=dict(state='passed',SIMULATION_ONLY=True,source_commit=head,cases=cases,model_forwards=0,
        GPU_initialized=False,formal_controller_started=False,production_READY_written=False,production_ACCEPTED_written=False)
    runtime.write_once(out/'consumer_binding_summary.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=Path,required=True);p.add_argument('--gpu',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.cpu,a.gpu,a.output)))
