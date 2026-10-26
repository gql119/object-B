"""CPU contracts and no-training dependency simulation; no learned-model readiness claim."""
import argparse,copy,json,ast
from pathlib import Path

def run(output):
    import torch
    from .runtime import ROOT,sha,write_once
    from .validate_canonical_s4_training_cpu import main as optimizer_component
    from .fourier_artifact import validate_native_params
    from .phase_victim import validate_phase_config,canonical_model_state_sha256
    from .evaluation import paired_metrics
    output=Path(output);optimizer_component(output/'native_component')
    pack=torch.load(output/'native_component/global_params.pt',weights_only=True)
    identity=validate_native_params(pack,num_classes=80,target_class_id=0,feature_layer='model.21')
    rejected=0
    for key,value in [('target_class_id',14),('support_mode','full'),('eps',16/255),
                      ('num_classes',20),('coords',[[1,1]]*16),('fourier_coeff',torch.ones(64,3)),
                      ('s4_identity',None)]:
        bad=copy.deepcopy(pack);bad[key]=value
        try:validate_native_params(bad,num_classes=80,target_class_id=0,feature_layer='model.21')
        except ValueError:rejected+=1
    assert rejected==7
    for name in ('clean200','s4_200'):
        validate_phase_config(json.loads((ROOT/f'coco_s4/configs/{name}.json').read_text()),name)
    state={'w':torch.tensor([.1,.2])}
    left=canonical_model_state_sha256(state);right=canonical_model_state_sha256(copy.deepcopy(state));assert left==right
    state['w'][0]+=1;assert left!=canonical_model_state_sha256(state)
    from .runtime import assert_formal_launch_authorized
    from .recovery_control import prepare,launch,register_fourier
    from .release_transport import close_and_register
    guards=[assert_formal_launch_authorized,lambda:prepare('unused',None,None),launch,
            lambda:register_fourier('unused',None),lambda:close_and_register(None,tag='unused',phase='fourier24',module='unused',spec=None,release_path=None,smoke_path=None,review_path=None,previous_stage_receipt_sha256=None)]
    for guard in guards:
        try:guard()
        except ValueError as e:assert 'EMERGENCY_REPAIR_ONLY' in str(e)
        else:raise AssertionError('Formal admission unexpectedly allowed')
    # Explicit synthetic graph. No official ACCEPTED/READY files are fabricated.
    stages=['fourier24','export25k','clean200','s4_200','official_eval','final_report'];graph=[]
    predecessor=sha(output/'native_component/global_params.pt')
    for stage in stages:
        row=dict(state='SIMULATION_ONLY',run_id='emergency_contract_fixture',phase=stage,predecessor_sha256=predecessor,
                 official_READY_written=False,official_ACCEPTED_written=False,training_invoked=False)
        path=output/(stage+'_simulation.json');write_once(path,row);predecessor=sha(path)
        graph.append(dict(phase=stage,path=str(path),sha256=predecessor))
    metrics=[dict(yolo_class_id=i,AP50=.5,AP75=.4,AP50_95=.3) for i in range(80)]
    damaged=copy.deepcopy(metrics);damaged[0]['AP50']=.1
    pair=paired_metrics(metrics,damaged);assert abs(pair['P']-.8)<1e-12 and pair['U']==1
    count=0
    for path in list((ROOT/'coco_s4').rglob('*.py'))+list((ROOT/'ue_framework').rglob('*.py')):
        compile(path.read_bytes(),str(path),'exec');count+=1
    assert not torch.cuda.is_initialized()
    result=dict(state='passed',scope='CPU contracts, original optimizer toy fixture and no-training graph only',
        compiled=count,native_pack_accepted=True,invalid_native_packs_rejected=rejected,
        formal_admission_guards_rejected=len(guards),matched_tensor_identity_exact=True,
        graph=graph,Teacher_not_metric_denominator=True,model_forwards=0,GPU_initialized=False,
        learned_Teacher_CPU_gate=False,learned_Teacher_GPU_gate=False,formal_launch=False)
    write_once(output/'report.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
