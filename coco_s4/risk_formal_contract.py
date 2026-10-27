"""Independent risk-protocol admission/provenance; no scientific arithmetic."""
from pathlib import Path
from .runtime import ROOT,load,sha
from .phase_contract import OWNED,WORKFLOW,FAILED_REPAIRED_WORKFLOW
POLICY='coco80-ntgt-native-tal-then-gt-risk-v1'

CPU_TAG='coco_dim_20261027_risk_formal_cpu16b'
GPU_TAG='coco_dim_20261027_risk_formal_gpu16a'

def failed_repaired_predecessor(cfg,manifest):
    from .serial import live
    proof=cfg['failed_repaired_predecessor']
    if proof!=manifest.get('failed_repaired_predecessor') or proof['run_id']!=FAILED_REPAIRED_WORKFLOW:
        raise ValueError('Missing latest failed12step predecessor identity')
    root=OWNED/'outputs'/FAILED_REPAIRED_WORKFLOW
    expected={'pipeline_state.json','fourier24/status.json','jobs/fourier24/exit.json',
        'fourier24/native_s4/latest_continuation.pt','fourier24/native_s4/diagnostics.csv'}
    if set(proof['files'])!=expected:raise ValueError('Incomplete retained12step failure evidence')
    scope=load(ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    requirements=load(ROOT/scope['actual_readiness_path'])
    if proof['files']!=requirements['retained_failed12step_files']:
        raise ValueError('Latest predecessor differs from preserved failure snapshot')
    for name,digest in proof['files'].items():
        if sha(root/name)!=digest:raise ValueError('Retained failed12step evidence changed')
    state=load(root/'pipeline_state.json')
    if (state['state']!='FAILED' or state['source_commit']!=proof['source_commit']
            or live(state.get('controller')) or live(state.get('active_child'))
            or load(root/'jobs/fourier24/exit.json')['returncode']==0):
        raise ValueError('Old12step workflow is not a dead retained failure')
    return proof

def validate_adoption(closure,manifest):
    scope=load(ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    required=dict(risk_estimator_policy=POLICY,workflow_run_id=WORKFLOW,
        human_authorization_sha256=scope['human_relaunch_authorization_sha256'],
        strict_Original_S4_equivalence_claim=False,fresh_Fourier_initialization=True,
        old12step_resume_allowed=False,original_independent_review_count=1,
        risk_F01_closed=True,full25K_risk_audit='accepted')
    if manifest.get('risk_estimator_policy')!=POLICY or any(closure.get(k)!=v for k,v in required.items()):
        raise ValueError('Independent risk-protocol adoption is not scientifically closed')
    population=closure['population'];path=Path(population['path']).resolve()
    if not path.is_relative_to((OWNED/'validation').resolve()) or sha(path)!=population['sha256']:
        raise ValueError('Completed25K risk audit receipt drift')
    census=load(path)
    if (census.get('state')!='passed' or census.get('population_state')!='COMPLETE_25K'
            or census.get('risk_estimator_policy')!=POLICY or census.get('parameter_updates')!=0
            or census['counts']['images']!=25000 or census['counts']['NT_GT']!=126379
            or census['counts']['eligible_Person_images']!=13630):
        raise ValueError('Full25K independent risk validation was not accepted')
