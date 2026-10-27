"""Immutable controller admission for separately gated downstream releases.

This is provenance, not reuse of the Fourier smoke as a downstream smoke.
Each downstream scientific phase still consumes its own current-source gate.
"""
from pathlib import Path
from .runtime import ROOT,load,sha

POLICY='coco80-ntgt-native-tal-then-gt-risk-v1'
PREVIOUS={'export25k':'fourier24','clean200':'export25k','s4_200':'clean200',
          'official_eval':'s4_200','final_report':'official_eval'}


def local_controller_binding():
    """Read the real successful transport, never the failed first transport."""
    from .phase_contract import WORKFLOW,OWNED,WORKFLOW_OUTPUT
    e=ROOT/'execution';current=load(e/'COCO_CURRENT_STATE.json')
    tag=current['successful_transport_run_id'];spec=load(e/(tag+'.json'))
    if (spec['run_id']!=tag or spec['metadata'].get('scientific_workflow_run_id')!=WORKFLOW
            or spec['remote']['output_root']!=WORKFLOW_OUTPUT.as_posix()):
        raise ValueError('Successful transport is not this scientific workflow')
    cfg_path=e/WORKFLOW/'workflow.json';cfg=load(cfg_path)
    manifest_path=Path(next(a['local_path'] for a in load(e/(WORKFLOW+'.json'))['anchors'] if a['name']=='recovery_release'))
    manifest=load(manifest_path)
    if (cfg['source_commit']!=spec['source']['commit'] or manifest['source_commit']!=cfg['source_commit']
            or cfg['release_sha256']!=sha(manifest_path) or cfg['run_id']!=WORKFLOW):
        raise ValueError('Local immutable controller binding drift')
    base=OWNED/'recovery_bindings'/WORKFLOW
    return dict(source_commit=cfg['source_commit'],controller_identity=current['active_controller'],
        config=dict(path=(base/'workflow.json').as_posix(),sha256=sha(cfg_path)),
        release=dict(path=(base/'source_release.json').as_posix(),sha256=sha(manifest_path)),
        formal_precheck_receipt=manifest['formal_precheck_receipt'],
        remote_repo_root=spec['remote']['repo_root'],transport_run_id=tag)


def attach(release):
    from .phase_contract import WORKFLOW
    if release.get('phase') not in PREVIOUS or release.get('workflow_run_id')!=WORKFLOW:
        raise ValueError('Controller admission is only for this downstream workflow')
    context=local_controller_binding()
    release.update(risk_estimator_policy=POLICY,strict_Original_S4_equivalence_claim=False,
        formal_precheck_receipt=context['formal_precheck_receipt'],workflow_controller_admission=context)
    scope=load(ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    for name in (scope['human_relaunch_authorization_path'],scope['actual_readiness_path']):
        release['files'][name]=sha(ROOT/name)
    return release


def validate(release):
    """Bind native16 admission to its real controller, not future phase commit.

The next phase's current commit is independently verified by verify_release
and its own production gate; no downstream scientific acceptance is inferred.
"""
    from .phase_contract import WORKFLOW,OWNED,WORKFLOW_OUTPUT
    from .serial import live
    phase=release.get('phase');context=release.get('workflow_controller_admission',{})
    if (phase not in PREVIOUS or release.get('workflow_run_id')!=WORKFLOW
            or release.get('risk_estimator_policy')!=POLICY
            or release.get('validation_only') is True
            or release.get('strict_Original_S4_equivalence_claim') is not False):
        raise ValueError('Downstream independent protocol identity is missing')
    base=(OWNED/'recovery_bindings'/WORKFLOW).resolve()
    values={}
    for key,name in (('release','source_release.json'),('config','workflow.json')):
        entry=context[key];path=Path(entry['path']).resolve()
        if path!=(base/name).resolve() or sha(path)!=entry['sha256']:
            raise ValueError('Immutable controller '+key+' drift')
        values[key]=load(path)
    controller=values['release'];cfg=values['config'];source=context['source_commit']
    if (controller.get('source_commit')!=source or cfg.get('source_commit')!=source
            or cfg.get('run_id')!=WORKFLOW or controller.get('workflow_run_id')!=WORKFLOW
            or controller.get('phase')!='fourier24' or controller.get('risk_estimator_policy')!=POLICY
            or cfg.get('risk_estimator_policy')!=POLICY or cfg.get('no_second_controller') is not True
            or cfg.get('release_sha256')!=context['release']['sha256']):
        raise ValueError('Controller source/config/protocol identity differs')
    admission=controller.get('formal_precheck_receipt')
    if admission!=context.get('formal_precheck_receipt') or admission!=release.get('formal_precheck_receipt'):
        raise ValueError('Downstream admission is not the actual controller precheck')
    state=load(WORKFLOW_OUTPUT/'pipeline_state.json')
    if (state.get('run_id')!=WORKFLOW or state.get('source_commit')!=source
            or state.get('config_sha256')!=context['config']['sha256']
            or state.get('release_sha256')!=context['release']['sha256']
            or state.get('risk_estimator_policy')!=POLICY
            or state.get('controller')!=context.get('controller_identity') or not live(state.get('controller'))):
        raise ValueError('Expected sole controller is not alive with this immutable binding')
    if (state.get('current_stage')!=phase or state.get('state') not in ('WAITING_PHASE_RELEASE','RUNNING')
            or (state.get('state')=='RUNNING' and state.get('active_job')!=phase)):
        raise ValueError('Downstream serial slot is not available to this phase')
    previous=PREVIOUS[phase];stage=state.get('stages',{}).get(previous,{})
    path=WORKFLOW_OUTPUT/previous/'accepted.json'
    if (stage.get('state')!='ACCEPTED' or sha(path)!=stage.get('receipt_sha256')
            or load(path).get('state')!='ACCEPTED'):
        raise ValueError('Immediate downstream predecessor is not actually accepted')
    return context
