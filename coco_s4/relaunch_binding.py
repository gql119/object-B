"""Explicit reuse of actual validated computation after routing/authorization-only edits.

The retained smoke keeps its real commit and release identity. This module never
relabels an old smoke as a new run or permits an unlisted scientific change.
"""
import ast
import json
from pathlib import Path
from .runtime import ROOT,load,sha,commit

VALIDATED_COMMIT='0113b9b65b0aba4b0132ff5b1f4f58782756cc53'
INVALID_WORKFLOW='s4_coco_experiment_a_seed0_20261026_dimcompat_v2'
AUTHORIZATION='docs/S4_COCO_MINITRAIN_20261023/USER_SHARED_GPU_AND_AUTOSTART_20261026.md'
PLUMBING={'coco_s4/recovery_control.py','coco_s4/serial_recovery.py','coco_s4/dimension_release.py'}


def normalized_python(content, removed=(), workflow=None):
    tree=ast.parse(content)
    tree.body=[n for n in tree.body if not (
        isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in removed)]
    if workflow is not None:
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORKFLOW' for t in n.targets):
                n.value=ast.Constant(INVALID_WORKFLOW)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in ('INVALIDATED_WORKFLOW','INVALIDATED_OUTPUT') for t in n.targets))]
    return ast.dump(tree,include_attributes=False)


def validate_source_delta(name, before, after, workflow):
    if before==after:return 'byte_exact'
    if name in PLUMBING:return 'explicitly_bound_serial_control_only'
    if name=='coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json':
        old=json.loads(before);new=json.loads(after)
        assert new['authorization_sha256']==old['authorization_sha256']
        assert new['invalid_workflow']==INVALID_WORKFLOW and new['invalid_runtime']==old['invalid_runtime']
        assert new['formal_relaunch_allowed'] is True and new['workflow_run_id']==workflow
        assert new['allow_other_user_gpu'] is True and new['requires_later_explicit_human_relaunch'] is False
        return 'explicit_human_conditional_relaunch_and_shared_admission'
    if name.startswith('coco_s4/configs/') and name.endswith('.json'):
        old=json.loads(before);new=json.loads(after)
        assert old.pop('workflow_run_id')==INVALID_WORKFLOW and new.pop('workflow_run_id')==workflow
        assert old==new,'Scientific phase configuration changed'
        return 'workflow_literal_only'
    rules={'coco_s4/runtime.py':('assert_formal_launch_authorized',),
           'coco_s4/serial.py':('capacity',),'coco_s4/phase_fourier.py':('formal_gate',)}
    if name in rules:
        assert normalized_python(before,rules[name])==normalized_python(after,rules[name]),name
        return 'authorization_capacity_or_formal_evidence_guard_only'
    if name=='coco_s4/phase_contract.py':
        assert normalized_python(before,workflow=workflow)==normalized_python(after,workflow=workflow)
        return 'unique_workflow_and_invalid_predecessor_literals_only'
    raise ValueError('Validated scientific computation drift: '+name)


def evidence_identity(manifest):
    """Return real evidence identity only after verifying the entire explicit binding."""
    binding=manifest.get('validated_computation_binding')
    if not binding:
        return dict(source_commit=manifest['source_commit'],release_sha256=None,
                    phase_config_sha256=manifest['files'][manifest['phase_config_path']])
    from .phase_contract import OWNED,WORKFLOW
    reference=Path(binding['reference_source_root']).resolve()
    anchor=Path(binding['reference_release_path']).resolve()
    assert reference.is_relative_to((OWNED/'releases').resolve())
    assert anchor.is_relative_to((OWNED/'releases').resolve()) and sha(anchor)==binding['reference_release_sha256']
    original=load(anchor)
    assert original['source_commit']==VALIDATED_COMMIT and commit(reference)==VALIDATED_COMMIT
    assert manifest['native_sources']==original['native_sources'] and manifest['versions']==original['versions']
    assert manifest['data_files']==original['data_files'] and manifest['data_root']==original['data_root']
    assert manifest['teacher_dependency']==original['teacher_dependency']
    old_cal=dict(original['calibration_dependency']);new_cal=dict(manifest['calibration_dependency'])
    old_cal.pop('accepted_path');new_path=new_cal.pop('accepted_path')
    assert old_cal==new_cal and Path(new_path)==OWNED/'outputs'/WORKFLOW/'calibration/accepted.json'
    assert manifest['workflow_run_id']==WORKFLOW and WORKFLOW!=INVALID_WORKFLOW
    changes={}
    for name,digest in original['files'].items():
        before=reference/name;after=ROOT/name
        assert sha(before)==digest and sha(after)==manifest['files'][name]
        disposition=validate_source_delta(name,before.read_bytes(),after.read_bytes(),WORKFLOW)
        if disposition!='byte_exact':changes[name]=dict(before=digest,after=sha(after),reason=disposition)
    assert changes==binding['routing_deltas'],'Unbound source delta'
    assert sha(ROOT/AUTHORIZATION)==binding['human_authorization_sha256']
    probe=Path(binding['exact_export_device_summary_path']).resolve()
    assert probe.is_relative_to((OWNED/'validation').resolve()) and sha(probe)==binding['exact_export_device_summary_sha256']
    device=load(probe)
    assert device['state']=='passed' and device['actual_native_forward_calls']==2 and device['CPU_output_rejected']
    assert device['production_hook_evidence'][0]['devices']==['cuda:0'] and device['production_hook_evidence'][0]['cuda_indices']==[0]
    assert device['exact_production_hook_source_sha256']==manifest['files']['coco_s4/phase_export.py']
    assert device['parameter_updates']==0 and device['Teacher_unchanged'] and device['checkpoint_unchanged']
    return dict(source_commit=VALIDATED_COMMIT,release_sha256=binding['reference_release_sha256'],
                phase_config_sha256=original['files'][original['phase_config_path']])


def invalidated_predecessor(cfg):
    """Verify stopped TFSS-invalid generation without modifying any old artifact."""
    from .phase_contract import INVALIDATED_OUTPUT
    from .serial import live
    p=cfg['invalidated_predecessor'];assert p['run_id']==INVALID_WORKFLOW
    expected={'pipeline_state.json','fourier24/status.json','jobs/fourier24/exit.json','receipts/SCIENTIFIC_INVALIDATION_20261026.json'}
    assert set(p['files'])==expected
    for name,digest in p['files'].items():assert sha(INVALIDATED_OUTPUT/name)==digest
    s=load(INVALIDATED_OUTPUT/'pipeline_state.json');v=load(INVALIDATED_OUTPUT/'receipts/SCIENTIFIC_INVALIDATION_20261026.json')
    assert s['state']=='FAILED' and not live(s.get('controller')) and not live(s.get('active_child'))
    assert v['state']=='STOPPED_SCIENTIFIC_INVALID' and v['reason']=='MISSING_TFSS_OPTIMIZATION_WIRING'
    assert not (INVALIDATED_OUTPUT/'export25k').exists()
    return p
