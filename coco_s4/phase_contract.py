"""Dependency and one-review contracts for immutable post-Teacher releases.

These checks register no job and never alter the running controller or Teacher.
An accepted previous stage is required even for a later phase's bounded smoke.
"""
from pathlib import Path

from .runtime import ROOT, load, safe_relative, sha, verify_release

WORKFLOW = 's4_coco_experiment_a_seed0_20261025_v3'
OWNED = Path('/project/guoqinglong/projects/s4_coco_experiment_a_20261025')
WORKFLOW_OUTPUT = OWNED/'outputs'/WORKFLOW
TEACHER_COMMIT = 'ee0b4256fdd344c5a275e6d15efc78cb7636d6e1'
TEACHER_RELEASE = 'c520c67ea99effba2a81d79695d0102cd866c9475a8a7284793f6f0b3d426e1a'


def teacher_dependency(binding, workflow_output=WORKFLOW_OUTPUT):
    """Fixed native EMA last@200; never select best or an intermediate epoch."""
    root=Path(workflow_output).resolve()
    accepted=root/'teacher200/accepted.json'
    last=root/'teacher200/train/weights/last.pt'
    full=root/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
    expected={'accepted_path':str(accepted),'checkpoint_path':str(last),'full_state_path':str(full)}
    if any(binding.get(k)!=v for k,v in expected.items()):
        raise ValueError('Teacher dependency route differs from this workflow fixed last@200')
    if sha(accepted)!=binding['accepted_sha256']:
        raise ValueError('Accepted Teacher receipt changed')
    a=load(accepted)
    required=dict(state='ACCEPTED',source_commit=TEACHER_COMMIT,release_sha256=TEACHER_RELEASE,
                  completed_epochs=200,finite_loss=True,zero_validation_forwards=True)
    if any(a.get(k)!=v for k,v in required.items()):
        raise ValueError('Teacher is not the accepted train-only200 dependency')
    if (sha(last)!=binding['checkpoint_sha256'] or a['last_checkpoint_sha256']!=binding['checkpoint_sha256']
            or sha(full)!=binding['full_state_sha256'] or a['full_state_sha256']!=binding['full_state_sha256']):
        raise ValueError('Fixed Teacher checkpoint/full native state drift')
    return last,a


def phase_release(path, phase):
    manifest=verify_release(path)
    if manifest.get('phase')!=phase or manifest.get('workflow_run_id')!=WORKFLOW:
        raise ValueError('Wrong phase/workflow release')
    if not ROOT.resolve().is_relative_to((OWNED/'releases').resolve()):
        raise ValueError('A future phase must use a separate owned immutable release')
    rel=str(safe_relative(manifest['phase_config_path']))
    if rel not in manifest['files']:
        raise ValueError('Phase configuration is outside the source hash closure')
    config=load(ROOT/rel)
    if config['phase']!=phase or config['workflow_run_id']!=WORKFLOW:
        raise ValueError('Phase configuration attribution mismatch')
    last,accepted=teacher_dependency(manifest['teacher_dependency'])
    return manifest,config,last,accepted


def formal_phase_gate(registration_path, phase, release_path, manifest, config_sha256):
    """The stable controller verifies order; the scientific child verifies closure."""
    registration_path=Path(registration_path).resolve()
    if registration_path!=(WORKFLOW_OUTPUT/'receipts'/(phase+'_READY.json')).resolve():
        raise ValueError('Wrong existing-controller registration route')
    r=load(registration_path)
    relative=safe_relative(r['release_relative'])
    if (ROOT/relative).resolve()!=Path(release_path).resolve():
        raise ValueError('Registration release escaped this immutable source')
    required=dict(phase=phase,workflow_run_id=WORKFLOW,pre_run_result='pass',
                  source_root=str(ROOT),release_sha256=sha(release_path),
                  module='coco_s4.phase_teacher_eval',
                  previous_stage_receipt_sha256=manifest['teacher_dependency']['accepted_sha256'])
    if any(r.get(k)!=v for k,v in required.items()):
        raise ValueError('Registration is not bound to this accepted Teacher/release')
    closure_path=Path(r['scientific_closure_path']).resolve()
    if closure_path.parent!=(WORKFLOW_OUTPUT/'receipts').resolve() or sha(closure_path)!=r['scientific_closure_sha256']:
        raise ValueError('Scientific closure route/hash mismatch')
    c=load(closure_path)
    expected=dict(phase=phase,source_commit=manifest['source_commit'],release_sha256=sha(release_path),
                  phase_config_sha256=config_sha256,pre_run_result='pass',reviewer_count=1,
                  teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'])
    if any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('Missing current-candidate one-review scientific closure')
    smoke_path=Path(c['smoke_path']).resolve()
    if not smoke_path.is_relative_to((OWNED/'validation').resolve()) or sha(smoke_path)!=c['smoke_sha256']:
        raise ValueError('Smoke evidence must belong to the isolated owned validation root')
    s=load(smoke_path)
    expected_smoke=dict(state='passed',phase=phase,source_commit=manifest['source_commit'],
                        release_sha256=sha(release_path),phase_config_sha256=config_sha256,
                        teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                        official_metrics_disabled=True,artifact_ingest_disabled=True,
                        computation_kind='inference',finite_outputs=True,production_entrypoint_reached=True)
    if (any(s.get(k)!=v for k,v in expected_smoke.items()) or not 1<=s.get('actual_inference_batches',0)<=100
            or not 1<=s.get('total_raw_forwards_including_warmup',0)<=100):
        raise ValueError('Actual current-release bounded inference smoke is required')
    probe=s.get('baseline_equivalence_probe',{})
    if (not probe.get('raw_outputs_exact') or not probe.get('decoded_predictions_exact')
            or probe.get('absolute_difference')!=0 or probe.get('tolerance')!=0):
        raise ValueError('Canonical native inference equivalence has not passed')
    return c,s
