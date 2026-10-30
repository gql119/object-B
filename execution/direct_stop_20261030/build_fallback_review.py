"""Build the single lean scientific packet only after real current GPU gates."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
SOURCE='87456fa7a57206551b0e3be0253cc4866f0cdcf1'
TAG='s4_at_serial_seed0_20261030_fresh_v1'


def load(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    tags=['s4_at_selected_native_20261030_gpu01','s4_at_selected_baseline_20261030_gpu01','s4_at_selected_phase_20261030_gpu01']
    paths=[HERE/'current_probes'/tag/'summary.json' for tag in tags];native,baseline,resume=map(load,paths)
    cpu=load(HERE/'CURRENT_RUNTIME_CPU_REUSE.json');protocol=load(ROOT/'robustness_s4_at/configs/PROTOCOL.json')
    assert all(v['state']=='passed' and v['source_commit']==SOURCE for v in (native,baseline,resume))
    assert cpu['source_commit']==SOURCE and not cpu['intentionally_changed_scientific_files']
    assert set(native['arms'])==set(protocol['order']) and native['actual_outer_batches']<=64
    assert resume['exact_model_optimizer_scaler_scheduler_EMA_gradients_PGD_RNG'] and resume['tolerance']==0
    proof=baseline['baseline_equivalence_probe'];assert proof['absolute_difference']==0 and proof['tolerance']==0
    proof=dict(proof,evidence_paths=[str(paths[1])])
    specs=[load(HERE/'current_probes'/tag/'runspec.json') for tag in tags]
    owned=protocol['remote_owned_root'];gate=owned+'/preflight_20261030_fresh_v1/accepted.json'
    command=' '.join(['/project/guoqinglong/miniconda3/envs/cfbdm/bin/python','-B','execution/direct_stop_20261030/fresh_serial_queue.py',
        '--out',owned+'/outputs/'+TAG+'/result','--run-id',TAG,'--source-commit',SOURCE,'--preflight',gate])
    packet=dict(schema_version='prerun.scientific-review.v1',review_mode='scientific_review',repo_root=str(ROOT),
        pre_run_code_commit=SOURCE,review_diff_base_commit='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb',
        approved_basis=[str(DOC/'USER_AT_TASK_20261027.md'),str(DOC/'USER_DIRECT_TERMINATION_AND_RESTART_20261030.md')],
        implementation_intent='Fresh epoch0 four-arm OriginalBatch36 fallback after finite optimization tests did not establish reliable speed gains. All19 scientific package files and60 OriginalS4 files are byte-exact accepted51ce source; no Candidate02/03 or new batch is adopted. Review scientific protocol, source-to-sink, fresh result identity and reference equivalence, excluding already CPU-validated RRCTL/lifecycle scaffolding. Prior AT-SCI-001 negative and actual main phase repair remain retained. No old checkpoint continuation, no ST finetuning, no S4 regeneration or COCO change.',
        exact_command=command,output_collision_policy='fail',
        local_validation=[dict(command='audit_current_runtime.py --source '+SOURCE,exit_code=0,
            observation='CURRENT_RUNTIME_CPU_REUSE.json:19 committed scientific files unchanged;60 OriginalS4 SHA exact; accepted CPU native contracts reused; no models/GPU/data census.'),
            dict(command='check_baseline_metadata.py',exit_code=0,observation='Exactly two diagnostic batch-setting labels bound to actual frozen protocol; reference/candidate math and zero tolerance unchanged.')],
        pre_review_smoke=dict(schema_version='prerun.pre-review-smoke.v1',disposition='passed',computation_kind='training',
            candidate_commit=SOURCE,run_id=tags[0],exact_command=' '.join(specs[0]['workload']['argv']),exit_code=0,
            step_budget=64,completed_steps=native['actual_outer_batches'],finite_loss=True,production_entrypoint_reached=True,
            isolated_output=True,official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',
            checkpoint_cleanup_completed=False,checkpoint_paths_remaining=native['checkpoint_paths_remaining'],deleted_paths=[],
            evidence_paths=[str(p) for p in paths]+[str(HERE/'CURRENT_RUNTIME_CPU_REUSE.json')],
            retained_evidence_paths=[str(p) for p in paths],baseline_equivalence_required=True,baseline_equivalence_probe=proof),
        critical_values=[
            dict(name='Static OriginalS4 and full GT',source='PROTOCOL/data.py/accepted materialization identities',sink='formal worker data_view -> full16551 clean or static protected train; clean4952 val; labels unchanged',evidence=str(paths[0])+'; CURRENT_RUNTIME_CPU_REUSE.json; retained original full-data audit referenced by current smoke'),
            dict(name='Dynamic full-image PGD10',source='PROTOCOL pgd4 eps4 step1 and pgd8 eps8 step2 /255, seed0 random start',sink='ATTrainer.preprocess_batch -> unchanged PGDAttack -> native full-GT box/cls/DFL input gradients, FP32 inner and frozen BN stats',evidence=str(paths[0])+' four actual640/batch36/workers16 CUDA arms with10 forwards per outer and strict budgets'),
            dict(name='Native outer AMP SGD/EMA',source='OriginalBatch36/nbs64/native warmup, optimizer and scaler',sink='unchanged native_loop and ATTrainer; native accumulation2 and legal overflow skips; no changed precision/scaler/update math',evidence=str(paths[0])+' actual SGD in all4; '+str(paths[1])+' independent native loss diff0'),
            dict(name='Checkpoint phase and fresh identity',source='Fresh seed0 YAML; checkpoint.py/native_loop.py accepted51ce repair',sink='four fresh init hashes equal; production same-run completed-epoch resume preserves optimizer-attempt phase; old abandoned checkpoint never an input',evidence=str(paths[2])+' exact controlled transformed-input resume tolerance0; no worker-prefetch bitwise claim'),
            dict(name='Evaluation and paired attribution',source='worker.evaluate/metrics.py/report.py unchanged accepted source',sink='cleanVOC4952/all20 last.pt@200 primary and best secondary; CleanAT4 vs PoisonAT4 and CleanAT8 vs PoisonAT8; no Teacher denominator',evidence='Committed source byte audit; evaluation runs only after each real200epoch arm, no official metrics produced by smoke')],
        experiment=dict(benchmark='VOC20 OriginalS4 AT4/AT8 four fresh200epoch arms',dataset='acceptedVOC0712 train16551 cleanval4952 Person14',
            checkpoint='fresh seed0 random-YAML epoch0, primary last.pt@200; no old checkpoint input',seed=0,
            metric_policy='matched-budget full20/perclass clean4952 last200primary/bestsecondary; null where unmeasured',output_path=owned+'/outputs/'+TAG+'/result'))
    with (HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW_PACKET.json').open('x',encoding='utf8') as f:json.dump(packet,f,indent=2)
    audit=dict(state='PASS_CURRENT_GPU_MAIN_AUDIT',source_commit=SOURCE,scientific_files_byte_exact=True,
        native_steps=native['actual_outer_batches'],four_actual_SGD={k:v['actual_updates'] for k,v in native['arms'].items()},
        initial_sha256=native['initial_model_sha256'],baseline_diff=proof['absolute_difference'],baseline_tolerance=0,
        phase_resume_exact=True,old_checkpoint_continuation=False,evidence=[dict(path=str(p),sha256=sha(p)) for p in paths])
    with (HERE/'CURRENT_GPU_MAIN_AUDIT.json').open('x',encoding='utf8') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit))
