import json,pathlib,sys,argparse
E=next(p/'execution' for p in pathlib.Path(__file__).resolve().parents if (p/'coco_s4').is_dir());sys.path.insert(0,str(E));import local_zip_recovery_20261024 as b
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();tag=a.tag;spec=json.loads((E/(tag+'.json')).read_text());release=E/tag/'source_release.json'
wait=json.loads((E/tag/'completed_wait.json').read_text());assert wait['status']['detail']['exit_code']==0
code=r'''
from pathlib import Path
import csv,hashlib,importlib.util,json,subprocess,sys
import torch
torch.set_num_threads(1)
root=Path(sys.argv[1]);out=Path(sys.argv[2])/'result';m=json.loads(Path(sys.argv[3]).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
 return h.hexdigest()
assert sha(Path(sys.argv[3]))==sys.argv[4]
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==m['source_commit']
for name,digest in m['files'].items():assert sha(root/name)==digest,name
native=Path(importlib.util.find_spec('ultralytics').origin).parent
for name,digest in m['native_sources'].items():assert sha(native/name)==digest,name
summary_name='cpu_probe_summary.json' if (out/'cpu_probe_summary.json').exists() else 'smoke_summary.json'
s=json.loads((out/summary_name).read_text());phase=json.loads((out/'phase_spec.json').read_text())
assert s['state']=='passed' and s['actual_forward_tensor_devices']==[s['actual_device']]
assert s['actual_device'] in ('cpu','cuda:0')
assert s['actual_cuda_device_indices']==([0] if s['actual_device']=='cuda:0' else [])
assert s['actual_fourier_optimizer_updates']==1 and 1<=s['production_forward_calls']<=100 and s['selected_channel_count']==39
assert s['non_target_class_count']==79 and s['independent_P3_selected_count']==48
rows=list(csv.DictReader((out/'diagnostics.csv').open()));assert len(rows)==1
row=rows[0]
for name in ('L_BG','L_rel','L_total','grad_norm_fourier','grad_BG_safe_norm','grad_rel_safe_norm'):
 assert __import__('math').isfinite(float(row[name]))
assert int(float(row['step']))==0 and int(float(row['epoch']))==1
assert float(row['grad_BG_safe_norm'])>0 and float(row['grad_rel_safe_norm'])>0
assert s['projector_precision_validation']['all_native_cached_ranks_eight'] is True
assert float(row['safe_bg_removal_alpha'])==.5
# The inherited CSV field is a float32 mean, not an individual integer rank.
assert __import__('math').isfinite(float(row['safe_projector_rank']))
assert float(row['strict_tsfl_final_selected_count'])==39
assert float(row['safe_bg_feature_layer_id'])==21 and float(row['orth_feature_layer_id'])==15
assert float(row['outside_support_max'])==0 and float(row['max_abs_delta'])<=8/255
pack=torch.load(out/'global_params.pt',map_location='cpu',weights_only=True)
assert tuple(pack['fourier_coeff'].shape)==(16,3) and torch.isfinite(pack['fourier_coeff']).all()
import yaml
source_eps=yaml.safe_load((root/m['config_path']).read_text())['experiment']['eps']
assert pack['eps']==source_eps and abs(source_eps-8/255)<1e-9
assert pack['target_class_id']==0 and pack['support_mode']=='bbox_half_overlap'
assert sha(Path(m['teacher_dependency']['checkpoint_path']))==m['teacher_dependency']['checkpoint_sha256']
assert sha(out/'native_rng_state.pt')==s['native_rng_sha256']
rng=torch.load(out/'native_rng_state.pt',map_location='cpu',weights_only=False)
rng_keys={'python','numpy','torch_cpu'} | ({'torch_cuda'} if s['actual_device']=='cuda:0' else set())
for point in ('before','after'):assert set(rng[point])==rng_keys
sys.path.insert(0,str(root))
from coco_s4.fourier_gradient_audit import validate_production_batch16
validate_production_batch16(s)
eligible=json.loads(Path(phase['imported_calibration_artifacts']['collected/eligible_support_manifest.json']['path']).read_text())
import numpy as np
expected=[Path(eligible['paths'][int(i)]).stem for i in np.random.RandomState(0).permutation(eligible['count'])[:16]]
assert expected==s['actual_training_image_ids']==s['expected_training_image_ids']
assert s['actual_native_batch_sizes']==[16,16] and s['actual_training_images']==16
assert s['cache_memory']['svd_precision_policy']=='selected_dimension_float64_svd_v1'
for name,digest in m['failed_dimension_predecessor']['files'].items():
 assert sha(Path('/project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_dimcompat_v1')/name)==digest
proof=json.loads((out/'per_loss_fourier_gradients.json').read_text())
assert sha(out/'per_loss_fourier_gradients.json')==s['per_loss_fourier_gradient_proof_sha256']
assert proof['measurement']=='direct_autograd_fourier_coeff_v1' and proof['target']=='trainer.fourier_coeff'
assert proof['parameter_shape']==[64,3] and proof['parameter_is_leaf'] is True and proof['parameter_device']==s['actual_device']
assert proof['EOT_draws']==2 and proof['optimizer_gradient_accumulation_unchanged'] is True
import numpy as np
for name in ('L_BG','L_rel'):
 gradients=np.asarray([d[name] for d in proof['gradients']],dtype=np.float32)
 assert gradients.shape==(2,64,3) and np.isfinite(gradients).all()
 for d,g in enumerate(gradients):
  assert np.linalg.norm(g)>0
  assert np.isclose(np.linalg.norm(g),proof['per_draw_norms'][d][name],rtol=2e-6,atol=1e-10)
 assert np.isclose(np.linalg.norm(gradients.mean(0)),s['SafeBG_loss_Fourier_gradient_norms'][name],rtol=2e-6,atol=1e-10)
 assert s['SafeBG_loss_Fourier_gradient_norms'][name]>0
categories=[json.loads(line) for line in (out/'categorical_diagnostics.jsonl').read_text().splitlines()]
assert len(categories)==2
for row_ids in categories:
 assert len(row_ids['active_nt_class_ids'])==3
 assert all(type(v) is int and 1<=v<=79 for v in row_ids['active_nt_class_ids'])
tfss=s['tfss_training_proof']
assert tfss['total_EOT_draws']==2 and tfss['adaptive_scaled_inputs']==2 and tfss['frozen_scaled_inputs']>=2
for draw in tfss['retained_draws']:
 assert draw['W_detached'] and draw['weighted_delta_exact']
 assert .85<=draw['W_support_quantiles'][0]<=draw['W_support_quantiles'][-1]<=1
 assert draw['weighted_vs_base_delta_max']>0 and draw['weighted_vs_base_coefficient_gradient_max']>0
 assert draw['weighted_delta_coefficient_gradient_finite']
export=s['native_export_link_validation']
assert export['state']=='passed' and export['native_final_image_matches_export_exact']
assert export['decoded_PNG_roundtrip_exact'] and export['Teacher_unchanged']
assert export['actual_composition_device']==s['actual_device'] and export['extra_model_forwards']==0
assert export['decoded_budget']['outside_nonzero']==0 and export['decoded_budget']['max_bytes']<=8 and export['decoded_budget']['overlap_max_bytes']<=4
ck=torch.load(out/'native_s4/latest_continuation.pt',map_location='cpu',weights_only=False)
assert ck['boundary']=='complete_optimizer_step' and ck['cursor_state']['step']==1
assert ck['trainer']['fourier_coeff'].shape==(64,3) and torch.isfinite(ck['trainer']['fourier_coeff']).all()
assert ck['optimizer']['state'] and ck['scheduler']['last_epoch']==0
assert set(ck['rng'])=={'python','numpy','torch','cuda'}
assert bool(ck['rng']['cuda'])==(s['actual_device']=='cuda:0')
assert not (out/'accepted.json').exists()
inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*.pt'))]
assert inventory and not torch.cuda.is_initialized()
print(json.dumps(dict(state='passed',source_commit=m['source_commit'],source_files_verified=len(m['files']),
 native_files_verified=len(m['native_sources']),actual_runtime_summary=s,coefficient_gradient_proof=proof,categorical_per_EOT_records=categories,F01_closed=True,F02_closed_additive_reporting=True,actual_original_batch16_audited=True,rank_validation_uses_exact_cached_integer_ranks=True,inherited_rank_display_mean=float(row['safe_projector_rank']),failed_predecessor_hashes_unchanged=True,one_actual_CSV_row=row,
 output_Fourier_tensor_shape=[16,3],checkpoint_paths_remaining=inventory,
 native_RNG_keys=sorted(rng_keys),Teacher_checkpoint_unchanged=True,
 audit_GPU_initialized=False,official_metrics_disabled=True,artifact_ingest_disabled=True,deleted_paths=[])))
'''
r=b.transport().run([b.PYTHON,'-B','-c',code,spec['remote']['repo_root'],spec['remote']['output_root'],spec['anchors'][0]['remote_path'],__import__('hashlib').sha256(release.read_bytes()).hexdigest()])
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3500:])
x=json.loads(r.stdout);b.save('COCO_S4_REAL_BATCH16_'+tag+'_INDEPENDENT_AUDIT.json',x)
print(json.dumps({k:x[k] for k in ('state','source_commit','source_files_verified','native_files_verified','audit_GPU_initialized','Teacher_checkpoint_unchanged')}))
