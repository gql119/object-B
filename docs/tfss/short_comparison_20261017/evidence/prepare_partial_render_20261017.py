import json, hashlib
from pathlib import Path

p=Path('execution/tfss_cu_short_analysis_seed0_20261017_v1.json')
d=json.loads(p.read_text()); inputs=json.loads(d['workload']['argv'][4])
wrapper=r'''import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
inputs=json.loads(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False,parents=True)
os.environ['MPLCONFIGDIR']=str(out/'runtime/mpl');os.environ['XDG_CACHE_HOME']=str(out/'runtime/cache')
import torch
torch.set_num_threads(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
methods=['S0-reference','S0-noop','TFSS-C','TFSS-U','Uniform-C','Uniform-U']
evidence={'state':'PARTIAL_59_OF_60_COMPARISON_UPDATES','new_optimizer_updates':0,'model_forward':False,'GPU_use':False,'inputs':inputs,'trials':[],'checkpoint_metadata':[],'raw_json_sha256':{},'missing':['batch1 Uniform-U step5','batch1 Uniform-U final NTGT diagnostics','batch1 Uniform-U final derived maps'],'limitations':'two sources, retained complete batch0 plus partial repaired batch1; no single complete whole experiment'}
traces={}
for key,comp in inputs.items():
 b=int(key);src=Path(comp['output_root']);control=Path(comp['control_root'])
 binding=json.loads((control/'binding.json').read_text());cs=json.loads((control/'status.json').read_text());status=json.loads((src/'status.json').read_text())
 assert cs['state']=='failed' and binding['commit']==comp['commit']==status['source_commit']
 evidence[f'batch{b}_environment']=status['environment']
 for m in methods:
  trial=src/f'batch{b}_{m}';rows=json.loads((trial/'trace.json').read_text());n=4 if b==1 and m=='Uniform-U' else 5
  assert len(rows)==2*n and [(r['step'],r['eot']) for r in rows]==[(s,e) for s in range(1,n+1) for e in [1,2]]
  traces[b,m]=rows
  record=next((x for x in status['trials'] if x['batch']==b and x['method']==m),None)
  if record:
   assert record['state']=='complete' and record['steps']==n and len(set(record['frozen_adaptive_before_after']))==1
  evidence['trials'].append({'batch':b,'method':m,'completed_updates':n,'terminal_models_unchanged':True if record else None,'terminal_model_hash_check':'PASSED' if record else 'NOT_RUN: interrupted before final hash check'})
  for file in [trial/'trace.json',trial/'nt_gt_diagnostics.json']:
   if file.exists():evidence['raw_json_sha256'][f'batch{b}/{m}/{file.name}']=hashlib.sha256(file.read_bytes()).hexdigest()
  for step in range(1,n+1):
   path=trial/f'step{step}.pt';pack=torch.load(path,map_location='cpu',weights_only=False)
   assert pack['step']==step and pack['scheduler']['last_epoch']==0 and len(pack['rng'])==4 and len(pack['rng'][3])>0
   assert torch.isfinite(pack['fourier_coeff']).all() and torch.isfinite(pack['freq_score']).all()
   adam=pack['optimizer']['state'];assert adam and all(float(v['step'])==step for v in adam.values())
   assert all(torch.isfinite(v[k]).all() for v in adam.values() for k in ['exp_avg','exp_avg_sq'])
   evidence['checkpoint_metadata'].append({'batch':b,'method':m,'step':step,'Adam_step':step,'scheduler_last_epoch':0,'RNG_present':True,'finite_coefficients_and_Adam':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  assert not (trial/f'step{n+1}.pt').exists()
 assert json.loads((src/f'batch{b}_S0-noop/equivalence.json').read_text())['max_difference']==0
assert len(evidence['checkpoint_metadata'])==59
for loss in ['L_BG','L_rel','L_ret','L_total']:
 fig,axes=plt.subplots(1,2,figsize=(10,3.2))
 for b,ax in enumerate(axes):
  for m in methods:
   rows=traces[b,m];steps=sorted({r['step'] for r in rows});ax.plot(steps,[np.mean([r[loss] for r in rows if r['step']==s]) for s in steps],marker='.',label=m)
  ax.set(title=f'batch {b}: {loss}',xlabel='update index; pre-update forward');ax.set_xticks([1,2,3,4,5])
 axes[1].legend(fontsize=6);fig.suptitle('Partial cohort: Uniform-U batch1 stops at 4; no AP/victim',fontsize=9);fig.tight_layout();fig.savefig(out/(loss+'.png'),dpi=150);plt.close(fig)
for b,indices in [(0,[0,1]),(1,[0,1,8,15])]:
 src=Path(inputs[str(b)]['output_root'])
 with np.load(src/f'maps_batch{b}.npz') as maps:
  for i in indices:
   fig,axes=plt.subplots(1,3,figsize=(9,3))
   for ax,key,title in zip(axes,['q','nt_risk','q_valid'],['Qt: invalid region strength preserved','NT risk','Qt valid attribution domain']):
    ax.imshow(maps[f'{key}_{i}'],vmin=0,vmax=1,cmap='viridis');ax.set_title(title,fontsize=7);ax.axis('off')
   fig.suptitle(f'fixed batch{b}, image index{i}; derived maps only',fontsize=9);fig.tight_layout();fig.savefig(out/f'heatmap_b{b}_i{i}.png',dpi=150);plt.close(fig)
 arrays={m:np.load(src/f'batch{b}_{m}/final_recorded_derived_maps.npz') for m in methods if not (b==1 and m=='Uniform-U')}
 for m,a in arrays.items():
  delta=a['actual_delta'];support=a['support'];assert np.max(np.abs(delta)*np.equal(support,0))==0
  assert (np.abs(delta)<=support*8/255+1e-7).all()
 for i in indices:
  fig,axes=plt.subplots(1,6,figsize=(15,3.2));im=None
  for ax,m in zip(axes,methods):
   ax.set_title(m,fontsize=8);ax.axis('off')
   if m not in arrays:ax.text(.5,.5,'NOT_RUN\nstep5 incomplete',ha='center',va='center',transform=ax.transAxes,fontsize=9);continue
   a=arrays[m];im=ax.imshow(np.ma.masked_where(a['support'][i,0]==0,a['actual_delta'][i].mean(0)),cmap='RdBu_r',vmin=-8/255,vmax=8/255)
  fig.suptitle(f'batch{b}, image index{i}: signed RGB-mean actual perturbation; pre-update step5/EOT2',fontsize=9)
  fig.colorbar(im,ax=axes.tolist(),fraction=.015,pad=.01,label='same fixed 8/255 color scale');fig.savefig(out/f'noise_b{b}_i{i}.png',dpi=150,bbox_inches='tight');plt.close(fig)
 for a in arrays.values():a.close()
(out/'CPU_METADATA_AND_PROVENANCE.json').write_text(json.dumps(evidence,indent=2))
(out/'RENDER_COMPLETE.json').write_text(json.dumps({'state':'complete','comparison_state':'PARTIAL_59_OF_60_UPDATES','optimizer_updates':0,'model_forward':False,'GPU_use':False,'source_RGB_exported':False,'derived_images_only':True,'figures':16}))
'''
d['workload']['argv']=[d['remote']['python'],'-B','-c',wrapper,json.dumps(inputs),d['remote']['output_root']]
d['source']['allowed_post_commit_paths']=['execution/','docs/tfss/short_comparison_20261017/']
paths=['CPU_METADATA_AND_PROVENANCE.json','RENDER_COMPLETE.json']+[x+'.png' for x in ['L_BG','L_rel','L_ret','L_total']]+[f'{kind}_b{b}_i{i}.png' for kind in ['heatmap','noise'] for b,indices in [(0,[0,1]),(1,[0,1,8,15])] for i in indices]
d['artifacts']=[{'path':x,'required':True} for x in paths]
adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'RENDER_COMPLETE.json';d=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({'protocol':'rrctl.adapter.v1','healthy':True,'complete':d.get('state')=='complete','progress':{'stage':'partial_output_only_CPU_render'},'observations':d,'artifacts':"+repr(paths)+" if d else []}))"
for h in d['health'].values():h['adapter_argv']=[d['remote']['python'],'-B','-c',adapter]
d['metadata'].update(inputs_partial=True,completed_input_updates=59,planned_input_updates=60,new_optimizer_updates=0,render_wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),no_model_forward=True)
p.write_text(json.dumps(d,indent=2),encoding='utf8')
print('Prepared UNLAUNCHED CPU-only partial render',len(paths),'artifacts')
