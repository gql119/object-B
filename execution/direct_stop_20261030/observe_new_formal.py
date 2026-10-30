"""Read-only exact binding, GPU PID/start and bounded native AT progress."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
TAG='s4_at_serial_seed0_20261030_fresh_v1'
SOURCE='87456fa7a57206551b0e3be0253cc4866f0cdcf1'
CODE=r'''
import datetime,json,os,subprocess,sys
from pathlib import Path
v=json.loads(sys.argv[1]);o=Path(v['output'])/'result';repo=Path(v['repo']).resolve();b=v['binding'];assert os.getuid()==1014
s=json.loads((o/'pipeline_state.json').read_text());arm=s.get('current_run');progress={};amp={};initial=None
if arm:
 p=o/arm/'audit/progress.json';a=o/arm/'audit/amp/latest.json';i=o/arm/'audit/initialization.json'
 if p.exists():
  r=json.loads(p.read_text());progress={k:r.get(k) for k in ('run_id','epoch','completed_epochs','at_batches','actual_updates','legal_amp_skips','loss','UTC')}
  g=r.get('latest_PGD',{});progress.update(actual_shape=g.get('actual_input_shape'),actual_PGD_steps=g.get('actual_inner_forwards'),raw_devices=g.get('actual_raw_devices'))
 if a.exists():amp=json.loads(a.read_text())
 if i.exists():initial=json.loads(i.read_text())['parameter_sha256']
def identity(pid,ticks=None):
 p=Path('/proc')/str(pid)
 try:
  row=(p/'stat').read_text().rsplit(')',1)[1].split();start=int(row[19]);uid=p.stat().st_uid
  return dict(pid=pid,start_ticks=start,uid=uid,alive=row[0]!='Z',expected_start_matches=ticks is None or start==ticks,ppid=int(row[1]))
 except FileNotFoundError:return dict(pid=pid,alive=False)
identities=dict(executor=identity(b['executor_pid'],b['executor_start_ticks']),queue=identity(b['workload_pid'],b['workload_start_ticks']),scientific_GPU=[])
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
for line in apps.splitlines():
 if not line.strip():continue
 pid=int(line.strip());p=Path('/proc')/str(pid)
 try:
  if p.stat().st_uid!=1014:continue
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x];cwd=(p/'cwd').resolve()
  if 'robustness_s4_at.worker' in argv and '--formal' in argv and '--out' in argv and arm and argv[argv.index('--out')+1]==str(o/arm):
   r=identity(pid);r.update(cwd_matches_release=cwd==repo,parent_matches_queue=r['ppid']==b['workload_pid'],run_id=arm);identities['scientific_GPU'].append(r)
 except FileNotFoundError:pass
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
tail=None
if arm and not progress:
 f=o/(arm+'.console.log')
 if f.exists():
  with f.open('rb') as stream:
   stream.seek(max(0,f.stat().st_size-1800));tail=stream.read().decode(errors='replace')
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source,pipeline_state=s['state'],current_arm=arm,progress=progress,AMP=amp,initial_model_sha256=initial,identities=identities,initialization_console_tail=tail)))
'''

if __name__=='__main__':
    spec=json.loads((HERE/TAG/'runspec.json').read_text());ctl,t=services();inspect=ctl.inspect(TAG);h=ctl.health(TAG,phase='periodic')
    payload=dict(output=spec['remote']['output_root'],repo=spec['remote']['repo_root'],binding=inspect['binding'])
    r=t.run([PYTHON,'-B','-c',CODE,json.dumps(payload)])
    if r.returncode:raise RuntimeError(str(r.stderr))
    v=json.loads(r.stdout);v.update(run_id=TAG,healthy=h['healthy'],RRCTL_state=inspect['status']['state'],run_spec_sha256=inspect['binding']['run_spec_sha256'])
    if v['source_commit']!=SOURCE:raise ValueError('New formal runtime identity mismatch')
    if v['progress'].get('at_batches',0)>0:
        for role in ('executor','queue'):
            q=v['identities'][role]
            if not q.get('alive') or not q.get('expected_start_matches') or q.get('uid')!=1014:raise ValueError('New formal controller identity mismatch')
        workers=v['identities']['scientific_GPU']
        if len(workers)!=1 or not workers[0].get('cwd_matches_release') or not workers[0].get('parent_matches_queue'):raise ValueError('New formal scientific GPU child identity/lineage mismatch')
        if v['progress']['actual_shape']!=[36,3,640,640] or v['progress']['actual_PGD_steps']!=10:raise ValueError('Formal native batch/PGD10 sink mismatch')
        expected=json.loads((HERE/'CURRENT_GPU_MAIN_AUDIT.json').read_text())['initial_sha256']
        if v['initial_model_sha256']!=expected or not all(v['AMP'].get(k) for k in ('model_finite','EMA_finite','optimizer_finite')):raise ValueError('Formal initialization/finite state gate failed')
    (HERE/'NEW_FORMAL_OBSERVATION.json').write_text(json.dumps(v,indent=2),encoding='utf8')
    print(json.dumps(v))
