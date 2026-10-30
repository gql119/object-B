"""Explicit human-authorized termination through the existing RRCTL abort API."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON

TAG='s4_at_serial_seed0_20261029_v2'
SOURCE='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb'
SPEC='e9f80f82d4288a9b08ed9877b900911f813854e390479eb6a453176c907c87b3'
AUTH=ROOT/'docs/S4_AT_ROBUSTNESS_20261027/USER_DIRECT_TERMINATION_AND_RESTART_20261030.md'
READ=r'''
import ast,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
b=json.loads(sys.argv[1]);assert os.getuid()==1014
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def proc(pid):
 d=Path('/proc')/str(pid);r=(d/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,uid=d.stat().st_uid,ppid=int(r[1]),state=r[0],start_ticks=int(r[19]),
  pgid=os.getpgid(pid),argv=(d/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
allp={}
for d in Path('/proc').iterdir():
 if d.name.isdigit():
  try:allp[int(d.name)]=proc(int(d.name))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
ex=allp[b['executor_pid']];q=allp[b['workload_pid']]
assert ex['start_ticks']==b['executor_start_ticks'] and q['start_ticks']==b['workload_start_ticks']
assert ex['uid']==q['uid']==1014 and ex['pgid']==q['pgid']==b['workload_process_group_id']
assert 'robustness_s4_at.queue' in q['argv'] and b['run_id'] in ' '.join(q['argv'])
desc={ex['pid']};changed=True
while changed:
 changed=False
 for pid,v in allp.items():
  if v['ppid'] in desc and pid not in desc:desc.add(pid);changed=True
group=[v for v in allp.values() if v['pgid']==q['pgid']]
assert all(v['uid']==1014 and v['pid'] in desc for v in group)
children=[v for v in group if v['ppid']==q['pid'] and 'robustness_s4_at.worker' in v['argv']]
assert len(children)==1 and b['commit'] in children[0]['argv']
out=Path(b['output_root'])/'result';pipeline=json.loads((out/'pipeline_state.json').read_text())
assert pipeline['state']=='RUNNING' and pipeline['current_run']=='S4AT4_Clean_seed0' and not pipeline['stages']
arm=out/pipeline['current_run'];p=json.loads((arm/'audit/progress.json').read_text())
epoch=p['completed_epochs'];cp=arm/'training/full_state'/f'epoch_{epoch:03d}.pt'
st=cp.stat();digest=sha(cp);assert (st.st_size,st.st_mtime_ns)==(cp.stat().st_size,cp.stat().st_mtime_ns)
zip_path=Path(b['stage_root'])/'rrctl-worker.pyz';assert sha(zip_path)==b['worker_sha256']
def fhash(raw,name):
 t=ast.parse(raw);f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
 return hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()
with zipfile.ZipFile(zip_path) as z:
 code=z.read('remote_run_control/worker.py').decode()
 cleanup=z.read('remote_run_control/cleanup.py').decode()
 api_hashes=dict(abort=fhash(code,'abort'),cleanup_output=fhash(cleanup,'cleanup_output'))
 def fsource(raw,name):
  t=ast.parse(raw);f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
  return ast.get_source_segment(raw,f)
 api_sources=dict(abort=fsource(code,'abort'),cleanup_output=fsource(cleanup,'cleanup_output'))
spec=json.loads((Path(b['control_root'])/'run_spec.json').read_text())
assert spec.get('output_cleanup') is None
logs=[]
for path in (Path(b['output_root'])/'console.log',Path(b['control_root'])/'console.log',out/'S4AT4_Clean_seed0.console.log'):
 if path.is_file():logs.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
gpu=[]
for line in apps.splitlines():
 if not line.strip():continue
 pid,mem=line.split(',');pid=int(pid);v=allp.get(pid)
 gpu.append(dict(pid=pid,used_mib=int(mem),uid=v['uid'] if v else None,owned_run=pid in desc))
assert all(v['owned_run'] for v in gpu if v['uid']==1014),'Independent same-UID GPU job exists'
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pipeline=pipeline,progress=p,
 executor=ex,supervisor=q,scientific_child=children[0],process_group=group,all_group_members_owned_and_descendants=True,
 checkpoint=dict(path=str(cp),sha256=digest,bytes=st.st_size),retained_checkpoint_files=len(list((arm/'training/full_state').glob('epoch_*.pt'))),
 logs=logs,GPU_processes=gpu,bound_worker_sha256=sha(zip_path),abort_implementation_AST=api_hashes,bound_abort_sources=api_sources,
 output_cleanup=None,retention='No checkpoint deletion path; normal full-group abort targets queue and scientific descendants together',signals_sent=0)))
'''
VERIFY=r'''
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.loads(sys.argv[1]);before=p['before'];alive=[]
for v in before['process_group']:
 d=Path('/proc')/str(v['pid'])
 try:
  r=(d/'stat').read_text().rsplit(')',1)[1].split()
  if int(r[19])==v['start_ticks'] and r[0]!='Z':alive.append(v['pid'])
 except FileNotFoundError:pass
cp=Path(before['checkpoint']['path']);h=hashlib.sha256()
with cp.open('rb') as f:
 for z in iter(lambda:f.read(1048576),b''):h.update(z)
assert h.hexdigest()==before['checkpoint']['sha256']
logs=[dict(path=v['path'],exists=Path(v['path']).is_file(),bytes=Path(v['path']).stat().st_size) for v in before['logs']]
assert all(v['exists'] for v in logs)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
own=[]
for raw in apps.splitlines():
 if not raw.strip():continue
 pid=int(raw)
 try:
  if (Path('/proc')/str(pid)).stat().st_uid==1014:own.append(pid)
 except FileNotFoundError:pass
out=Path(p['binding']['output_root'])/'result'
not_started={n:not (out/n).exists() for n in ('S4AT4_Poison_seed0','S4AT8_Clean_seed0','S4AT8_Poison_seed0')}
assert not alive and not own and all(not_started.values())
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_live_processes=alive,
 own_GPU_pids=own,checkpoint_SHA_unchanged=True,logs_retained=logs,next_arms_not_started=not_started,
 old_science_completed=False,old_science_accepted=False,deleted_paths=[])))
'''

def dump(name,v):
    path=HERE/name
    with path.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
    return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inspect','terminate']);a=ap.parse_args()
    assert AUTH.is_file()
    ctl,t=services();s=ctl.inspect(TAG);b=s['binding']
    assert b['commit']==SOURCE and b['run_spec_sha256']==SPEC
    assert s['status']['state']=='running'
    r=t.run([PYTHON,'-B','-c',READ,json.dumps(b)])
    if r.returncode:raise RuntimeError(repr((r.stdout,r.stderr)))
    before=json.loads(r.stdout)
    from remote_run_control import worker,cleanup
    def local_fhash(module,name):
        raw=Path(module.__file__).read_text(encoding='utf8');tree=ast.parse(raw)
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
        return hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record=dict(binding=b,before=before,authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest())
    pre=dump(stamp+'_before_stop.json',record)
    local_hashes=dict(abort=local_fhash(worker,'abort'),cleanup_output=local_fhash(cleanup,'cleanup_output'))
    # Parse both sources with this interpreter. Python3.12+ adds AST fields
    # (e.g. FunctionDef.type_params) absent from the server's Python3.11.
    bound_reparsed={}
    for name,raw in before['bound_abort_sources'].items():
        f=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==name)
        bound_reparsed[name]=hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()
    dump(stamp+'_bound_source_closure.json',dict(local_AST=local_hashes,bound_source_parsed_locally=bound_reparsed,
         cross_interpreter_AST_digest_not_used=True,equal=bound_reparsed==local_hashes))
    if a.action=='inspect':
        print(json.dumps(dict(evidence=str(pre),progress={k:v for k,v in before['progress'].items() if k!='latest_PGD'},group_size=len(before['process_group']),GPU=before['GPU_processes'],checkpoint=before['checkpoint'],bound_AST=before['abort_implementation_AST'],local_AST=local_hashes,can_use_existing_full_group_abort='PENDING_BOUND_SOURCE_REVIEW')))
        return
    assert bound_reparsed==local_hashes,'Bound abort/retention source differs under a common AST parser'
    dump(stamp+'_termination_intent.json',dict(run_id=TAG,authorization_sha256=record['authorization_sha256'],pre_stop_evidence=str(pre),action='existing Controller.abort confirmed=true; terminate entire verified queue workload process group; retain all outputs'))
    result=ctl.abort(TAG,confirmed=True)
    dump(stamp+'_RRCTL_TERMINATION_RECEIPT.json',result)
    after_inspect=ctl.inspect(TAG)
    r=t.run([PYTHON,'-B','-c',VERIFY,json.dumps(record)])
    if r.returncode:raise RuntimeError(repr((r.stdout,r.stderr)))
    after=json.loads(r.stdout);assert after_inspect['status']['state']=='aborted'
    final=dict(state='OLD_AT_TERMINATED',RunID=TAG,before_evidence=str(pre),RRCTL=result,verified=after,binding=b,
               old_progress_abandoned_by_human=True,old_completed=False,old_accepted=False)
    path=dump(stamp+'_OLD_AT_TERMINATED.json',final)
    print(json.dumps(dict(evidence=str(path),**final)))

if __name__=='__main__':main()
