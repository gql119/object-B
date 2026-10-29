"""Ownership-checked single-supervisor hold/release, never a training stop.

The public entry below runs an isolated CPU RRCTL lifecycle smoke only. The
original AT supervisor is not signaled by this program.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from at_cuda_rrctl_20261029 import EX, PYTHON, services
from remote_run_control.models import RunSpec

HERE = Path(__file__).resolve().parent
TAG = "s4_at_queue_window_20261030_cpu04"

CONTROL = r'''
import ctypes,datetime,hashlib,json,os,platform,re,signal,sys,time
from pathlib import Path
q=json.loads(sys.argv[1]);sent=False
def info(pid):
 d=Path('/proc',str(pid));raw=(d/'cmdline').read_bytes();s=(d/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,uid=d.stat().st_uid,ppid=int(s[1]),state=s[0],start_ticks=int(s[19]),argv=[v for v in raw.decode().split(chr(0)) if v],cwd=os.readlink(d/'cwd'),wchan=(d/'wchan').read_text().strip())
try:
 scope=q.get('scope')
 if scope=='isolated_cpu_queue_control':
  if not q['run_id'].startswith('s4_at_queue_window_'):raise ValueError('Not the bound CPU lifecycle fixture')
 elif scope=='original_AT_queue_supervisor':
  if q['run_id']!='s4_at_serial_seed0_20261029_v2':raise ValueError('Wrong original AT binding')
  expected_control='/project/guoqinglong/projects/s4_at_robustness_20261027/control/'+q['run_id']
  if q.get('control_root')!=expected_control:raise ValueError('Wrong original control root')
  binding=json.loads((Path(expected_control)/'binding.json').read_text())
  if binding.get('commit')!='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb':raise ValueError('Unexpected original source')
  if q['pid']!=binding['workload_pid'] or q['start_ticks']!=binding['workload_start_ticks'] or q['cwd']!=binding['repo_root']:raise ValueError('Original supervisor differs from RRCTL binding')
  out=Path(binding['output_root'])/'result';pipeline=json.loads((out/'pipeline_state.json').read_text())
  if pipeline.get('current_run')!='S4AT4_Clean_seed0' or pipeline.get('state')!='RUNNING':raise ValueError('Natural-boundary plan is only for the original CleanAT4 arm')
  progress=json.loads((out/'S4AT4_Clean_seed0/audit/progress.json').read_text())
  if q['argv'][1:4]!=['-B','-m','robustness_s4_at.queue']:raise ValueError('Target is not the queue supervisor module')
  if q['action']=='assess':
   print(json.dumps(dict(eligible=True,signal_sent=False,scope=scope,completed_epochs=progress['completed_epochs'],eligible_for_hold=195<=progress['completed_epochs']<200,minimum_completed_epochs=195,source=binding['commit'])));sys.exit(0)
  if q['action']=='hold' and not 195<=progress['completed_epochs']<200:raise ValueError('Hold is restricted to the natural end-of-arm reservation window, completed epochs195..199')
 else:raise ValueError('Unrecognized control scope')
 if os.getuid()!=1014 or q['uid']!=1014:raise ValueError('UID mismatch')
 p=info(q['pid'])
 for k in ('uid','start_ticks','argv','cwd'):
  if p[k]!=q[k]:raise ValueError('Identity mismatch: '+k)
 if p['state']=='Z':raise ValueError('Target is a zombie')
 if q['pid']==q['child_pid']:raise ValueError('Supervisor and child must differ')
 try:child=info(q['child_pid'])
 except FileNotFoundError:
  if q['action']!='release':raise
  child={'state':'exited'}
 if child['state']!='exited' and (child['uid']!=1014 or child['ppid']!=p['pid'] or child['start_ticks']!=q['child_start_ticks']):raise ValueError('Child ownership or lineage mismatch')
 if q['action']=='hold' and p['wchan']!='do_wait':raise ValueError('Supervisor is not waiting for its child')
 if q['action']=='hold' and child['state'] in ('T','t','Z'):raise ValueError('Child is not actively independent')
 if q['action']=='release' and p['state'] not in ('T','t'):raise ValueError('Supervisor is not held')
 if scope=='original_AT_queue_supervisor' and q['action']=='release':
  import subprocess
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
  for raw in apps.splitlines():
   if not raw.strip():continue
   apid=int(raw.strip())
   try:
    if Path('/proc',str(apid)).stat().st_uid==1014 and apid!=q['child_pid']:raise ValueError('An owned validation GPU process remains; do not release the next formal arm')
   except FileNotFoundError:pass
 if q['action'] not in ('inspect','hold','release'):raise ValueError('Unknown action')
 if q['action']=='inspect':
  print(json.dumps(dict(eligible=True,signal_sent=False,supervisor=p,child=child)));sys.exit(0)
 if platform.machine()!='x86_64':raise ValueError('Unvalidated syscall architecture')
 header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text();nums={}
 for line in header.splitlines():
  m=re.match(r'#define\s+__NR_(pidfd_open|pidfd_send_signal)\s+(\d+)\s*$',line)
  if m:nums[m.group(1)]=int(m.group(2))
 if nums!={'pidfd_send_signal':424,'pidfd_open':434}:raise ValueError('Unexpected Linux pidfd syscall interface')
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 fd=libc.syscall(ctypes.c_long(nums['pidfd_open']),ctypes.c_int(p['pid']),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open failed')
 try:
  # Revalidate after opening a stable kernel process handle; no PID reuse fallback.
  current=info(p['pid'])
  if current['start_ticks']!=q['start_ticks'] or current['uid']!=1014 or current['argv']!=q['argv']:raise ValueError('Identity changed after pidfd_open')
  sig=signal.SIGSTOP if q['action']=='hold' else signal.SIGCONT
  rc=libc.syscall(ctypes.c_long(nums['pidfd_send_signal']),ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
  if rc<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal failed')
  sent=True
 finally:os.close(fd)
 deadline=time.monotonic()+2;state=None
 while time.monotonic()<deadline:
  try:state=info(p['pid'])['state']
  except FileNotFoundError:state='exited'
  if (q['action']=='hold' and state in ('T','t')) or (q['action']=='release' and state not in ('T','t')):break
  time.sleep(.02)
 print(json.dumps(dict(eligible=True,signal_sent=sent,action=q['action'],pid=p['pid'],start_ticks=p['start_ticks'],state_after=state,single_pid_only=True,process_group_signals=False,TERM_or_KILL_used=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
except Exception as exc:
 print(json.dumps(dict(eligible=False,signal_sent=sent,error=repr(exc))))
 sys.exit(2)
'''

CHILD = r'''
import datetime,json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]);start=time.monotonic();count=0
while time.monotonic()-start<30:
 count+=1
 q=p/'child_progress.writing';q.write_text(json.dumps(dict(pid=os.getpid(),ticks=count,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));q.replace(p/'child_progress.json')
 time.sleep(.2)
(p/'child_complete.json').write_text(json.dumps(dict(CPU_control_fixture_only=True,scientific_model_forwards=0,GPU_contexts=0,ticks=count)))
'''

SUPERVISOR = r'''
import datetime,json,os,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'result';p.mkdir(parents=True,exist_ok=False)
def proc(pid):
 d=Path('/proc',str(pid));s=(d/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,uid=d.stat().st_uid,start_ticks=int(s[19]),argv=[x for x in (d/'cmdline').read_bytes().decode().split(chr(0)) if x],cwd=os.readlink(d/'cwd'))
child_code=CHILD_CODE
child=subprocess.Popen([sys.executable,'-B','-c',child_code,str(p)])
manifest=proc(os.getpid());manifest.update(child_pid=child.pid,child_start_ticks=proc(child.pid)['start_ticks'],role='isolated_cpu_queue_supervisor')
(p/'supervisor.json').write_text(json.dumps(manifest))
(p/'status.json').write_text(json.dumps(dict(state='CPU_CHILD_RUNNING',GPU_contexts=0,model_forwards=0)))
code=child.wait()
if code:raise RuntimeError('CPU fixture child failed')
assert (p/'child_complete.json').exists()
(p/'stage2_started.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
(p/'CPU_CONTROL_SMOKE.json').write_text(json.dumps(dict(state='PASSED',child_completed_naturally=True,supervisor_PID=os.getpid(),scientific_model_forwards=0,GPU_contexts=0,formal_AT_touched=False)))
(p/'status.json').write_text(json.dumps(dict(state='PASSED',GPU_contexts=0,model_forwards=0)))
'''.replace("CHILD_CODE", repr(CHILD))

ADAPTER = """import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='PASSED',progress=s,artifacts=['result/CPU_CONTROL_SMOKE.json','result/status.json'])))"""


def write_once(p, value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)


def run_smoke():
    directory=HERE/TAG
    if directory.exists():raise FileExistsError('Existing CPU control RunID is inspect-only')
    directory.mkdir()
    s=json.loads((EX/'s4_at_serial_seed0_20261029_v2.json').read_text())
    owned='/project/guoqinglong/projects/s4_at_robustness_20261027'
    s.update(run_id=TAG,project='s4-at-queue-boundary-control-CPU-smoke')
    for k,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control')):s['remote'][k]=owned+'/'+folder+'/'+TAG
    s['remote']['output_root']=owned+'/validation/'+TAG
    s['session']['name']=TAG;s['anchors']=[]
    s['workload']=dict(cwd='.',argv=[PYTHON,'-B','-c',SUPERVISOR,s['remote']['output_root']])
    for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',ADAPTER],gpu_utilization_policy='advisory',timeout_seconds=120)
    s['artifacts']=[dict(path='result/'+n,required=True) for n in ('CPU_CONTROL_SMOKE.json','status.json','supervisor.json','child_complete.json','stage2_started.json')]
    s['local_pull_root']=str(HERE/'pull')
    s['metadata']=dict(execution_purpose='bounded_CPU_process_supervision_smoke_only',scientific_model_forwards=0,GPU_work=False,formal_experiment=False,official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),no_signal_to_formal_AT=True)
    spec=RunSpec.from_dict(s);write_once(directory/'runspec.json',s)
    ctl,t=services();ready=ctl.ready(spec);write_once(directory/'ready.json',ready)
    if not ready.get('ready'):raise RuntimeError('CPU control launcher readiness failed')
    launch=ctl.launch(spec);write_once(directory/'launch.json',launch)
    root=s['remote']['output_root']+'/result'
    def read(name):
        return json.loads(t.download(root+'/'+name))
    manifest=read('supervisor.json')
    inspected=ctl.inspect(TAG);write_once(directory/'inspect.json',inspected)
    binding=inspected['binding']
    if manifest['pid']!=binding['workload_pid'] or manifest['start_ticks']!=binding['workload_start_ticks']:
        raise ValueError('CPU supervisor does not match its exact RRCTL workload binding')
    q={k:manifest[k] for k in ('pid','uid','start_ticks','argv','cwd','child_pid','child_start_ticks')}
    q.update(scope='isolated_cpu_queue_control',run_id=TAG)
    def control(action, overrides=None, label=None):
        req=dict(q,action=action);req.update(overrides or {})
        r=t.run([PYTHON,'-B','-c',CONTROL,json.dumps(req)])
        value=json.loads(r.stdout)
        write_once(directory/(label or action+'.json'),value)
        return r.returncode,value
    rejections=[]
    for key,value in (('uid',0),('start_ticks',manifest['start_ticks']+1),('argv',['wrong']),('pid',manifest['child_pid'])):
        rc,v=control('hold',{key:value},'reject_'+key+'.json')
        if rc!=2 or v.get('signal_sent'):raise AssertionError('Identity guard failed before signal')
        rejections.append(key)
    before=read('child_progress.json');held=False
    try:
        rc,h=control('hold')
        if rc or h.get('state_after') not in ('T','t'):raise RuntimeError('CPU supervisor hold failed')
        held=True
        time.sleep(1)
        after=read('child_progress.json')
        if after['ticks']<=before['ticks']:raise AssertionError('Holding supervisor stopped child progress')
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            r=t.run([PYTHON,'-B','-c',"import json,sys;from pathlib import Path;p=Path(sys.argv[1]);print(json.dumps(dict(child_done=(p/'child_complete.json').exists(),stage2=(p/'stage2_started.json').exists())))",root])
            state=json.loads(r.stdout)
            if state['stage2']:raise AssertionError('Stage2 started while supervisor held')
            if state['child_done']:break
            time.sleep(1)
        else:raise TimeoutError('Bounded CPU child did not finish')
        write_once(directory/'held_child_completion.json',dict(before=before,after=after,child=read('child_complete.json'),stage2_absent=True,ownership_rejections=rejections))
    finally:
        if held:
            rc,v=control('release')
            if rc:raise RuntimeError('CPU supervisor release requires attention: '+repr(v))
    health=ctl.health(TAG,phase='completion');write_once(directory/'completion_health.json',health)
    terminal=ctl.wait(TAG,poll_seconds=1);write_once(directory/'completion_wait.json',terminal)
    pulled=ctl.pull(TAG);write_once(directory/'completion_pull.json',pulled)
    result=dict(state='PASS_REAL_RRCTL_CPU_QUEUE_BOUNDARY_SMOKE',RunID=TAG,identity_rejections=rejections,
        supervisor_hold_did_not_stop_child=True,child_finished_naturally_before_stage2=True,
        same_supervisor_released=True,GPU_contexts=0,scientific_model_forwards=0,
        original_AT_signals_sent=0,TERM_or_KILL_used=False,original_AT_control_ready=False,
        control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),
        rrctl_worker_sha256=binding['worker_sha256'],
        note='Original AT remains untouched. Production policy and near-boundary identity checks still required.')
    write_once(directory/'MAIN_CONTROL_CLOSURE.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-smoke',action='store_true',required=True)
    parser.parse_args();run_smoke()
