"""Prepare/launch one finite AT CUDA preflight via the existing RRCTL stack."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from robustness_s4_at.common import ROOT,load,now,protocol,scientific_source,sha,write_once

EX=ROOT/'execution';BASE=ROOT.parent
PYTHON='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
PROFILES=BASE/'s4_robustness_evaluation_20261019/execution/profiles.json'
COCO=BASE/'s4_coco_minitrain25k_20261023'
READ=r'''
import datetime,json,os,shutil,subprocess,time,sys
from pathlib import Path
p=json.loads(sys.argv[1]);o=Path(p['existing_COCO_output']);s=json.loads((o/'pipeline_state.json').read_text());assert os.getuid()==1014
def live(v):
 if not v:return False
 f=Path('/proc')/str(v['pid'])
 if not f.exists():return False
 row=(f/'stat').read_text().rsplit(')',1)[1].split()
 return f.stat().st_uid==v['uid'] and int(row[19])==v['start_ticks'] and row[0]!='Z'
assert s['state']=='COMPLETED_AND_ACCEPTED' and all(v['state']=='ACCEPTED' for v in s['stages'].values())
assert s['stages']['final_report']['receipt_sha256']=='016f4889cc1375ee970259dbdb4a68560ce367dfc1a13e346931b0a0f0ca6d89'
blockers=[]
if live(s.get('controller')):blockers.append('COCO_CONTROLLER_STILL_LIVE')
if live(s.get('active_child')):blockers.append('COCO_CHILD_STILL_LIVE')
mem=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))/2**20
owned=Path(p['remote_owned_root']);disk=shutil.disk_usage(owned if owned.exists() else owned.parent).free/2**30
def ticks():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
io=[]
for _ in range(3):
 a=ticks();time.sleep(1);b=ticks();d=[y-x for x,y in zip(a,b)];io.append(100*d[4]/max(1,sum(d)))
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
free={x.split(',')[0].strip():int(x.split(',')[1]) for x in g.splitlines()}['0']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20);own=[]
for value in apps.splitlines():
 if value.strip():
  pid=int(value.strip())
  try:
   if (Path('/proc')/str(pid)).stat().st_uid==1014:own.append(pid)
  except FileNotFoundError:pass
if own:blockers.append('OWN_GPU_ALREADY_ACTIVE')
if mem<32:blockers.append('RAM_BELOW_32GiB')
if disk<150:blockers.append('DISK_BELOW_150GiB')
if free<24000:blockers.append('GPU_FREE_BELOW_24000MiB')
if all(x>=20 for x in io):blockers.append('SUSTAINED_IOWAIT_THREE_WINDOWS')
print(json.dumps(dict(admitted=not blockers,blockers=blockers,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),COCO_completed=True,COCO_controller_alive=live(s['controller']),own_GPU_pids=own,RAM_available_GiB=mem,disk_free_GiB=disk,GPU_free_MiB=free,IOwait_percent=io)))
'''

def services():
 from remote_run_control.controller import Controller
 from remote_run_control.profiles import ProfileStore
 from remote_run_control.transport import transport_for
 return Controller(profiles_path=PROFILES,state_root=EX/'rrctl_state'),transport_for(ProfileStore(PROFILES).load('school-s1'))

def current():
 c=scientific_source()
 if c['uncommitted_files']:raise ValueError('AT computation has uncommitted source changes')
 for path,digest in protocol()['original_sources'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Original S4 source changed: '+path)
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/s4-at-robustness-20261027'],cwd=ROOT,text=True).split()
 if not remote or remote[0]!=c['commit']:raise ValueError('Current AT commit is not published and remotely verified')
 return c['commit']

def admission(transport):
 proof=load(COCO/'execution/COCO_FINAL_PUBLICATION_VERIFIED_20261029.json')
 if proof.get('state')!='passed':raise ValueError('COCO final publication is not verified')
 r=transport.run([PYTHON,'-B','-c',READ,json.dumps(protocol())])
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3000:])
 return json.loads(r.stdout)

def prepare(tag):
 if (EX/(tag+'.json')).exists() or (EX/tag).exists() or (EX/'rrctl_state'/tag).exists():raise FileExistsError('Existing AT candidate is inspect-only')
 head=current();_,t=services();a=admission(t)
 if not a['admitted']:print(json.dumps(a));return
 p=protocol();owned=p['remote_owned_root'];s=load(COCO/'execution/coco_official_eval_20261029_gpu03.json')
 s.update(run_id=tag,project='original-s4-voc20-at-cuda-preflight')
 s['source']=dict(repo_root=str(ROOT),branch='codex/s4-at-robustness-20261027',commit=head,allowed_post_commit_paths=['execution/','docs/S4_AT_ROBUSTNESS_20261027/'])
 s['remote'].update(stage_root=owned+'/stage/'+tag,repo_root=owned+'/releases/'+tag,control_root=owned+'/control/'+tag,output_root=owned+'/validation/'+tag)
 s['session']['name']=tag;s['anchors']=[]
 unit='s4at-cuda-'+hashlib.sha256(tag.encode()).hexdigest()[:20]
 s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit='+unit,PYTHON,'-B','-m','robustness_s4_at.validate_cuda','--out',s['remote']['output_root']+'/result','--source-commit',head])
 adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress={k:s.get(k) for k in ('state','source_commit','current_run','completed_bounded_arms','actual_outer_batches','error')},artifacts=['result/status.json','result/smoke_summary.json','result/GT_and_budget.json'])))"
 for h in s['health'].values():h['adapter_argv']=[PYTHON,'-B','-c',adapter];h['gpu_utilization_policy']='advisory';h['timeout_seconds']=900
 s['artifacts']=[dict(path='result/'+name,required=True) for name in ('smoke_summary.json','status.json','GT_and_budget.json')]
 s['local_pull_root']=str(EX/'p')
 s['metadata']=dict(execution_purpose='finite_original_S4_AT_native_CUDA_preflight',candidate_commit=head,computation_kind='training',max_total_native_outer_batches=96,PGD_steps_per_batch=10,formal_queue_launched=False,official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',no_second_long_controller=True)
 from remote_run_control.models import RunSpec
 RunSpec.from_dict(s);write_once(EX/(tag+'.json'),s);write_once(EX/tag/'admission_prepare.json',a)
 print(json.dumps(dict(prepared=tag,source_commit=head,launched=False,admission=a)))

def launch(tag):
 if (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():raise FileExistsError('Bound/intended AT candidate is inspect-only')
 s=load(EX/(tag+'.json'))
 if current()!=s['source']['commit']:raise ValueError('AT candidate source changed')
 ctl,t=services();a=admission(t);write_once(EX/tag/'admission_launch.json',a)
 if not a['admitted']:print(json.dumps(a));return
 paths=[s['remote'][key] for key in ('stage_root','repo_root','control_root','output_root')]
 code="import json,sys;from pathlib import Path;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print('all_absent')"
 r=t.run([PYTHON,'-B','-c',code,json.dumps(paths)])
 if r.returncode:raise RuntimeError('Remote AT candidate collision; no launch')
 from remote_run_control.models import RunSpec
 spec=RunSpec.from_path(EX/(tag+'.json'));ready=ctl.ready(spec);write_once(EX/tag/'rrctl_ready.json',ready)
 if not ready.get('ready'):raise ValueError('RRCTL preflight readiness failed')
 temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
 write_once(EX/tag/'launch_intent.json',dict(UTC=now(),source_commit=s['source']['commit']))
 result=ctl.launch(spec);write_once(EX/tag/'launch.json',result)
 print(json.dumps(dict(launched_once=True,run_id=tag,result=result,formal_AT_started=False)))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=('prepare','launch'));a.add_argument('--tag',required=True);v=a.parse_args()
 if not re.fullmatch('s4_at_cuda_[a-z0-9_]{1,48}',v.tag):raise ValueError('Unique AT CUDA candidate tag required')
 {'prepare':prepare,'launch':launch}[v.action](v.tag)
