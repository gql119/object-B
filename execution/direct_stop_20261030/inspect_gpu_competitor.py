import json,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
_,t=services()
CODE=r'''
import datetime,json,os,subprocess
from pathlib import Path
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
 if not line.strip():continue
 pid,mem=line.split(',');pid=int(pid);d=Path('/proc')/str(pid)
 try:
  uid=d.stat().st_uid;r=(d/'stat').read_text().rsplit(')',1)[1].split()
  v=dict(pid=pid,uid=uid,used_memory_MiB=int(mem),ppid=int(r[1]),state=r[0],start_ticks=int(r[19]))
  if uid==1014:
   v['argv']=(d/'cmdline').read_bytes().decode().rstrip('\0').split('\0');v['cwd']=os.readlink(d/'cwd')
   v['RRCTL_RUN_ID']=[x.decode() for x in (d/'environ').read_bytes().split(b'\0') if x.startswith(b'RRCTL_RUN_ID=')]
  rows.append(v)
 except (PermissionError,FileNotFoundError):rows.append(dict(pid=pid,metadata_unavailable=True))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_processes=rows,signals_sent=0)))
'''
r=t.run([PYTHON,'-B','-c',CODE]);assert not r.returncode
value=json.loads(r.stdout);p=HERE/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_GPU_COMPETITION.json')
p.write_text(json.dumps(value,indent=2),encoding='utf8');print(json.dumps(dict(evidence=str(p),**value)))
