import json,sys,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
ctl,t=services();d=HERE/'probes/bo1';s=json.loads((d/'runspec.json').read_text())
raw=t.download(s['remote']['output_root']+'/result/bounded_telemetry.json')
(d/'FAILED_BOUNDED_TELEMETRY.json').write_bytes(raw)
rows=json.loads(raw);errors=[(i,v.get('sampling_error')) for i,v in enumerate(rows) if v.get('sampling_error')]
other=[(i,v['other_GPU_pids']) for i,v in enumerate(rows) if v.get('other_GPU_pids')]
r=t.run([PYTHON,'-B','-c',"import subprocess,json;print(json.dumps(dict(GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True))))"])
summary=dict(samples=len(rows),sampling_errors=errors,other_GPU_rows=other,current_GPU=json.loads(r.stdout),first=rows[:2],last=rows[-2:])
with (d/'TELEMETRY_FAILURE_DIAGNOSIS.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
