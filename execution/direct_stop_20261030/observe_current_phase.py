"""One read-only bounded progress snapshot, no model/GPU initialization."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
tag='s4_at_selected_phase_20261030_gpu01';s=json.loads((HERE/'current_probes'/tag/'runspec.json').read_text())
ctl,t=services()
code="from pathlib import Path;import json,sys;p=Path(sys.argv[1]);v={};\nfor rel in ['status.json','same_run/audit/progress.json','same_run/resume_audit/progress.json']:\n f=p/rel\n if f.exists():\n  r=json.loads(f.read_text());v[rel]={k:r.get(k) for k in ['state','epoch','completed_epochs','at_batches','actual_updates','legal_amp_skips','loss','UTC','error']}\nprint(json.dumps(v))"
r=t.run([PYTHON,'-B','-c',code,s['remote']['output_root']+'/result']);print(r.stdout.decode() if isinstance(r.stdout,bytes) else r.stdout)
