"""Recover only the post-stop read-only check after local argv length failure."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
from stop_old import VERIFY
from guarded_abort_repair import CONTROL
path=sorted(HERE.glob('*_REPAIRED_ABORT_RESULT.json'))[-1]
raw=json.loads(path.read_text());assert not raw['returncode']
assert json.loads(raw['stdout'])['result']['state']=='aborted'
prefix=path.name.split('_REPAIRED_ABORT_RESULT')[0]
pre=HERE/(prefix+'_REPAIRED_ABORT_PRESTATE.json');record=json.loads(pre.read_text())
b=record['binding'];before=record['before'];ctl,t=services()
terminal=ctl.inspect(b['run_id']);assert terminal['status']['state']=='aborted'
compact=dict(binding=b,before=dict(process_group=[dict(pid=v['pid'],start_ticks=v['start_ticks']) for v in before['process_group']],checkpoint=before['checkpoint'],logs=before['logs']))
r=t.run([PYTHON,'-B','-c',VERIFY,json.dumps(compact)])
assert not r.returncode,repr((r.stdout,r.stderr))
after=json.loads(r.stdout)
final=dict(state='OLD_AT_TERMINATED',UTC=datetime.now(timezone.utc).isoformat(),run_id=b['run_id'],
 old_completed=False,old_accepted=False,old_progress_abandoned_by_human=True,prestate=str(pre),
 RRCTL=terminal['status'],verification=after,control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),
 binding_unmodified=True,checkpoint_retained=before['checkpoint'],
 postcheck_transport_recovery='First postcheck exceeded Windows argv limit before remote execution; minimal identity/checkpoint/log payload used. Abort was not repeated.')
out=HERE/(prefix+'_OLD_AT_TERMINATED.json')
with out.open('x',encoding='utf8') as f:json.dump(final,f,indent=2)
print(json.dumps(dict(evidence=str(out),**final)))
