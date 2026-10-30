"""Append a new abandonment annotation, preserving the old pipeline snapshot."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
p=sorted(HERE.glob('*_OLD_AT_TERMINATED.json'))[-1];v=json.loads(p.read_text())
ctl,t=services();old=ctl.inspect(v['run_id']);assert old['status']['state']=='aborted'
annotation=dict(schema='s4.at.human-abandonment.v1',state='ABORTED_BY_EXPLICIT_HUMAN_AUTHORIZATION',
 old_run_id=v['run_id'],completed=False,accepted=False,old_checkpoint_continuation_forbidden=True,
 abandoned_completed_epochs=86,checkpoint=v['checkpoint_retained'],RRCTL_terminal=old['status'],
 authorization_sha256='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8',
 next='Finite CUDA optimization then uniformly fresh four-arm epoch0 restart; chosen generation pending',
 original_pipeline_state_is_retained_pretermination_snapshot=True,original_binding_and_locks_preserved=True)
local=HERE/'HUMAN_ABANDONMENT_ANNOTATION.json'
with local.open('x',encoding='utf8') as f:json.dump(annotation,f,indent=2)
remote=old['binding']['control_root']+'/HUMAN_ABANDONMENT_20261030.json'
r=t.run([PYTHON,'-B','-c',"from pathlib import Path;import sys;p=Path(sys.argv[1]);assert p.parent.is_dir() and not p.exists()",remote]);assert not r.returncode
digest=hashlib.sha256(local.read_bytes()).hexdigest();t.upload(local,remote,digest);assert t.download(remote)==local.read_bytes()
proof=dict(state='APPENDED_AND_EXACT_READBACK',path=remote,sha256=digest,old_snapshot_not_overwritten=True)
with (HERE/'HUMAN_ABANDONMENT_ANNOTATION_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
