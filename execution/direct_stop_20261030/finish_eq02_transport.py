"""Recover a completed computation's incorrect union-of-phase artifact adapter."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
from run_probe import save,record_state
ctl,t=services();d=HERE/'probes/eq02c4';s=json.loads((d/'runspec.json').read_text());tag=s['run_id']
v=ctl.inspect(tag);assert v['status']['state']=='workload_complete'
try:ctl._worker_lifecycle_command(tag,['complete'],phase='complete')
except Exception as e:
    err=dict(error=repr(e),details=getattr(e,'details',None));save(d/'original_complete_artifact_error.json',err)
    assert 'candidate_resume_summary.json' in json.dumps(err)
else:raise AssertionError('Original metadata mismatch was not reproduced')
FIX=r'''
import ast,hashlib,json,sys,zipfile
from pathlib import Path
b=json.loads(sys.argv[1]);zpath=Path(b['stage_root'])/'rrctl-worker.pyz'
assert hashlib.sha256(zpath.read_bytes()).hexdigest()==b['worker_sha256']
sys.path.insert(0,str(zpath));from remote_run_control import worker as w
control=Path(b['control_root']);spec=w._load_spec(control/'run_spec.json')
assert spec.run_id=='s4_at_eq02_clean4_20261030_direct01' and spec.source.commit=='dbdc05f938d1e553fbd62af32b2fa36f908c68d0'
assert w.read_status(control)['state']=='workload_complete'
assert spec.metadata['case']['kind']=='eq'
assert sorted(x.path for x in spec.artifacts)==['result/status.json','result/summary.json']
out=Path(spec.remote.output_root);status=json.loads((out/'result/status.json').read_text());summary=json.loads((out/'result/summary.json').read_text())
assert status['state']==summary['state']=='PASS_NATIVE_FIXED_INPUT_EQUALITY_COMPONENT'
assert summary['exact_step_and_state_hashes'] and summary['actual_native_outer_batches']==32 and summary['actual_native_SGD_per_side']>=1
with zipfile.ZipFile(zpath) as z:tree=ast.parse(z.read('remote_run_control/worker.py').decode())
f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='complete')
class PhaseArtifacts(ast.NodeTransformer):
 def __init__(self):self.changes=0
 def visit_Call(self,node):
  if isinstance(node.func,ast.Name) and node.func.id=='build_artifact_manifest':
   for k in node.keywords:
    if k.arg=='adapter_paths':k.value=ast.parse('tuple(item.path for item in spec.artifacts)',mode='eval').body;self.changes+=1
  return self.generic_visit(node)
fix=PhaseArtifacts();f=fix.visit(f);assert fix.changes==1;ast.fix_missing_locations(f)
scope=dict(vars(w));exec(compile(ast.Module(body=[f],type_ignores=[]),'<phase artifact union repair>','exec'),scope)
result=scope['complete'](control)
print(json.dumps(dict(state='TRANSPORT_COMPLETION_METADATA_REPAIRED',result=result,science_rerun=False,
 original_spec_binding_source_unchanged=True,required_scientific_status_and_summary_preserved=True,
 removed_inapplicable_adapter_union_path='result/candidate_resume_summary.json',
 summary_sha256=hashlib.sha256((out/'result/summary.json').read_bytes()).hexdigest())))
'''
r=t.run([PYTHON,'-B','-c',FIX,json.dumps(v['binding'])]);assert not r.returncode,repr((r.stdout,r.stderr))
save(d/'completion_metadata_repair.json',json.loads(r.stdout))
end=ctl.inspect(tag);assert end['status']['state']=='completed';save(d/'completed_inspect.json',end)
pull=ctl.pull(tag);save(d/'collect.json',pull)
raw=t.download(s['remote']['output_root']+'/result/summary.json');(d/'summary.json').write_bytes(raw)
summary=json.loads(raw);record_state(s,'eq02c4','COLLECTED',summary)
print(json.dumps(dict(state=summary['state'],native_outer_batches=summary['actual_native_outer_batches'],
 SGD_per_side=summary['actual_native_SGD_per_side'],AMP_skips_per_side=summary['actual_AMP_skips_per_side'],
 exact_step_and_state_hashes=summary['exact_step_and_state_hashes'],RRCTL_terminal=end['status']['state'],summary=str(d/'summary.json'))))
