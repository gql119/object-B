"""Scoped RRCTL post-systemd-exec identity repair; original bindings stay immutable."""
import hashlib
import json

OLD_TAG='s4_at_serial_seed0_20261029_v2'
CPU_TAG='s4_at_abort_exec_guard_20261030_cpu02'
AUTH_SHA='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8'

CONTROL=r'''
import ast,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
q=json.loads(sys.argv[1]);assert os.getuid()==q['uid']==1014
OLD='s4_at_serial_seed0_20261029_v2';CPU='s4_at_abort_exec_guard_20261030_cpu02'
assert q['run_id'] in (OLD,CPU) and q['action'] in ('inspect','abort')
assert q['authorization_sha256']=='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8'
owned=Path('/project/guoqinglong/projects/s4_at_robustness_20261027')
control=owned/'control'/q['run_id'];binding_path=control/'binding.json';binding_bytes=binding_path.read_bytes()
assert hashlib.sha256(binding_bytes).hexdigest()==q['binding_sha256']
b=json.loads(binding_bytes);zip_path=Path(b['stage_root'])/'rrctl-worker.pyz'
assert hashlib.sha256(zip_path.read_bytes()).hexdigest()==b['worker_sha256']
sys.path.insert(0,str(zip_path))
from remote_run_control import worker as w
spec=w._load_spec(control/'run_spec.json')
assert spec.run_id==q['run_id'] and spec.digest==q['spec_sha256'] and b['run_spec_sha256']==spec.digest
if q['run_id']==OLD:
 assert spec.digest=='e9f80f82d4288a9b08ed9877b900911f813854e390479eb6a453176c907c87b3'
 assert b['commit']=='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb'
 assert b['worker_sha256']=='696cb314eb11784a9099728b117eac69cd2e01315d7732154d1ddeab609eff6a'
 assert not q.get('force_bootstrap_fixture',False)
else:
 assert spec.project=='s4-at-guarded-abort-CPU-smoke'
 assert spec.metadata['GPU_work'] is False and spec.metadata['scientific_model_forwards']==0
assert spec.output_cleanup is None
argv=list(spec.workload.argv)
assert argv[:4]==['systemd-run','--user','--scope','--quiet'] and argv[4].startswith('--unit=s4at-')
canonical=argv[5:]
assert canonical[:2]==['/project/guoqinglong/miniconda3/envs/cfbdm/bin/python','-B']
if q['run_id']==OLD:
 assert canonical[2:4]==['-m','robustness_s4_at.queue']
 assert canonical[4:]==['--preflight',str(owned/'accepted_preflight_v2.json'),'--out',str(owned/'outputs'/OLD/'result'),'--run']
work=w._process_identity(b['workload_pid']);ex=w._process_identity(b['executor_pid'])
assert work['start_ticks']==b['workload_start_ticks']==q['workload_start_ticks']
assert ex['start_ticks']==b['executor_start_ticks']==q['executor_start_ticks']
assert work['argv']==canonical and w._is_descendant(b['workload_pid'],b['executor_pid'])
assert work['process_group_id']==ex['process_group_id']==b['workload_process_group_id']==b['executor_process_group_id']
expected_cmdline=w.sha256_json(canonical);bootstrap=w.sha256_json(argv)
assert b['workload_argv_sha256']==bootstrap
recorded=b['workload_cmdline_sha256']
simulated=False
if q['run_id']==CPU and q.get('force_bootstrap_fixture'):
 recorded=bootstrap;simulated=True
assert recorded in (bootstrap,expected_cmdline)
assert work['cmdline_sha256']==expected_cmdline
group=[]
for d in Path('/proc').iterdir():
 if d.name.isdigit():
  try:
   pid=int(d.name)
   if os.getpgid(pid)==work['process_group_id']:
    assert d.stat().st_uid==1014 and w._is_descendant(pid,b['executor_pid'])
    group.append(pid)
  except (FileNotFoundError,ProcessLookupError):pass
assert b['workload_pid'] in group and b['executor_pid'] in group
with zipfile.ZipFile(zip_path) as z:raw=z.read('remote_run_control/worker.py').decode()
tree=ast.parse(raw);f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='abort')
assert hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()=='fef1252d98a2d0b0b5aa4c473924f20a5e3a3108c82f98553629cfd6dae20a2d'
needle=ast.dump(ast.parse('binding.get("workload_cmdline_sha256")',mode='eval').body,include_attributes=False)
class PostExec(ast.NodeTransformer):
 def __init__(self):self.changes=0
 def visit_Compare(self,node):
  if ast.unparse(node.left)=='workload_identity[\'cmdline_sha256\']' and len(node.comparators)==1 and ast.dump(node.comparators[0],include_attributes=False)==needle:
   assert len(node.ops)==1 and isinstance(node.ops[0],ast.NotEq)
   node.comparators[0]=ast.Name(id='expected_postexec_cmdline_sha256',ctx=ast.Load());self.changes+=1
  return self.generic_visit(node)
fix=PostExec();f=fix.visit(f);assert fix.changes==1;ast.fix_missing_locations(f)
scope=dict(vars(w),expected_postexec_cmdline_sha256=expected_cmdline)
exec(compile(ast.Module(body=[f],type_ignores=[]),'<scoped RRCTL postexec abort guard>','exec'),scope)
receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=q['run_id'],action=q['action'],
 binding_sha256=q['binding_sha256'],binding_unmodified=True,recorded_bootstrap_or_canonical=recorded,
 actual_canonical_cmdline_sha256=expected_cmdline,exact_RunSpec_suffix_matched=True,
 workload_pid=b['workload_pid'],workload_start_ticks=work['start_ticks'],executor_pid=b['executor_pid'],
 executor_start_ticks=ex['start_ticks'],UID=1014,verified_group=group,only_one_cmdline_comparison_repaired=True,
 all_original_abort_guards_and_retention_preserved=True,fixture_bootstrap_capture_simulated=simulated,signals_sent=False)
if q['action']=='inspect':print(json.dumps(receipt));sys.exit(0)
path=control/('postexec_abort_guard_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
with path.open('x') as out:json.dump(receipt,out,indent=2)
result=scope['abort'](control)
print(json.dumps(dict(guard=receipt,result=result,controlled_abort_invoked=True)))
'''

def request(binding, raw_binding, *, action='inspect', fixture=False):
    return dict(run_id=binding['run_id'],uid=1014,action=action,authorization_sha256=AUTH_SHA,
                spec_sha256=binding['run_spec_sha256'],binding_sha256=hashlib.sha256(raw_binding).hexdigest(),
                workload_start_ticks=binding['workload_start_ticks'],executor_start_ticks=binding['executor_start_ticks'],
                force_bootstrap_fixture=fixture)

def invoke(transport, remote_python, req):
    result=transport.run([remote_python,'-B','-c',CONTROL,json.dumps(req)])
    return dict(returncode=result.returncode,
                stdout=result.stdout.decode(errors='replace') if isinstance(result.stdout,bytes) else result.stdout,
                stderr=result.stderr.decode(errors='replace') if isinstance(result.stderr,bytes) else result.stderr)
