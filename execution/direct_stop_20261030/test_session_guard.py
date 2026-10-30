"""Targeted pure guard cases; not a process/GPU/scientific test."""
import ast,hashlib,json
from pathlib import Path
from guarded_abort_repair import CONTROL
tree=ast.parse(CONTROL)
f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_session_guard')
ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),'<actual all-pane guard>','exec'),ns)
guard=ns['_session_guard'];assert guard('$11\tAT\t123\n','AT',123)=='$11'
cases=dict(extra_pane='$11\tAT\t123\n$11\tAT\t999\n',other_window_pane='$11\tAT\t123\n$11\tAT\t456\n',
           wrong_single_pane='$11\tAT\t999\n',wrong_session='$11\tOTHER\t123\n')
rejected=[]
for name,text in cases.items():
    try:guard(text,'AT',123)
    except AssertionError:rejected.append(name)
    else:raise AssertionError('Wrong pane/session was accepted: '+name)
v=dict(state='PASS_ACTUAL_GUARD_PURE_CASES',simulation_only=True,control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),
       valid_single_pane_passed=True,rejected=rejected,signals_sent=0,GPU_work=False)
path=Path(__file__).parent/'ALL_PANES_GUARD_LOCAL_CASES.json'
with path.open('x',encoding='utf8') as out:json.dump(v,out,indent=2)
print(json.dumps(v))
