"""CPU-only check of batch-setting labels; no model/remote operation."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
s=json.loads((ROOT/'execution/s4_at_cuda_baseline_20261029_gpu03.json').read_text(encoding='utf8'))
argv=s['workload']['argv'];old=argv[argv.index('-c')+1]
first='parameters=dict(batch=36';second='sample_count=36'
assert old.count(first)==old.count(second)==1
new=old.replace(first,"parameters=dict(batch=load(PROTOCOL)['train_args']['batch']").replace(second,"sample_count=load(PROTOCOL)['train_args']['batch']")
ast.parse(new)
assert new.replace("parameters=dict(batch=load(PROTOCOL)['train_args']['batch']",first).replace("sample_count=load(PROTOCOL)['train_args']['batch']",second)==old
result=dict(state='PASS_METADATA_BINDING_ONLY',changed_sites=2,computation_changed=False,tolerance_changed=False,models=0,GPU_work=False,
    prior_shell_compile_command_failed_before_test_execution=True,prior_failure_reason='PowerShell inline quoting; this saved UTF8 script avoids shell nesting')
(HERE/'BASELINE_METADATA_BINDING_CPU.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result))
