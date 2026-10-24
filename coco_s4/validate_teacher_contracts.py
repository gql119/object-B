"""Targeted source/data/argument guard probes; no GPU or formal trainer setup."""
from pathlib import Path
from types import SimpleNamespace
import copy, datetime, hashlib, json, sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import torch
import yaml
from coco_s4.teacher import COCOTrainOnlyTeacher, teacher_overrides

EX=R/'execution'
p=json.loads((EX/'TEACHER_CPU_VALIDATION_20261025.json').read_text())
source_sha=hashlib.sha256((R/'coco_s4/teacher.py').read_bytes()).hexdigest()
assert p['source_sha256']==source_sha and p['validation_forwards']==0
out=EX/('teacher_guard_probe_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(exist_ok=False)
base=yaml.safe_load((Path(p['output'])/'teacher_train_only.yaml').read_text())
def load_data(data, name):
    path=out/(name+'.yaml');path.write_text(yaml.safe_dump(data),encoding='utf-8')
    t=COCOTrainOnlyTeacher.__new__(COCOTrainOnlyTeacher)
    t.args=SimpleNamespace(data=str(path))
    return t.get_dataset()
good=load_data(base,'valid')
assert good['val'] is None and good['nc']==80 and len(good['names'])==80
invalid=[dict(base,val='images/val'),dict(base,nc=20),dict(base,train='images/val'),dict(base,download='https://example.invalid/data.zip')]
bad=copy.deepcopy(base);bad['names'][1]='wrong-class-order';invalid.append(bad)
for i,data in enumerate(invalid):
    try: load_data(data,'invalid_'+str(i))
    except ValueError: pass
    else: raise AssertionError('Invalid train-only data route accepted')
cfg=yaml.safe_load((R/'docs/S4_COCO_MINITRAIN_20261023/S4_COCO_EXPERIMENT_CONFIG.yaml').read_text())
args=teacher_overrides(cfg,out/'valid.yaml',out/'must_not_exist')
mutations=[{'epochs':199},{'batch':18},{'workers':0},{'amp':False},{'classes':[0]},
           {'single_cls':True},{'pretrained':True},{'resume':True},{'val':True},
           {'mixup':.5},{'warmup_epochs':4},{'model':'weights.pt'},{'name':'../escape'},
           {'fraction':.5},{'save_period':1}]
for change in mutations:
    try: COCOTrainOnlyTeacher(overrides={**args,**change})
    except (ValueError,FileExistsError): pass
    else: raise AssertionError('Invalid frozen Teacher argument accepted')
    assert not (out/'must_not_exist').exists()
for f,key in [('ORIGINAL_S4_METHOD_SOURCE_LOCK.json','files'),('ORIGINAL_VICTIM_COMPONENT.json','files')]:
    lock=json.loads((R/'docs/S4_COCO_MINITRAIN_20261023'/f).read_text())
    for rel,digest in lock[key].items(): assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==digest,rel
assert not torch.cuda.is_initialized()
proof={'state':'TEACHER_FROZEN_ARGUMENT_AND_DATA_GUARDS_PASSED','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'source_sha256':source_sha,'invalid_data_routes_rejected':len(invalid),
       'frozen_argument_mutations_rejected':len(mutations),'all80_official_names_checked':True,
       'original8_math_and2_victim_sources_unchanged':True,'no_output_created_by_invalid_formal_requests':True,
       'GPU_initialized':False,'formal_training_started':False,'deleted_paths':[],'output':str(out)}
(EX/'TEACHER_CONTRACT_VALIDATION_20261025.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))
