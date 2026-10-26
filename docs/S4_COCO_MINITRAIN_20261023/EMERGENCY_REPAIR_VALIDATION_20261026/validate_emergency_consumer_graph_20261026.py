"""Explicit synthetic contract fixtures, never experimental ACCEPTED/READY receipts."""
import copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
R=next(p for p in Path(__file__).resolve().parents if (p/'coco_s4').is_dir());sys.path.insert(0,str(R))
from coco_s4 import phase_victim as victim,phase_official_eval as evaluation,phase_final_report as final
from coco_s4.runtime import sha
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);O=ap.parse_args().output.resolve();assert O.is_relative_to(R/'execution');O.mkdir(exist_ok=False)
def put(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes((json.dumps(value,sort_keys=True)+'\n').encode())
def receipt(stage,**kwargs):
 p=O/stage/'accepted.json';put(p,dict(state='ACCEPTED',SIMULATED_CONTRACT_FIXTURE_ONLY=True,**kwargs));return p
def marker(stage,name):
 p=O/stage/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(b'SIMULATED CONTRACT MARKER; NOT A MODEL CHECKPOINT\n');return p
t=marker('teacher200','train/weights/last.pt')
receipt('teacher200',completed_epochs=200,zero_validation_forwards=True,last_checkpoint_sha256=sha(t))
receipt('teacher_eval',image_count=5000,classes=80,parameter_updates=False,checkpoint_selection_allowed=False,teacher_checkpoint_sha256=sha(t))
c=json.loads((R/'execution/p/coco_dim_20261025_1340z_schema/result/candidate_receipt.json').read_text())
c.update(SIMULATED_CONTRACT_FIXTURE_ONLY=True)
put(O/'calibration/accepted.json',c)
put(O/'fourier24/accepted.json',dict(state='ACCEPTED',SIMULATED_CONTRACT_FIXTURE_ONLY=True,epochs_completed=24,validation_used=False,calibration_receipt_sha256=sha(O/'calibration/accepted.json')))
manifest=marker('export25k','poisoned_manifest.jsonl')
receipt('export25k',image_count=25000,labels_byte_identical=True,decoded_budget_passed=True,validation_used=False,manifest_sha256=sha(manifest),original_train_manifest_sha256='1'*64)
def deps(names):
 result={}
 for name in names:
  p=O/name/'accepted.json';row=dict(accepted_path=str(p),accepted_sha256=sha(p))
  if name in ('teacher200','clean200','s4_200'):
   last=O/name/'train/weights/last.pt';row.update(checkpoint_path=str(last),checkpoint_sha256=sha(last))
  if name=='export25k':row['manifest_sha256']=sha(manifest)
  if name=='clean200':row['initial_state_sha256']='2'*64
  result[name]=row
 return dict(phase_dependencies=result)
def sync():
 put(O/'pipeline_state.json',dict(SIMULATED_CONTRACT_FIXTURE_ONLY=True,stages={p.parent.name:dict(state='ACCEPTED',receipt_sha256=sha(p)) for p in O.glob('*/accepted.json')}))
checked=[];rejected=[]
def reject(name,fn):
 try:fn()
 except (ValueError,FileNotFoundError,KeyError):rejected.append(name)
 else:raise AssertionError('Invalid contract accepted: '+name)
for phase in ('clean200','s4_200'):
 sync();names=['teacher200','teacher_eval','calibration','fourier24','export25k']+(['clean200'] if phase=='s4_200' else [])
 m=deps(names);cfg=json.loads((R/f'coco_s4/configs/{phase}.json').read_text())
 binding=victim.validate_phase_binding(m,cfg,phase,O);checked.append(phase+'_production_dependency_validator')
 bad=copy.deepcopy(m);bad['phase_dependencies']['fourier24']['accepted_sha256']='0'*64
 reject(phase+'_old_receipt_hash',lambda:victim.validate_phase_binding(bad,cfg,phase,O))
 wrong=copy.deepcopy(cfg);wrong['initialization']='Teacher weights'
 reject(phase+'_Teacher_initialization',lambda:victim.validate_phase_binding(m,wrong,phase,O))
 wrong=copy.deepcopy(cfg);wrong['training']['batch']=72
 reject(phase+'_batch_drift',lambda:victim.validate_phase_binding(m,wrong,phase,O))
 last=marker(phase,'train/weights/last.pt');full=marker(phase,'train/state_snapshots/epoch_200_full_state.pt')
 receipt(phase,completed_epochs=200,primary_checkpoint='last.pt@200',zero_validation_forwards=True,pretrained=False,
         primary_checkpoint_sha256=sha(last),full_state_sha256=sha(full),initial_state_sha256='2'*64,matched_clean_initial_state=True)
sync();m=deps(evaluation.STAGES);accepted=evaluation.accepted_dependencies(m,O)
checked.append('official_eval_production_dependency_validator')
for key,value in [('initial_state_sha256','3'*64),('matched_clean_initial_state',False)]:
 p=O/'s4_200/accepted.json';old=p.read_bytes();r=json.loads(old);r[key]=value;put(p,r);sync();bad=deps(evaluation.STAGES)
 reject('eval_'+key,lambda:evaluation.accepted_dependencies(bad,O));p.write_bytes(old)
sync()
for name in ('official_eval','final_report'):
 config=json.loads((R/f'coco_s4/configs/{name}.json').read_text())
 (evaluation.validate_config if name=='official_eval' else final.validate_config)(config)
 checked.append(name+'_exact_config')
bad=copy.deepcopy(m);bad['phase_dependencies']['teacher200']['accepted_path']=str(O/'teacher200/train/weights/best.pt')
reject('teacher_best_not_last',lambda:evaluation.accepted_dependencies(bad,O))
# Native metrics/export schemas are separately checked on real bounded artifacts.
# These values test report wiring only; no generated AP is claimed as a result.
mapping=json.loads((R/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json').read_text())['mapping']
clean=[dict(x,AP50=.5,AP75=.4,AP50_95=.3,Recall50=.6,Precision50=.7) for x in mapping]
s4=copy.deepcopy(clean);s4[0]['AP50']=.1
report=final.build_report(clean,clean,s4,mapping,{'SIMULATED_ONLY':True},provenance={'calibration_dimension':c,'SIMULATED_ONLY':True})
assert report['paired_summary']['P']==.8 and report['paired_summary']['U']==1
checked.append('actual_final_report_builder_Clean_not_Teacher_denominator')
changed_teacher=copy.deepcopy(clean);changed_teacher[0]['AP50']=.99
report2=final.build_report(changed_teacher,clean,s4,mapping,{'SIMULATED_ONLY':True},provenance={'calibration_dimension':c,'SIMULATED_ONLY':True})
assert report2['paired_summary']==report['paired_summary']
from coco_s4.runtime import assert_formal_launch_authorized
reject('synthetic_acceptance_cannot_override_emergency_scope',assert_formal_launch_authorized)
assert not list(O.glob('*READY*'))
out=dict(state='passed',scope='SIMULATED contract fixtures only; NO model/training/inference/AP result',
         production_validators_exercised=checked,adversarial_rejections=rejected,
         fixture_files=[dict(path=str(p.relative_to(O)),sha256=sha(p)) for p in O.rglob('*') if p.is_file()],
         official_READY_written=False,official_ACCEPTED_written=False,model_forwards=0,parameter_updates=0,
         fixture_ACCEPTED_location=str(O),formal_launch=False)
put(O/'report.json',out);print(json.dumps({k:out[k] for k in ('state','production_validators_exercised','adversarial_rejections','formal_launch')}))
