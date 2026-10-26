"""Finite RRCTL test of the exact production export hook; no formal release/READY."""
import copy,json,sys
from pathlib import Path
E=next(p/'execution' for p in Path(__file__).resolve().parents if (p/'coco_s4').is_dir());sys.path.insert(0,str(E.parent))
from coco_s4.dimension_release import services,preflight,current_source,finite_spec,release,PYTHON,EX
from coco_s4.runtime import sha
TAG='coco_dim_20261026_2240hk_devcheck'
CODE=r'''
import ast,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(1)
from coco_s4.runtime import verify_release,sha,write_once,atomic,DATA_ROOT,ROOT
from coco_s4.phase_contract import teacher_dependency
from coco_s4.phase_calibration import load_teacher,train_manifest
from coco_s4.coco_eval import raw_signature,raw_tensor_device_details
from ue_framework.methods.base import BasePoisonGenerator
from ue_framework.methods.s4_tfss import model_hash
from ue_framework.data_utils import load_image_rgb_float
anchor,output=sys.argv[1:];output=Path(output);output.mkdir(parents=True,exist_ok=False)
m=verify_release(anchor);checkpoint,_=teacher_dependency(m['teacher_dependency'])
common=dict(source_commit=m['source_commit'],release_sha256=sha(anchor),validation_only=True,
 official_metrics_disabled=True,artifact_ingest_disabled=True,parameter_updates=0,formal_READY_written=False)
atomic(output/'status.json',dict(common,state='RUNNING'))
try:
 source=ROOT/'coco_s4/phase_export.py';assert sha(source)==m['files']['coco_s4/phase_export.py']
 module=ast.parse(source.read_bytes());fn=next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name=='observe_forward')
 calls=[];ns=dict(raw_signature=raw_signature,raw_tensor_device_details=raw_tensor_device_details,forward_calls=calls)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 observer=ns['observe_forward']
 paths,hashes,_=train_manifest();eligible=json.loads(Path(m['dimension_reconstruction_dependency']['phase_spec_path']).read_text())
 manifest=Path(eligible['artifact_root'])/'collected/eligible_support_manifest.json'
 document=json.loads(manifest.read_text());path=Path(document['paths'][int(np.random.RandomState(0).permutation(document['count'])[0])])
 assert path in paths and sha(path)==hashes[path.stem][0]
 image=load_image_rgb_float(str(path));x=torch.from_numpy(image).permute(2,0,1)[None].float()
 fixture=object.__new__(BasePoisonGenerator);fixture.imgsz=640;fixture.pad_value=.5
 x=BasePoisonGenerator._letterbox_tensor(fixture,x)
 teacher,expected=load_teacher(checkpoint,m['teacher_dependency']['checkpoint_sha256'],torch.device('cpu'))
 before=model_hash(teacher);assert before==expected
 count=[]
 h0=teacher.register_forward_hook(lambda module,args,result:count.append(dict(device=str(args[0].device),raw=raw_tensor_device_details(result))))
 h1=teacher.register_forward_hook(observer)
 rejected=False
 with torch.no_grad():
  try:teacher(x)
  except ValueError as exc:
   assert 'CUDA0' in str(exc);rejected=True
  assert rejected and not torch.cuda.is_initialized()
  teacher.to('cuda:0');teacher(x.to('cuda:0'))
 h1.remove();h0.remove()
 assert len(count)==2 and len(calls)==1
 assert calls[0]['devices']==['cuda:0'] and calls[0]['cuda_indices']==[0]
 assert model_hash(teacher)==before and sha(checkpoint)==m['teacher_dependency']['checkpoint_sha256']
 result=dict(common,state='passed',actual_native_forward_calls=2,CPU_negative_forward_calls=1,
  CUDA_positive_forward_calls=1,CPU_output_rejected=True,actual_forward_evidence=count,
  exact_production_hook_source_sha256=sha(source),exact_production_hook_line=fn.lineno,
  production_hook_evidence=calls,actual_image_id=path.stem,actual_image_sha256=sha(path),
  Teacher_unchanged=True,checkpoint_unchanged=True,model_parameter_updates=0)
 write_once(output/'device_observer_summary.json',result);atomic(output/'status.json',result)
except BaseException as e:
 atomic(output/'status.json',dict(common,state='FAILED',error=repr(e)));raise
'''

def main():
 if (EX/(TAG+'.json')).exists() or (EX/TAG).exists():raise FileExistsError('Inspect existing diagnostic; no relaunch')
 ctl,t=services();check=preflight(t)
 (E/'COCO_S4_DEVICE_OBSERVER_ADMISSION_LATEST_20261026.json').write_bytes(json.dumps(check,indent=2).encode())
 if check['state']!='READY':
  print(json.dumps(dict(state=check['state'],job_created=False,capacity=check['capacity'])));return
 head=current_source();m=release(head,'fourier24')
 previous=json.loads((EX/'coco_dim_20261026_2230hk_gpu16/source_release.json').read_text())
 m.update({k:previous[k] for k in ('phase_config_path','dimension_reconstruction_dependency','dimension_reconstruction_validation_allowed','calibration_dependency')})
 m['failed_dimension_predecessor']=check['failed_dimension_predecessor']
 path=finite_spec(TAG,m,'cpu');spec=json.loads(path.read_text());work=spec['workload']['argv'];split=work.index('--')
 out=spec['remote']['output_root']
 spec['workload']['argv']=work[:split+1]+[PYTHON,'-B','-c',CODE,spec['anchors'][0]['remote_path'],out+'/result']
 spec['metadata']['validation_kind']='production-export-device-hook-only'
 spec['metadata']['actual_production_step_limit']=0
 spec['artifacts']=[dict(path='result/'+n,required=True) for n in ('status.json','device_observer_summary.json')]
 from remote_run_control.models import RunSpec
 RunSpec.from_dict(spec);path.write_bytes(json.dumps(spec,indent=2).encode())
 from coco_s4.dimension_release import launch
 launch(TAG)
if __name__=='__main__':main()
