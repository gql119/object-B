"""Finite CPU diagnostics: defined per-GT terms, real boundary cases, regressions.

The strict original function still raises on the failed GT. Its actual forward
frame supplies native predictions/matches for a diagnostic per-object reference.
Undefined original terms are recorded null and never used to construct a risk map.
No original source or training labels are modified.
"""
import sys,json,inspect,hashlib,io,tempfile,unittest,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(1)
from coco_s4.runtime import verify_release,DATA_ROOT,sha,write_once,atomic
from coco_s4.phase_calibration import load_teacher
from coco_s4.phase_contract import teacher_dependency
from coco_s4.risk_fallback import gradients as candidate,_with_fallback_gradients,risk_candidate_mask
from ue_framework.methods.tfss_nt_sensitivity import gradients as reference
from ue_framework.methods.s4_tfss import model_hash
from ue_framework.methods.tfss_spatial import normalize_map,spatial_weights
from ue_framework.methods.legacy_kproto_ret import _rectangle
from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

anchor,out,chain=map(Path,sys.argv[1:4]);m=verify_release(anchor);out.mkdir(parents=True,exist_ok=False)
common=dict(source_commit=m['source_commit'],validation_only=True,official_metrics_disabled=True,
 artifact_ingest_disabled=True,formal_READY_written=False,real_Teacher_parameter_updates=0,actual_risk_Fourier_updates=0,device='cpu')
atomic(out/'status.json',dict(common,state='RUNNING'))
counter=[];forward0=DetectionModel.forward
def forward(model,x,*args,**kw):
 value=forward0(model,x,*args,**kw);counter.append(dict(device=str(x.device),shape=list(x.shape)))
 assert len(counter)<=100
 return value
DetectionModel.forward=forward
try:
 checkpoint,_=teacher_dependency(m['teacher_dependency']);model,before=load_teacher(checkpoint,m['teacher_dependency']['checkpoint_sha256'],torch.device('cpu'))
 def data(stem):
  image=DATA_ROOT/'images/train'/(stem+'.jpg');label=DATA_ROOT/'labels/train'/(stem+'.txt')
  return load_image_rgb_float(str(image)),read_yolo_annotations(str(label)),dict(image_id=stem,image_sha256=sha(image),label_sha256=sha(label))
 image,anns,meta=data('000000247692');kw=dict(imgsz=640,coverage_policy='independent_same_TAL_for_uncovered_valid_GT',target_class_id=0,num_classes=80)
 original_context={};captured={}
 def line_of(function,text):
  source,start=inspect.getsourcelines(function)
  return start+next(i for i,v in enumerate(source) if text in v)
 original_line=line_of(reference,'losses = {}');candidate_line=line_of(_with_fallback_gradients,'losses.setdefault(cls')
 def trace(frame,event,arg):
  if event=='line' and frame.f_code is reference.__code__ and frame.f_lineno==original_line:
   original_context.update(frame.f_locals)
  if event=='line' and frame.f_code is _with_fallback_gradients.__code__ and frame.f_lineno==candidate_line:
   v=frame.f_locals;g=torch.autograd.grad(v['loss'],v['x'],retain_graph=True)[0]
   captured[v['j']]=dict(loss=float(v['loss'].detach()),gradient=g.detach().clone(),cls=v['cls'])
  return trace
 old_trace=sys.gettrace();sys.settrace(trace)
 try:
  try:reference(model,image,anns,**kw)
  except ValueError as exc:assert str(exc)=='NO_VALID_POSITIVE: GT index=13, class=24'
  else:raise AssertionError('Strict original boundary unexpectedly changed')
  maps,rows=candidate(model,image,anns,**kw)
 finally:sys.settrace(old_trace)
 c=original_context;criterion=c['criterion'];keep=[i for i in range(80) if i!=0]
 comparisons=[];undefined=[]
 for j,a in enumerate(anns):
  if a['cls']==0:continue
  hit=c['fg'][0] & (c['gi'][0]==j);truth,scores=c['assigned_boxes'],c['scores']
  if not bool(hit.any()):
   with torch.no_grad():
    _,truth,scores,fg,_=criterion.assigner(c['ps'].detach().sigmoid(),c['decoded'].detach()*c['stride'],
     c['anchors']*c['stride'],c['labels'][:,j:j+1],c['gtboxes'][:,j:j+1],c['gtboxes'][:,j:j+1].sum(-1,keepdim=True)>0)
   hit=fg[0]
  weight=scores[0,hit].sum(-1).detach()
  if not bool(hit.any()) or float(weight.sum())<=0:
   undefined.append(dict(gt_index=j,cls=a['cls'],original_loss=None,reason='no native effective supervision'))
   continue
  weight=weight/weight.sum()
  bce=torch.nn.functional.binary_cross_entropy_with_logits(c['ps'][0,hit][:,keep],scores[0,hit][:,keep],reduction='none').sum(-1)
  truth=truth/c['stride'];ciou=1-bbox_iou(c['decoded'][0,hit],truth[0,hit],xywh=False,CIoU=True).flatten()
  reg_max=criterion.bbox_loss.dfl_loss.reg_max;ltrb=bbox2dist(c['anchors'],truth,reg_max-1)
  dfl=criterion.bbox_loss.dfl_loss(c['pd'][0,hit].reshape(-1,reg_max),ltrb[0,hit]).flatten()
  terms=torch.stack(((ciou*weight).sum(),(bce*weight).sum(),(dfl*weight).sum()))
  loss=(terms*terms.new_tensor([criterion.hyp.box,criterion.hyp.cls,criterion.hyp.dfl])).sum()
  gradient=torch.autograd.grad(loss,c['x'],retain_graph=True)[0]
  measured=captured[j];assert float(loss.detach())==measured['loss'] and torch.equal(gradient,measured['gradient'])
  comparisons.append(dict(gt_index=j,cls=a['cls'],loss_exact=True,full_input_gradient_exact=True,tolerance=0))
 expected={j for j,a in enumerate(anns) if a['cls']!=0}
 assert set(captured)==expected=={r['gt_index'] for r in rows} and len(rows)==len(expected)
 assert len(comparisons)==13 and [u['gt_index'] for u in undefined]==[13]
 # Check the actual class-normalized maps against the already validated full-chain cache.
 cached=list((chain/'clean_maps').glob('*.json'));assert len(cached)==1
 record=json.loads(cached[0].read_text());npz=cached[0].with_suffix('.npz');assert sha(npz)==record['npz_sha256']
 assert record['identity']['annotations']==anns and record['identity']['risk_estimator_policy']==m['risk_estimator_policy']
 target=torch.zeros(image.shape[:2],dtype=torch.bool)
 for a in anns:
  if a['cls']==0:target|=_rectangle(image.shape[:2],a['bbox'])[0,0].bool()
 normalized=[normalize_map(torch.as_tensor(v),target,constant_policy='preserve')[0] for _,v in sorted(maps.items())]
 risk=torch.stack(normalized).mean(0)
 with np.load(npz,allow_pickle=False) as z:
  q=torch.from_numpy(z['q'].copy());assert np.array_equal(risk.numpy(),z['risk'])
 weight=spatial_weights(q,risk,target.float(),mode='TFSS-U',eta=.85)
 assert torch.equal(weight,torch.where(target,1-(1-.85)*risk*(1-q),torch.ones_like(q)))
 assert torch.isfinite(weight).all() and float(weight.min())>=.85 and float(weight.max())<=1
 class_counts={str(cls):sum(int(a['cls'])==cls for a in anns) for cls in sorted(maps)}
 # Find genuine train-only boundary images from original JSON, verify their actual labels.
 document=json.loads((DATA_ROOT/'metadata/instances_minitrain2017.json').read_text())
 category={int(v['id']):i for i,v in enumerate(sorted(document['categories'],key=lambda v:v['id']))}
 present={int(v['id']):set() for v in document['images']}
 for a in document['annotations']:
  if not a.get('iscrowd',0) and not a.get('ignore',0):present[int(a['image_id'])].add(category[int(a['category_id'])])
 no_person=next(k for k,v in sorted(present.items()) if v and 0 not in v)
 no_nt=next(k for k,v in sorted(present.items()) if v=={0})
 boundary=[]
 for image_id,kind in ((no_person,'real_no_Person'),(no_nt,'real_Person_only')):
  im,ann,identity=data(f'{image_id:012d}');start=len(counter);mm,rr=candidate(model,im,ann,**kw)
  if kind=='real_Person_only':assert mm=={} and rr==[] and len(counter)==start
  else:
   assert all(a['cls']!=0 for a in ann) and len(rr)==len(ann) and set(mm)=={int(a['cls']) for a in ann}
   assert len({r['gt_index'] for r in rr})==len(ann) and all(np.isfinite(v).all() and np.linalg.norm(v)>0 for v in mm.values())
  boundary.append(dict(**identity,kind=kind,state='passed',GT_count=len(ann),forward_count=len(counter)-start))
 # Explicit synthetic candidate-selection contracts; never write them as real GT.
 from types import SimpleNamespace
 assigner=SimpleNamespace(select_candidates_in_gts=lambda anchors,box,valid:torch.zeros(1,1,len(anchors),dtype=torch.bool))
 mask,detail=risk_candidate_mask(assigner,torch.tensor([[4.,4.],[6.,4.]]),torch.tensor([[[4.5,4.5,5.5,5.5]]]))
 assert mask.tolist()==[True,False] and detail['count']==1 and 'nearest' in detail['rule']
 for box in (torch.tensor([[[1.,1.,1.,2.]]]),torch.tensor([[[2.,1.,1.,2.]]]),torch.full((1,1,4),float('nan'))):
  try:risk_candidate_mask(assigner,torch.zeros(1,2),box)
  except ValueError:pass
  else:raise AssertionError('Invalid GT accepted')
 # Original requested VOC regression modules, retaining all fixture artifacts.
 root=Path.cwd();sys.path.insert(0,str(root/'tests'))
 write_once(out/'mixed_pre_regression.json',dict(common,state='mixed_contracts_passed_regressions_pending',
  covered_per_object_comparisons=comparisons,original_undefined_objects=undefined,real_boundary_cases=boundary,
  class_normalization_and_mean_matches_native_cache_exact=True,W_min=float(weight.min()),W_max=float(weight.max())))
 from ultralytics import YOLO
 fixture=out/'synthetic_random_VOC20.pt'
 torch.manual_seed(0);YOLO(str(root/'configs/voc_yolov8n_20cls.yaml')).save(str(fixture))
 os.environ['KPROTO_CKPT']=str(fixture)
 os.environ['KPROTO_CONFIG']=str(root/'ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml')
 class RetainedTemporaryDirectory:
  def __init__(self,*args,**kwargs):self.name=tempfile.mkdtemp(prefix='risk_regression_',dir=out)
  def __enter__(self):return self.name
  def __exit__(self,*args):return False
  def cleanup(self):pass  # Torch's lazy integration also expects this API; retain files.
 temporary0=tempfile.TemporaryDirectory;tempfile.TemporaryDirectory=RetainedTemporaryDirectory
 modules=['test_b2_semantic','test_b2_adaptive','test_legacy_kproto_ret','test_bbox_half_overlap','test_tfss_spatial','test_tfss_static_control','test_orth_reference_design','coco_s4.tests.test_compatibility']
 log=io.StringIO()
 try:result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
 finally:tempfile.TemporaryDirectory=temporary0
 (out/'regressions.log').write_text(log.getvalue());assert result.wasSuccessful() and not result.skipped and result.testsRun==55
 assert model_hash(model)==before and sha(checkpoint)==m['teacher_dependency']['checkpoint_sha256']
 assert not any(p.grad is not None for p in model.parameters()) and not torch.cuda.is_initialized()
 summary=dict(common,state='passed',fixture=meta,covered_per_object_comparisons=comparisons,original_undefined_objects=undefined,
  all_NT_GT_count=len(expected),all_NT_GT_each_once=True,class_GT_counts=class_counts,class_normalization_and_mean_matches_native_cache_exact=True,
  W_min=float(weight.min()),W_max=float(weight.max()),W_mean=float(weight[target].mean()),original_complete_Rnt=None,
  real_boundary_cases=boundary,nearest_candidate_and_invalid_geometry_fixtures='passed',regressions=dict(tests=55,failures=0,errors=0,skips=0,synthetic_VOC_Adam_updates=1),
  actual_forward_count=len(counter),actual_forward_evidence=counter,Teacher_unchanged=True,checkpoint_unchanged=True,
  full25K_statistics='OPEN',audit_GPU_initialized=False)
 write_once(out/'mixed_validation_summary.json',summary);atomic(out/'status.json',summary)
except BaseException as exc:
 atomic(out/'status.json',dict(common,state='FAILED',error=repr(exc)));raise
finally:DetectionModel.forward=forward0
