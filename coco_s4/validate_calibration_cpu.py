"""Real native CPU gradient comparison and synthetic coordinate safety probes."""
from pathlib import Path
import copy,datetime,hashlib,json,os
import numpy as np
from PIL import Image
import torch
import yaml

R=Path(__file__).resolve().parents[1]
O=R/'execution'/('calibration_cpu_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
O.mkdir(parents=True,exist_ok=False);(O/'yolo_settings').mkdir();os.environ['YOLO_CONFIG_DIR']=str(O/'yolo_settings')
torch.set_num_threads(1)
from ultralytics.nn.tasks import DetectionModel
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer
from ue_framework.methods.semantic_coordinates import gt_evidence
from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
from coco_s4.calibration import collect,build_coordinates,ProtocolSpaceBlocked
from coco_s4.runtime import sha

cfg=yaml.safe_load((R/'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml').read_text())
cfg['surrogate']['imgsz']=64
method=cfg['methods']['legacy_kproto_ret']
torch.manual_seed(9)
model=DetectionModel(str(R/'configs/coco_yolov8n_80cls.yaml'),nc=80,verbose=False).eval()
for p in model.parameters():p.requires_grad_(False)
original={k:v.clone() for k,v in model.state_dict().items()}
images=O/'fixture/images';labels=O/'fixture/labels';images.mkdir(parents=True);labels.mkdir()
ann_text=''.join(f'{c} 0.5 0.5 0.4 0.4\n' for c in range(80))
for i in range(2):
    Image.fromarray(np.random.default_rng(i).integers(0,256,(64,64,3),dtype=np.uint8)).save(images/f'{i:012d}.png')
    (labels/f'{i:012d}.txt').write_text(ann_text)
trainer=LegacyKProtoTrainer(cfg,method,torch.device('cpu'),model)
paths=sorted(images.glob('*.png'))
raw=collect(trainer,model,paths,labels,O/'collected')
assert raw['target_count']==2 and raw['images']==2 and len(raw['eligible_person_images'])==2
assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
for layer in ('model.15','model.21'):
    # Named canonical builder loop, with independent raw forwards per layer.
    target=None;nt=None;by={c:None for c in range(1,80)}
    for path in paths:
        im=load_image_rgb_float(str(path));anns=read_yolo_annotations(str(labels/(path.stem+'.txt')))
        x=trainer._to_tensor(im).requires_grad_(True);trainer.multi_features.clear();decoded=trainer._forward_raw(x)
        feature=trainer.multi_features[layer]
        for c,e in gt_evidence(decoded,anns,im.shape[:2],trainer.imgsz):
            g=torch.autograd.grad(e,feature,retain_graph=True)[0].detach().abs().mean((0,2,3)).double().cpu()
            if c==0:target=g.clone() if target is None else target+g
            else:
                nt=g.clone() if nt is None else nt+g
                by[c]=g.clone() if by[c] is None else by[c]+g
    item=raw['layers'][layer]
    assert torch.equal(item['target_sum'],target) and torch.equal(item['nt_sum'],nt)
    assert all(torch.equal(item['nt_sums'][c],by[c]) for c in by)

# Explicit synthetic full-rank arrays test artifact flow and the immutable24D
# requirement. They are not COCO learned statistics or method efficacy evidence.
gen=torch.Generator().manual_seed(17);classes=list(range(1,80));N=32
layers={}
for layer,C in [('model.15',64),('model.21',256)]:
    st=torch.ones(C,dtype=torch.float64)*.1
    if C==256:st[130:233]=2.
    else:st[15:45]=2.
    nt=1+torch.arange(C,dtype=torch.float64)/C
    by={c:nt.clone()*N for c in classes}
    layers[layer]=dict(target_sum=st*2,nt_sum=sum(by.values()),nt_sums=by,nt_counts={c:N for c in classes},
                       pools={c:list(torch.randn(N,C,generator=gen)) for c in classes})
synthetic=dict(layers=layers,target_count=2,images=2,clean_train_manifest_sha256='1'*64,
               frozen_state_sha256='2'*64,eligible_person_images=list(map(str,paths)),validation_used=False)
cfg['surrogate']['ckpt']=str(O/'synthetic_teacher_identity.json');Path(cfg['surrogate']['ckpt']).write_text('{}')
def bound_cfg(out):
    c=copy.deepcopy(cfg);s=c['methods']['legacy_kproto_ret']['semantic'];s['artifact_dir']=str(out/'coordinates')
    s['safe_bg']['reference_artifact_dir']=str(out/'reference');s['strict_tsfl']['layer_selection_path']=str(out/'FIXED_P5.json');return c
out=O/'good';result=build_coordinates(synthetic,bound_cfg(out),out,builder_path=R/'coco_s4/calibration.py',source_root=R)
assert len(result['selected_channels'])==24
summary=json.loads((out/'coordinates/tsfl_summary.json').read_text())
assert all(k in summary for k in ('num_channels_selected','mean_S_target_selected','mean_S_nt_selected',
           'mean_S_target_unselected','mean_S_nt_unselected','specificity_selected_mean','channel_selection_overlap_with_previous_run'))
assert (out/'reference/target_specific_channels.pt').read_bytes()==(out/'coordinates/orth_target_specific_channels.pt').read_bytes()
assert (out/'reference/non_target_subspaces.pt').read_bytes()==(out/'coordinates/orth_non_target_subspaces.pt').read_bytes()
assert set(torch.load(out/'coordinates/safe_bg_selected_subspaces.pt',weights_only=True)['classes'])==set(range(1,80))
bad=copy.deepcopy(synthetic);bad['layers']['model.21']['target_sum'].fill_(1)
blocked=False
try:build_coordinates(bad,bound_cfg(O/'incompatible'),O/'incompatible',builder_path=R/'coco_s4/calibration.py',source_root=R)
except ProtocolSpaceBlocked:blocked=True
assert blocked and (O/'incompatible/BLOCKED_PROTOCOL_COMPATIBILITY.json').is_file()
assert not torch.cuda.is_initialized()
proof=dict(state='CALIBRATION_COMPONENT_CPU_VALIDATED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_sha256=sha(R/'coco_s4/calibration.py'),actual_native_CPU_images=2,actual_classes_per_image=80,
 P3_P5_all_per_GT_gradient_sums_equal_independent_canonical_loops=True,frozen_model_unchanged=True,
 synthetic_79_class_artifact_flow_passed=True,P3_reference_copies_byte_exact=True,
 incompatible_selected_space_blocks_without_truncation_or_threshold_relaxation=True,
 formal_COCO_statistics=False,GPU_initialized=False,GPU_smoke_passed=False,scientific_review_passed=False,output=str(O))
(R/'execution/CALIBRATION_COMPONENT_CPU_VALIDATION_20261025.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
