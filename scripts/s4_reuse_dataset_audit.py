"""Full CPU-only re-audit of existing S4 bytes; never exports images/noise."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args()
    config=json.loads((ROOT/'docs/S4_AMP_RECOVERY_20261018/VICTIM_RECOVERY_CONFIG.json').read_text())
    out=Path(args.output)
    if not out.is_relative_to(Path(config['remote_root'])/'outputs'):raise ValueError('Output boundary')
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    import ultralytics
    from ue_framework.config import load_config
    from ue_framework.data_utils import list_images,read_yolo_annotations,load_image_rgb_float
    from ue_framework.methods.s4_pixel_audit import domains
    from ue_framework.methods.legacy_kproto_ret import _rectangle
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    torch.set_num_threads(1)
    started=time.monotonic();parent=Path(config['parent_output']);data=Path(config['poisoned_dataset_path'])
    lockpath=ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json';lock=json.loads(lockpath.read_text())
    assert sha(lockpath)==config['frozen_s4_config_sha256']
    checkpoint=parent/'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt'
    integrity=parent/'dataset_integrity.json';manifest=parent/'materialization_manifest.json'
    assert sha(checkpoint)==config['parent_noise_checkpoint_sha256']
    assert sha(integrity)==config['parent_dataset_integrity_sha256']
    assert sha(manifest)==config['parent_materialization_manifest_sha256']
    prior=json.loads(integrity.read_text());assert prior['state']=='passed'
    cfg=load_config(str(ROOT/lock['parent_config_path']));clean=Path(cfg['data']['dataset_root'])
    canonical=json.loads((ROOT/'docs/tfss/S0_CANONICAL_SOURCE_SHA256.json').read_text())
    metadata=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text());s0root=Path(metadata['parent_root'])
    assert all(sha(ROOT/p)==expected and sha(s0root/p)==expected for p,expected in canonical.items())
    rows=json.loads(manifest.read_text());assert len(rows)==16551 and len({r['id'] for r in rows})==16551
    output_by_id={Path(p).stem:Path(p) for p in list_images(str(data/'images/train'))}
    assert len(output_by_id)==16551 and set(output_by_id)=={r['id'] for r in rows}
    expected_labels={r['id']+'.txt' for r in rows}
    assert {p.name for p in (data/'labels/train').glob('*.txt')}==expected_labels
    changed=targets=non_targets=0;outside_max=global_max=overlap_max=0
    for index,row in enumerate(rows):
        source=Path(row['source']);dest=output_by_id[row['id']]
        assert dest==Path(row['output']) and sha(dest)==row['output_sha256']
        label=clean/'labels/train'/(row['id']+'.txt');exported_label=data/'labels/train'/label.name
        assert sha(label)==row['label_sha256']==sha(exported_label)
        ann=read_yolo_annotations(str(label));has=any(x['cls']==14 for x in ann)
        assert has==row['has_target']
        if has:
            before=np.rint(load_image_rgb_float(str(source))*255).astype(np.uint8)
            after=np.rint(load_image_rgb_float(str(dest))*255).astype(np.uint8)
            assert before.shape==after.shape
            absolute=np.abs(after.astype(np.int16)-before.astype(np.int16)).max(2)
            target,other=domains(before.shape,ann)
            outside=int(absolute[~target].max()) if (~target).any() else 0
            overlap=int(absolute[target&other].max()) if (target&other).any() else 0
            assert outside==0 and int(absolute.max())<=8 and overlap<=4
            outside_max=max(outside_max,outside);global_max=max(global_max,int(absolute.max()));overlap_max=max(overlap_max,overlap)
            for gt in ann:
                if gt['cls']==14:
                    mask=_rectangle(before.shape[:2],gt['bbox'])[0,0].numpy().astype(bool)
                    assert absolute[mask].any();changed+=1
            targets+=1
        else:
            assert sha(source)==row['output_sha256'];non_targets+=1
        if (index+1)%1000==0:
            write(out/'progress.json',{'state':'running','train_audited':index+1,'target_images':targets})
            print(json.dumps({'train_audited':index+1,'elapsed_seconds':time.monotonic()-started}),flush=True)
    assert (targets,non_targets,changed)==(6095,10456,13256)
    cleanval=[Path(p) for p in list_images(str(clean/'images/val'))]
    exportedval=[Path(p) for p in list_images(str(data/'images/val'))]
    assert len(cleanval)==len(exportedval)==4952
    assert {p.name for p in cleanval}=={p.name for p in exportedval}
    assert len({p.stem for p in exportedval})==4952
    assert {p.stem for p in (data/'labels/val').glob('*.txt')}=={p.stem for p in cleanval}
    for p in cleanval:
        assert sha(p)==sha(data/'images/val'/p.name)
        label=clean/'labels/val'/(p.stem+'.txt');assert sha(label)==sha(data/'labels/val'/label.name)
    splits={}
    for split in ('train','val'):
        paths=[Path(p) for p in list_images(str(clean/'images'/split))]+list((clean/'labels'/split).glob('*.txt'))
        splits[split]=split_fingerprint(paths)
    combined=hashlib.sha256((splits['train']+'\n'+splits['val']).encode()).hexdigest()
    assert combined==lock['dataset_fingerprint']
    # S0 checkpoint is read on CPU only; no model inference or transfer.
    provenance=json.loads((ROOT/'docs/S4_TFSS_UTILITY/PARENT_IDENTITY.json').read_text())
    s0checkpoint=Path(lock['victim_protocol']['train_args']['project'])/'victim/weights/last.pt'
    assert sha(s0checkpoint)==provenance['victim_last_sha256']
    packed=torch.load(s0checkpoint,map_location='cpu',weights_only=False)
    site=Path(sys.prefix)/'lib/python3.11/site-packages';native=site/'ultralytics/engine/trainer.py'
    text=native.read_text();lines=text.splitlines();tree=ast.parse(text)
    methods={node.name:{'line':node.lineno,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])} for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in ('optimizer_step','_handle_nan_recovery')}
    slices=[]
    for i,line in enumerate(lines):
        if any(x in line for x in ('self.optimizer_step()','on_train_batch_end','on_train_start','self.scaler.scale(self.loss)','on_train_epoch_start')):
            slices.append({'line':i+1,'source':'\n'.join(lines[max(0,i-5):i+7])})
    runtime={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'native_trainer_sha256':sha(native),
             'native_methods':methods,'callback_locations':slices,'S0_checkpoint_version':packed.get('version'),
             'S0_checkpoint_date':packed.get('date'),'S0_train_args':packed.get('train_args'),
             'S0_checkpoint_sha256':sha(s0checkpoint),'historical_optimizer_source_sha256':None,
             'historical_source_limit':'contemporaneous installed trainer source not retained; checkpoint version and archived launcher/protocol are provenance'}
    assert runtime['torch']==provenance['environment']['torch'] and runtime['ultralytics']==provenance['environment']['ultralytics']
    write(out/'runtime_and_S0_provenance.json',runtime)
    result={'state':'passed','read_only_input':True,'no_noise_regeneration':True,'GPU_forward_calls':0,'source45_passed':True,
            'train':16551,'val':4952,'target_images':targets,'changed_Person_GT':changed,'non_target_images_byte_identical':non_targets,
            'labels_byte_identical':True,'clean_validation_byte_identical':True,'output_manifest_hashes_verified':True,
            'duplicate_or_missing_IDs':0,'outside_max_byte':outside_max,'global_max_byte':global_max,'overlap_max_byte':overlap_max,
            'original_dataset_fingerprint':combined,'noise_checkpoint_sha256':sha(checkpoint),'dataset_integrity_sha256':sha(integrity),
            'poisoned_dataset_manifest_sha256':sha(manifest),'elapsed_seconds':time.monotonic()-started}
    write(out/'dataset_reuse_audit.json',result);write(out/'status.json',{'state':'complete','stage':'CPU_dataset_reuse_audit','result':result})
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
