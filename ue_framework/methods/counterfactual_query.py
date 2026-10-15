"""Deterministic train-only query schedule, independently checkpoint-bound.

The manifest is built/audited against the dataset train split before admission.
Only explicit records with NT GT are eligible; never select by validation AP.
"""
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from ..data_utils import load_image_rgb_float,read_yolo_annotations
from .csb_budget import digest


class TrainQueryProvider:
    def __init__(self, manifest_path, manifest_sha256, adaptive, batch_size, seed):
        if batch_size < 1:
            raise ValueError('Explicit positive train-query batch size required')
        if digest(manifest_path) != manifest_sha256:
            raise ValueError('Train query manifest SHA mismatch')
        self.manifest_sha256=manifest_sha256
        self.adaptive=adaptive
        self.batch_size=batch_size
        self.seed=seed
        manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if manifest.get('split') != 'train' or manifest.get('status') != 'AUDITED_TRAIN_ONLY':
            raise ValueError('Requires a verified train-only query manifest')
        self.records=manifest['records']
        self.train_ids=set(manifest['full_train_image_ids'])
        self.val_ids=set(manifest['validation_image_ids'])
        if self.train_ids & self.val_ids:
            raise ValueError('Dataset train/validation identity leakage')
        seen=set()
        for row in self.records:
            ident=row['image_id']
            if ident in seen or ident not in self.train_ids or ident in self.val_ids:
                raise ValueError('Query identity is not a unique train member')
            seen.add(ident)
            if not row['non_target_gt_count'] > 0:
                raise ValueError('Query manifest requires valid non-target GT')
        if not seen:
            raise ValueError('Empty train-query manifest')

    def __call__(self,support_ids,step):
        forbidden=set(support_ids)
        # Canonical identities MUST be used by the outer loop and manifest.
        if not forbidden <= self.train_ids or forbidden & self.val_ids:
            raise ValueError('Outer support identities are not audited train members')
        eligible=[r for r in self.records if r['image_id'] not in forbidden]
        if len(eligible)<self.batch_size:
            raise ValueError('Insufficient disjoint NT train-query images')
        order=random.Random(f'{self.seed}:{step}:{self.manifest_sha256}').sample(eligible,self.batch_size)
        items=[]
        for row in order:
            if digest(row['label_path']) != row['label_sha256'] or digest(row['image_path']) != row['image_sha256']:
                raise ValueError('Query dataset bytes changed after locking')
            image=load_image_rgb_float(row['image_path'])
            anns=read_yolo_annotations(row['label_path'])
            count=sum(int(a['cls']) != self.adaptive.target_class_id for a in anns)
            if count != row['non_target_gt_count']:
                raise ValueError('Query non-target GT coverage changed')
            items.append((image,anns,None))
        h=(max(im.shape[0] for im,_,_ in items)+31)//32*32
        w=(max(im.shape[1] for im,_,_ in items)+31)//32*32
        images=torch.stack([F.pad(torch.from_numpy(im).permute(2,0,1).float(),
                                   (0,w-im.shape[1],0,h-im.shape[0]),value=.5)
                            for im,_,_ in items]).to(next(self.adaptive.model.parameters()).device)
        batch=self.adaptive._batch(images,items)
        batch.update(split='train',image_ids=[r['image_id'] for r in order],
                     preprocessing_id='Parent_padded_letterbox_clean_train_query')
        return batch

    def state_dict(self):
        # Stateless schedule: future batches depend on restored outer step,
        # actual support identities, manifest SHA, batch size and seed only.
        return dict(manifest_sha256=self.manifest_sha256,batch_size=self.batch_size,seed=self.seed)

    def load_state_dict(self,state):
        if state != self.state_dict():
            raise ValueError('Train query resume identity changed')
