import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.prepare_counterfactual_inputs import train_manifest
from ue_framework.methods.adaptive_learner import AdaptiveLearner
from ue_framework.methods.counterfactual_query import TrainQueryProvider
from ue_framework.methods.csb_budget import digest


def inputs(root):
    images=root/'train'; labels=root/'labels'; validation=root/'val'
    for p in (images,labels,validation): p.mkdir()
    for j in range(3):
        Image.fromarray(np.full((32,32,3),j+128,dtype=np.uint8)).save(images/f'train{j}.png')
        (labels/f'train{j}.txt').write_text('1 0.5 0.5 0.5 0.5\n14 0.7 0.7 0.2 0.2\n')
    Image.fromarray(np.full((32,32,3),128,dtype=np.uint8)).save(validation/'val.png')
    return images,labels,validation


def test_manifest_query_disjointness_and_stateless_resume(tmp_path):
    images,labels,validation=inputs(tmp_path)
    manifest=train_manifest(images,labels,validation)
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    adaptive=AdaptiveLearner.__new__(AdaptiveLearner)
    adaptive.imgsz=64; adaptive.target_class_id=14; adaptive.model=torch.nn.Linear(1,1)
    a=TrainQueryProvider(path,digest(path),adaptive,1,0)
    b=TrainQueryProvider(path,digest(path),adaptive,1,0); b.load_state_dict(a.state_dict())
    support=[str((images/'train0.png').resolve())]
    x,y=a(support,7),b(support,7)
    assert x['image_ids']==y['image_ids'] and not set(support)&set(x['image_ids'])
    torch.testing.assert_close(x['img'],y['img'],rtol=0,atol=0)
    assert all('val.png' not in s for s in x['image_ids'])
    chosen=next(r for r in manifest['records'] if r['image_id']==x['image_ids'][0])
    Path(chosen['label_path']).write_text('2 0.5 0.5 0.5 0.5\n')
    with pytest.raises(ValueError,match='bytes changed'): a(support,7)


def test_manifest_rejects_label_silent_skip(tmp_path):
    images,labels,validation=inputs(tmp_path)
    (labels/'train0.txt').write_text('1 0.5\n')
    with pytest.raises(ValueError,match='Malformed'): train_manifest(images,labels,validation)


def test_manifest_rejects_validation_identity_leakage(tmp_path):
    images,labels,validation=inputs(tmp_path)
    Image.fromarray(np.full((32,32,3),128,dtype=np.uint8)).save(validation/'train0.png')
    with pytest.raises(ValueError,match='overlap'): train_manifest(images,labels,validation)
