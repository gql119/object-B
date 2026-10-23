"""Real CPU random-YAML native head/loss probes; not a formal pretrained teacher."""
import runpy,unittest
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ue_framework.methods.tfss_nt_sensitivity import gradients

ROOT=Path(__file__).resolve().parents[2]

class NativeSensitivity(unittest.TestCase):
    def model(self,nc):
        torch.manual_seed(21)
        m=DetectionModel(str(ROOT/'configs/voc_yolov8n_20cls.yaml'),nc=nc,verbose=False).eval()
        m.args = get_cfg()
        for p in m.parameters():p.requires_grad_(False)
        return m
    def inputs(self,target,other):
        im=np.random.default_rng(1).random((96,160,3),dtype=np.float32)
        return im,[{'cls':target,'bbox':[.3,.5,.35,.5]},{'cls':other,'bbox':[.7,.5,.4,.5]}]
    def test_COCO_Person0_NT79_native_loss_real_input_gradient(self):
        m=self.model(80);im,ann=self.inputs(0,79)
        maps,records=gradients(m,im,ann,imgsz=64,target_class_id=0,num_classes=80,coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
        self.assertEqual(set(maps),{79});self.assertEqual(maps[79].shape,(96,160));self.assertTrue(np.isfinite(maps[79]).all());self.assertGreater(float(np.linalg.norm(maps[79])),0)
        self.assertEqual({x['cls'] for x in records},{79});self.assertTrue(all(p.grad is None for p in m.parameters()))
    def test_VOC_real_gradient_and_matching_exact_old_implementation(self):
        old=ROOT/'coco_s4/tests/reference_tfss_nt_sensitivity_voc.py'
        ref=runpy.run_path(str(old))['gradients'];m=self.model(20);im,ann=self.inputs(14,19)
        a,ar=ref(m,im,ann,imgsz=64,coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
        b,br=gradients(m,im,ann,imgsz=64,coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
        self.assertEqual(ar,br);self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))
    def test_no_NT_and_wrong_class_schema(self):
        m=self.model(80);im,ann=self.inputs(0,79)
        self.assertEqual(gradients(m,im,ann[:1],target_class_id=0,num_classes=80),({},[]))
        with self.assertRaises(ValueError):gradients(m,im,ann,target_class_id=0,num_classes=20)

if __name__=='__main__':unittest.main()
