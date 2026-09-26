import unittest
import torch
from ue_framework.methods.semantic_coordinates import select_channels, class_subspace, normalized_orth, gt_evidence
from ue_framework.methods.adaptive_learner import AdaptiveLearner


class SemanticTests(unittest.TestCase):
    def test_tsfl_and_gt_evidence(self):
        st = torch.tensor([4., 4., 1., 1.]); sn = torch.ones(4)
        indices, _, _, _ = select_channels(st, sn, .5)
        self.assertEqual(indices.tolist(), [0, 1])
        # Production alignment must exactly agree with the existing B2 evidence primitive.
        a = type('Evidence', (), {'target_class_id':14, 'imgsz':640, '_iou':staticmethod(AdaptiveLearner._iou)})()
        decoded = torch.rand(1,24,120); decoded[:,:4] *= 640
        anns = [{'cls':14,'bbox':[.5,.4,.2,.3]}]
        val = gt_evidence(decoded, anns, (320,640),640)[0][1]
        batch = {'batch_idx':torch.tensor([0]), 'cls':torch.tensor([[14.]]),
                 'bboxes':torch.tensor([[.5,.45,.2,.15]])}
        torch.testing.assert_close(val, AdaptiveLearner._target_evidence(a, decoded, batch))

    def test_pca_rank_and_orth(self):
        torch.manual_seed(0)
        sample = list(torch.randn(20,8))
        pca = class_subspace(sample, .9, 4, .5)
        self.assertEqual(pca['rank'],4)
        torch.testing.assert_close(pca['basis'].T@pca['basis'],torch.eye(4),atol=1e-5,rtol=1e-5)
        displacement = torch.tensor([[1.,2.,3.]], requires_grad=True)
        bases={0:torch.eye(3)[:,:1],1:torch.eye(3)[:,1:2],2:torch.eye(3)[:,2:3]}
        loss, conflicts, worst = normalized_orth(displacement,bases,torch.tensor([2.]),2)
        self.assertEqual(worst,[2,1])
        torch.testing.assert_close(loss,torch.tensor(13/14))
        self.assertGreater(float(torch.autograd.grad(loss,displacement)[0].norm()),0.)

    def test_baseline_files_identical(self):
        import hashlib
        from pathlib import Path
        root=Path(__file__).resolve().parent.parent
        manifest=root/'b2_frozen_source_manifest.json'
        import json
        for name,expected in json.loads(manifest.read_text()).items():
            actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
            self.assertEqual(actual,expected,name)


if __name__=='__main__':
    unittest.main()
