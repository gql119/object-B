import copy,unittest
import numpy as np
from ultralytics.utils.instance import Instances
from robustness_s4_extended.integration import native_absolute_boxes,CutoutTransform

class NativeBoundaryRoundoffTests(unittest.TestCase):
    def labels(self,boxes,fmt='xyxy',normalized=False):
        return dict(img=np.zeros((640,640,3),np.uint8),instances=Instances(np.array(boxes,np.float32).reshape(-1,4),bbox_format=fmt,normalized=normalized),cls=np.full((len(boxes),1),14,np.float32))
    def test_actual_reproduced_float32_boundary(self):
        x=self.labels([[-3.0517578125e-05,493.0758056640625,182.14382934570312,640.]])
        before=copy.deepcopy(x['instances']);b,a=native_absolute_boxes(x)
        self.assertEqual(b[0,0],0);self.assertEqual(a['boundary_roundoff_max_pixels'],3.0517578125e-05)
        np.testing.assert_array_equal(before.bboxes,x['instances'].bboxes)
    def test_native_xywh_roundtrip(self):
        x=self.labels([[0.,493.0758056640625,182.14382934570312,640.]])
        x['instances'].convert_bbox('xywh');before=x['instances'].bboxes.copy()
        b,a=native_absolute_boxes(x);self.assertTrue((b>=0).all());np.testing.assert_array_equal(before,x['instances'].bboxes)
    def test_real_oob_and_degenerate_still_rejected(self):
        for b in [[-.01,10,20,30],[10,10,640.01,30],[20,10,20,30],[30,10,20,30]]:
            with self.assertRaises(ValueError):native_absolute_boxes(self.labels([b]))
    def test_empty_and_normalized(self):
        b,_=native_absolute_boxes(self.labels([]));self.assertEqual(b.shape,(0,4))
        x=self.labels([[0,.5,1,1]],normalized=True);before=x['instances'].bboxes.copy();b,_=native_absolute_boxes(x)
        np.testing.assert_array_equal(b,[[0,320,640,640]]);np.testing.assert_array_equal(before,x['instances'].bboxes)
    def test_cutout_retains_original_gt_bytes(self):
        x=self.labels([[-3.0517578125e-05,493.0758056640625,182.14382934570312,640.]])
        before=x['instances'].bboxes.tobytes()
        cfg=dict(n_holes=1,area_fraction=[.25,.25],probability=1.,fill_value=0,max_gt_occlusion=.5,max_attempts=8)
        r=CutoutTransform(cfg)(x);self.assertEqual(before,r['instances'].bboxes.tobytes());self.assertTrue(r['_extended_audit']['original_training_GT_unchanged'])
