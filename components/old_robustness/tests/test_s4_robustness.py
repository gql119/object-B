import unittest
import numpy as np
import cv2
from ue_framework.s4_robustness import transform_rgb,paired_scores

class Contracts(unittest.TestCase):
    def test_frozen_parameter_mismatch_is_rejected(self):
        x=np.zeros((32,32,3),dtype=np.uint8)
        with self.assertRaises(AssertionError):transform_rgb(x,'jpeg50',dict(quality=30))
        with self.assertRaises(AssertionError):transform_rgb(x,'bdr',dict(bits=4))
        with self.assertRaises(AssertionError):transform_rgb(x,'gauss',dict(kernel=[3,3],sigma=1))

    def test_defenses_geometry_and_legacy_filters(self):
        x=np.random.default_rng(11).integers(0,256,(43,57,3),dtype=np.uint8)
        for d in ['jpeg50','jpeg30','gray','bdr','gauss','meanf','medianf','none','data_aug']:
            y=transform_rgb(x,d)
            self.assertEqual(y.shape,x.shape);self.assertEqual(y.dtype,x.dtype)
            np.testing.assert_array_equal(y,transform_rgb(x,d))
        for d,reference in [('gauss',cv2.GaussianBlur(x,(3,3),0)),('meanf',cv2.blur(x,(3,3))),('medianf',cv2.medianBlur(x,3))]:
            np.testing.assert_array_equal(transform_rgb(x,d),reference)
        y=transform_rgb(x,'gray');np.testing.assert_array_equal(y[:,:,0],y[:,:,1]);np.testing.assert_array_equal(y[:,:,1],y[:,:,2])

    def test_bdr_all_uint8_inputs(self):
        x=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
        y=transform_rgb(x,'bdr');self.assertEqual(len(np.unique(y)),32)
        z=x.astype(np.float32)/255;expected=np.clip(np.round(np.round(z*31)/31*255),0,255).astype(np.uint8)
        np.testing.assert_array_equal(y,expected);np.testing.assert_array_equal(transform_rgb(y,'bdr'),y)

    def test_scores_signed_and_undefined(self):
        c=dict(mAP50_target=.8,mAP50_non_target=.7,AP50_per_class=[.7]*20)
        p=dict(mAP50_target=.04,mAP50_non_target=.63,AP50_per_class=[.63]*20)
        s=paired_scores(c,p);self.assertAlmostEqual(s['P'],.95);self.assertAlmostEqual(s['U'],.9)
        self.assertAlmostEqual(s['S'],2*.95*.9/(.95+.9));self.assertAlmostEqual(s['WCD'],.1)
        p['mAP50_target']=.9;self.assertLess(paired_scores(c,p)['P'],0)
        c['mAP50_target']=0;self.assertIsNone(paired_scores(c,p)['S'])
        with self.assertRaises(ValueError):paired_scores(dict(c,mAP50_target=80),p)

if __name__=='__main__':unittest.main()
