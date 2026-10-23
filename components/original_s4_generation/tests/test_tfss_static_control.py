import unittest
import torch
from ue_framework.methods.tfss_spatial import match_realized_rms, realized_rms
from ue_framework.methods.tfss_static_diagnostics import feasibility, spatial_selectivity


class StaticControls(unittest.TestCase):
    def test_independent_unreachable_same_base_reachable_after_clamp(self):
        x=torch.tensor([[[0.,.999, .2, .8]]]).expand(3,-1,-1).clone()
        base=torch.tensor([[[-.02,.02,.03,-.03]]]).expand_as(x).clone()
        mask=torch.ones(1,4);w=torch.tensor([[.85,1.,.9,.98]])
        target=float(realized_rms(x,base*w,mask))
        self.assertTrue(feasibility(x,base*.5,mask,target)['unreachable_at_locked_tolerance'])
        with self.assertRaisesRegex(ValueError,'exceeds own S0 maximum'):
            match_realized_rms(x,base*.5,mask,target)
        a,e=match_realized_rms(x,base,mask,target)
        self.assertTrue(0<=a<=1);self.assertLessEqual(e,1e-7)
        self.assertLessEqual(e/target,1e-4)

    def test_global_RMS_does_not_imply_same_spatial_energy(self):
        x=torch.full((3,1,4),.5);base=torch.full_like(x,.02)
        mask=torch.ones(1,4);q=torch.tensor([[1.,1.,0.,0.]])
        risk=1-q;u=base*(1-.15*risk*(1-q))
        a,_=match_realized_rms(x,base,mask,float(realized_rms(x,u,mask)))
        stats=spatial_selectivity(x,base,u,base*a,mask,q,risk,mask.bool())
        self.assertAlmostEqual(stats['person']['Utility_RMS'],stats['person']['Uniform_RMS'],places=7)
        self.assertGreater(stats['high_Q']['Utility_energy_retention'],stats['high_Q']['Uniform_energy_retention'])
        self.assertLess(stats['high_R_low_Q']['Utility_energy_retention'],stats['high_R_low_Q']['Uniform_energy_retention'])

    def test_absent_regions_remain_undefined_no_nt_identity(self):
        x=torch.zeros(3,2,2);base=torch.ones_like(x)*.01
        m=torch.tensor([[1.,.5],[0.,1.]]);base*=m
        q=torch.ones_like(m);r=torch.zeros_like(m)
        a,e=match_realized_rms(x,base,m,float(realized_rms(x,base,m)))
        self.assertEqual(a,1.);self.assertEqual(e,0.)
        s=spatial_selectivity(x,base,base,base*a,m,q,r,m.bool())
        self.assertIsNone(s['high_R']['Utility_RMS'])
        self.assertIsNone(s['R_weighted']['Utility_energy_retention'])


if __name__=='__main__':unittest.main()
