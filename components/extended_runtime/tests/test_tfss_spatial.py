"""CPU numeric contracts; no VOC/model/optimizer-result claims."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch

from ue_framework.methods.tfss_spatial import (
    normalize_map, spatial_weights, match_realized_rms, realized_rms,
    SoftSpatialProjection, target_union,
)


class TFSSContracts(unittest.TestCase):
    def test_exact_formulas_and_no_nt(self):
        q = torch.tensor([[0., .2, .8, 1.]])
        r = torch.tensor([[1., .7, .4, 0.]])
        s = torch.tensor([[1., .5, 1., 0.]])
        c = spatial_weights(q, r, s, mode='TFSS-C')
        u = spatial_weights(q, r, s, mode='TFSS-U')
        torch.testing.assert_close(c[:, :3], .85 + .15*q[:, :3])
        torch.testing.assert_close(u[:, :3], 1-.15*r[:, :3]*(1-q[:, :3]))
        self.assertEqual(c[0, 3], 1)
        self.assertTrue(torch.equal(spatial_weights(q, r*0, s, mode='TFSS-U'), torch.ones_like(q)))

    def test_masked_smoothing_and_degenerate_policy(self):
        mask = torch.zeros(5, 5, dtype=torch.bool);mask[1:4, 1:4] = True
        a = torch.ones(5, 5);a[~mask] = 1e9
        q, record = normalize_map(a, mask, constant_policy='preserve')
        self.assertFalse(record['valid'])
        self.assertTrue(torch.equal(q, mask.float()))
        self.assertFalse(q.requires_grad)

    def test_multi_person_max_no_amplitude_addition(self):
        m = torch.ones(2, 3, dtype=torch.bool)
        q = target_union([(m*.3, m), (m*.9, m)], m)
        torch.testing.assert_close(q, m*.9)

    def test_actual_rms_with_saturation(self):
        image = torch.tensor([[[.999, .4, .001, .2]]]).expand(3, -1, -1)
        delta = torch.tensor([[[.03, -.02, -.03, .01]]]).expand_as(image)
        support = torch.tensor([[[1., .5, 1., 0.]]])
        delta = delta * (support > 0)
        desired = float(realized_rms(image, delta*.877, support))
        alpha, error = match_realized_rms(image, delta, support, desired)
        self.assertLess(error, 1e-9)
        self.assertAlmostEqual(alpha, .877, places=5)
        with self.assertRaises(ValueError):
            match_realized_rms(image, delta, support, 1.)

    def test_composition_budget_and_adaptive_cache(self):
        class Parent:
            def _compose_delta_batched(self, image, support, ring):
                delta = torch.full_like(image, .031372549)*support
                return delta, delta, (image+delta).clamp(0, 1), support > 0, support
        class Candidate(SoftSpatialProjection, Parent):
            pass
        c = Candidate();x = torch.full((1, 3, 2, 3), .5)
        s = torch.tensor([[[[1., .5, 0.], [1., .5, 0.]]]])
        original = c._compose_delta_batched(x, s, s*0)
        c.tfss_mode = 'TFSS-C';c.tfss_weight = torch.full_like(s, .85)
        result = c._compose_delta_batched(x, s, s*0)
        torch.testing.assert_close(result[1], original[1]*.85)
        self.assertIs(c._last_poison, result[2])
        self.assertEqual(float(result[1][..., 2].abs().max()), 0)
        self.assertLessEqual(float(result[1][..., 1].abs().max()), .031372549*.5)

    def test_float32_staircase_and_both_tolerances(self):
        image = torch.full((3, 1, 1), .5)
        delta = torch.full_like(image, .01)
        support = torch.ones(1, 1, 1)
        target = .00500001  # Between adjacent attained FP32 image increments.
        alpha, error = match_realized_rms(image, delta, support, target)
        self.assertGreaterEqual(alpha, 0.)
        self.assertLessEqual(alpha, 1.)
        self.assertLessEqual(error, 1e-7)
        self.assertLessEqual(error/target, 1e-4)
        self.assertEqual(error, abs(float(realized_rms(image,
            delta*delta.new_tensor(alpha), support))-target))
        with self.assertRaises(ValueError):
            match_realized_rms(image, delta, support, target, tolerance=1e-12)
        # A tiny target can pass the absolute gate but fail the relative gate.
        with self.assertRaises(ValueError):
            match_realized_rms(image, delta, support, 1e-9)

    def test_valid_small_GT_preserves_strength_without_attribution(self):
        import numpy as np
        from unittest.mock import patch
        from ue_framework.methods.tfss_clean_maps import build_maps
        from ue_framework.methods.legacy_kproto_ret import _rectangle
        image = np.linspace(.1, .9, 64*64*3, dtype=np.float32).reshape(64, 64, 3)
        ann = {'cls': 14, 'bbox': [.5, .5, .1, .1]}
        domain = _rectangle((64, 64), ann['bbox'])[0, 0].bool()
        class Trainer:
            device = torch.device('cpu')
            channels = torch.tensor([0])
            surrogate = object()
            imgsz = 640
            def _feature(self, image, grad):
                return torch.nn.functional.adaptive_avg_pool2d(image.mean(1, keepdim=True), (20, 20))
            def _masks(self, *args):
                return [torch.ones(1, 1, 20, 20)*(.25/400)], [], []
        with patch('ue_framework.methods.tfss_clean_maps.gradients', return_value=({}, [])):
            maps, records = build_maps(Trainer(), [(image, [ann], domain.float().numpy())])
        q, risk, valid = maps[0]
        self.assertEqual(len(records[0]['target_objects']), 1)  # GT retained.
        self.assertEqual(records[0]['target_objects'][0]['reason'], 'canonical_P5_pool_undercoverage')
        self.assertIsNone(records[0]['target_objects'][0]['raw_norm'])
        self.assertFalse(bool(valid[domain].any()))
        for mode in ['TFSS-C', 'TFSS-U']:
            weight = spatial_weights(q, risk, domain.float(), mode=mode)
            self.assertTrue(torch.equal(weight[domain], torch.ones_like(weight[domain])))

    def test_soft_gradient_and_detached_maps(self):
        coeff = torch.tensor(.025, requires_grad=True)
        q = torch.tensor([[0., 1.]], requires_grad=True)
        s = torch.tensor([[1., .5]])
        w = spatial_weights(q, q*0, s, mode='TFSS-C')
        value = (coeff*s*w).sum()
        grad = torch.autograd.grad(value, coeff)[0]
        torch.testing.assert_close(grad, (s*w).sum())
        self.assertFalse(w.requires_grad)
        self.assertIsNone(q.grad)

    def test_invalid_maps_fail(self):
        with self.assertRaises(ValueError):
            normalize_map(torch.full((2, 2), float('nan')), torch.ones(2, 2, dtype=torch.bool), constant_policy='preserve')
        with self.assertRaises(ValueError):
            spatial_weights(torch.ones(2, 2)*2, torch.zeros(2, 2), torch.ones(2, 2), mode='TFSS-U')

    def test_higher_order_diagnostic_does_not_consume_backward(self):
        from ue_framework.methods.tfss_probe_metrics import collect
        coeff = torch.tensor(.01, requires_grad=True)
        x = torch.full((1, 3, 2, 3), .5)
        support = torch.tensor([[[[1., .5, 0.], [1., .5, 0.]]]])
        delta = coeff*support.expand_as(x)
        poison = (x+delta).clamp(0, 1)
        theta = torch.tensor(.2, requires_grad=True)
        inner = (theta*poison.mean()-.3).square()
        updated = theta-.0001*torch.autograd.grad(inner, theta, create_graph=True)[0]
        adaptive = (updated*.5).square()
        bg = poison.square().mean();rel = poison.mean();ret = (poison-x).square().mean()
        total = bg+rel+ret+adaptive
        reference_grad = torch.autograd.grad(total, coeff, retain_graph=True)[0]
        t = SimpleNamespace(lambda_adapt=1., _last_poison=poison, fourier_coeff=coeff,
            tfss_input=x, tfss_delta=delta, tfss_s0_delta=delta, tfss_support=support,
            tfss_applied_weight=torch.ones_like(support), eps=.031372549)
        metrics = {'weighted_L_BG': bg, 'weighted_L_rel': rel, 'weighted_L_ret': ret,
            'L_adapt_target': adaptive, 'target_score_poison_after': adaptive.detach(),
            'target_score_clean_after': adaptive, 'inner_update_norm': theta.new_tensor(.001)}
        owned = Path(__file__).resolve().parents[1]/'execution/cpu_tests'
        owned.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='contract_', dir=owned))  # retain; no deletion
        result = collect(t, metrics, [(torch.zeros(2, 3, 3).numpy(), [], support[0, 0].numpy())],
            [(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool))], mode='S0',
            reference_path=folder/'reference.pt', output_prefix=folder/'output')
        self.assertAlmostEqual(result[0]['protection_gradient_cosine_to_s0'], 1.)
        self.assertEqual(result[0]['high_protection_operator_energy_retention'], 1.)
        total.backward()
        torch.testing.assert_close(coeff.grad, reference_grad)
        self.assertGreater(float(coeff.grad.abs()), 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
