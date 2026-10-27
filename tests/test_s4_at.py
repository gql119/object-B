import copy
import unittest
import torch
from robustness_s4_at.common import protocol, run_config
from robustness_s4_at.pgd import AttackConfig, safe_interval, validate_batch
from robustness_s4_at.metrics import pair


class ATContracts(unittest.TestCase):
    def test_fixed_budgets_and_no_silent_protocol_variants(self):
        for eps, alpha in ((4, 1), (8, 2)):
            AttackConfig(eps, alpha)
        for values in ((4, 2), (8, 1), (16, 4)):
            with self.assertRaises(ValueError): AttackConfig(*values)
        with self.assertRaises(ValueError): AttackConfig(4, 1, steps=9)
        with self.assertRaises(ValueError): AttackConfig(4, 1, random_start=False)

    def test_strict_interval_all_decoded_levels_and_arbitrary_floats(self):
        rng = torch.Generator().manual_seed(7)
        x = torch.cat((torch.arange(256, dtype=torch.float32) / 255,
                       torch.rand(10000, generator=rng)))
        for e in (4, 8):
            lo, hi = safe_interval(x, e / 255)
            self.assertLessEqual(float((x.double() - lo.double()).max()), e / 255)
            self.assertLessEqual(float((hi.double() - x.double()).max()), e / 255)
            self.assertTrue(bool((lo <= x).all() and (hi >= x).all()))
            self.assertGreaterEqual(float(lo.min()), 0)
            self.assertLessEqual(float(hi.max()), 1)

    def test_all_classes_and_legal_empty_GT(self):
        b = dict(img=torch.zeros(1, 3, 32, 32), cls=torch.arange(20).view(-1, 1).float(),
                 batch_idx=torch.zeros(20), bboxes=torch.tensor([[.5, .5, .1, .1]]).repeat(20, 1))
        validate_batch(b, 20)
        b.update(cls=torch.empty(0, 1), batch_idx=torch.empty(0), bboxes=torch.empty(0, 4))
        validate_batch(b, 20)

    def test_bad_GT_and_input_rejected(self):
        base = dict(img=torch.zeros(1, 3, 32, 32), cls=torch.tensor([[14.]]),
                    batch_idx=torch.zeros(1), bboxes=torch.tensor([[.5, .5, .1, .1]]))
        for key, value in (('cls', torch.tensor([[20.]])), ('cls', torch.tensor([[14.5]])),
                           ('batch_idx', torch.ones(1)), ('bboxes', torch.tensor([[.5, .5, 0., .1]])),
                           ('img', torch.full((1, 3, 32, 32), float('nan')))):
            with self.assertRaises(ValueError): validate_batch(dict(base, **{key: value}), 20)

    def test_four_runs_only_budget_and_arm_differ(self):
        p = protocol()
        runs = [run_config(n) for n in p['order']]
        self.assertEqual([r['arm'] for r in runs], ['clean', 'poison', 'clean', 'poison'])
        self.assertEqual(runs[0]['attack'], runs[1]['attack'])
        self.assertEqual(runs[2]['attack'], runs[3]['attack'])
        self.assertEqual(len({r['protocol_sha256'] for r in runs}), 1)

    @staticmethod
    def metric(v):
        return dict(class_ids=list(range(20)), validation_count=4952,
                    **{k: [v] * 20 for k in ('AP50', 'AP75', 'AP50_95', 'Recall', 'Precision')})

    def test_matched_scores_and_absolute_differences(self):
        c, p = self.metric(.8), self.metric(.4)
        r = pair(c, p)
        self.assertAlmostEqual(r['P_AT'], .5)
        self.assertAlmostEqual(r['U_AT'], .5)
        self.assertAlmostEqual(r['S_AT'], .5)
        self.assertAlmostEqual(r['Person_change_pp'], -40)

    def test_invalid_denominators_not_zero_imputed(self):
        r = pair(self.metric(0), self.metric(.1))
        self.assertIsNone(r['P_AT']); self.assertIsNone(r['U_AT']); self.assertIsNone(r['S_AT'])
        self.assertIsNone(r['WCD']); self.assertIsNone(r['TCD3'])

    def test_signed_scores_are_not_clipped(self):
        r = pair(self.metric(.4), self.metric(.8))
        self.assertAlmostEqual(r['P_AT'], -1)
        self.assertAlmostEqual(r['U_AT'], 2)
        self.assertAlmostEqual(r['S_AT'], -4)


if __name__ == '__main__':
    unittest.main()
