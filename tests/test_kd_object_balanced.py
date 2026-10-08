import unittest

import torch

from ue_framework.methods.kd_utility import object_balanced_mean


class ObjectBalancedKDTests(unittest.TestCase):
    def test_objects_get_equal_weight_despite_unequal_positive_counts(self):
        values = torch.tensor([1., 1., 1., 1., 4.], requires_grad=True)
        ids = torch.tensor([3, 3, 3, 3, 8])
        loss, objects, means, counts = object_balanced_mean(values, ids)
        torch.testing.assert_close(loss, torch.tensor(2.5))
        self.assertEqual(objects.tolist(), [3, 8])
        self.assertEqual(counts.tolist(), [4, 1])
        torch.testing.assert_close(means, torch.tensor([1., 4.]))
        self.assertEqual(torch.autograd.grad(loss, values)[0].tolist(),
                         [.125, .125, .125, .125, .5])

    def test_zero_positive_objects_are_excluded_not_zero_filled(self):
        values = torch.tensor([2., 4.], requires_grad=True)
        loss, objects, means, counts = object_balanced_mean(values, torch.tensor([5, 5]))
        torch.testing.assert_close(loss, torch.tensor(3.))
        self.assertEqual(objects.tolist(), [5])
        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(means.tolist(), [3.])

    def test_equal_positive_counts_match_the_old_unit_mean(self):
        values = torch.arange(1., 21.)
        ids = torch.tensor([2] * 10 + [7] * 10)
        balanced, _, _, _ = object_balanced_mean(values, ids)
        torch.testing.assert_close(balanced, values.mean())


if __name__ == "__main__":
    unittest.main()
