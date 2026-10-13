import unittest

import numpy as np

from scripts.generate_fidelity_baseline import (
    BUDGET_CYCLE_FACTOR,
    BUDGET_FINAL_RESERVE_SECONDS,
    BUDGET_REFERENCE_CYCLE_SECONDS,
    budget_allows_next_cycle,
)
from ue_framework.baselines_det.fidelity import (
    PersistentShuffledSampler,
    decoded_fidelity_pixel_audit,
    fidelity_support_mask,
    validate_spatial_protocol_lock,
)


class FidelityProtocolTest(unittest.TestCase):
    def test_protocol_lock_must_match_full_bbox8_or_half_overlap(self):
        lock = {"poison": {
            "epsilon": "8/255", "overlap_scale": 1.0,
            "overlap_effective_epsilon": "8/255", "outside_target_epsilon": 0,
            "support": "person_box_union",
        }}
        validate_spatial_protocol_lock(lock, "new_full_bbox8")
        with self.assertRaisesRegex(ValueError, "overlap_scale"):
            validate_spatial_protocol_lock(
                {"poison": {**lock["poison"], "overlap_scale": 0.5,
                            "overlap_effective_epsilon": "4/255"}},
                "new_full_bbox8",
            )
        half = {"poison": {**lock["poison"], "overlap_scale": 0.5,
                           "overlap_effective_epsilon": "4/255"}}
        validate_spatial_protocol_lock(half, "old_overlap_half")

    def test_full_bbox_overlap_and_outside(self):
        anns = [
            {"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]},
            {"cls": 11, "bbox": [0.625, 0.625, 0.5, 0.5]},
        ]
        old = fidelity_support_mask(8, 8, anns, "old_overlap_half")
        new = fidelity_support_mask(8, 8, anns, "new_full_bbox8")
        self.assertEqual(new[4, 4], 1.0)
        self.assertEqual(old[4, 4], 0.5)
        self.assertEqual(new[1, 1], 0.0)
        self.assertEqual(new[6, 6], 0.0)
        np.testing.assert_array_equal(new > 0, old > 0)
        clean = np.full((8, 8, 3), 100, dtype=np.uint8)
        poison = clean.copy()
        poison[4, 4] += 8
        audit = decoded_fidelity_pixel_audit(clean, poison, anns, "new_full_bbox8")
        self.assertEqual(audit["max_abs_delta_byte"]["overlap"], 8)
        self.assertEqual(audit["changed_overlap_pixels"], 1)
        with self.assertRaises(RuntimeError):
            decoded_fidelity_pixel_audit(clean, poison, anns, "old_overlap_half")

    def test_sampler_full_coverage_and_exact_resume(self):
        ids = tuple(f"person_{i:04d}" for i in range(19))
        original = PersistentShuffledSampler(ids, seed=0)
        first = [original.next_batch(4) for _ in range(3)]
        restored = PersistentShuffledSampler.from_state_dict(original.state_dict(), ids)
        tail_a = [original.next_batch(4) for _ in range(5)]
        tail_b = [restored.next_batch(4) for _ in range(5)]
        self.assertEqual(tail_a, tail_b)
        self.assertEqual(original.coverage(), restored.coverage())
        self.assertEqual(original.coverage()["surrogate_unique_images"], 19)
        self.assertEqual(original.coverage()["surrogate_coverage_ratio"], 1.0)
        self.assertEqual(len(first[0]), 4)
        with self.assertRaises(ValueError):
            PersistentShuffledSampler.from_state_dict(original.state_dict(), tuple(reversed(ids)))

    def test_em_registered_batch_schedule_covers_all_target_images(self):
        ids = tuple(f"person_{i:05d}" for i in range(6095))
        sampler = PersistentShuffledSampler(ids, seed=0)
        cycle_coverage = []
        for _ in range(5):
            for _ in range(10):
                sampler.next_batch(128)
            cycle_coverage.append(sampler.coverage()["surrogate_unique_images"])
        self.assertEqual(cycle_coverage[:4], [1280, 2560, 3840, 5120])
        self.assertEqual(cycle_coverage[4], 6095)
        self.assertEqual(sampler.coverage()["surrogate_coverage_ratio"], 1.0)
        self.assertTrue(all(sampler.seen[sample_id] >= 1 for sample_id in ids))

    def test_sem_budget_guard_reserves_a_complete_final_refinement(self):
        now = 1000.0
        projected = BUDGET_REFERENCE_CYCLE_SECONDS["sem_det"] * BUDGET_CYCLE_FACTOR
        required = 2 * projected + BUDGET_FINAL_RESERVE_SECONDS
        self.assertTrue(budget_allows_next_cycle("sem_det", now + required + 1, now, []))
        self.assertFalse(budget_allows_next_cycle("sem_det", now + required - 1, now, []))

    def test_em_budget_guard_uses_measured_full_cycle_and_finalization_reserve(self):
        now = 1000.0
        measured_cycle = BUDGET_REFERENCE_CYCLE_SECONDS["em_det"] * 1.1
        required = measured_cycle * BUDGET_CYCLE_FACTOR + BUDGET_FINAL_RESERVE_SECONDS
        self.assertTrue(budget_allows_next_cycle("em_det", now + required + 1,
                                                now, [measured_cycle]))
        self.assertFalse(budget_allows_next_cycle("em_det", now + required - 1,
                                                 now, [measured_cycle]))


if __name__ == "__main__":
    unittest.main()
