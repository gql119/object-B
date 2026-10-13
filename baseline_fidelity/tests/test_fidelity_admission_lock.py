import unittest

from scripts.run_fidelity_baseline_pipeline import validate_admission


def lock(allowed=True):
    return {"protocol_id": "voc20_person_full_bbox8_last_primary_seed0_20261012_em_formal_v3",
            "official_full_experiments_allowed": allowed,
            "method_generation_authorization": {
                "allowed_methods": ["em_det"],
                "noise_time_policy": "warning_only_continue_to_registered_schedule",
            },
            "poison": {"spatial_protocol": "new_full_bbox8"},
            "victim": {"checkpoint_for_evaluation": "last.pt primary; best.pt secondary"}}


def schedule(method="em_det", status="ready_to_launch", enabled=True, lock_sha="lock-sha"):
    return {"protocol_lock_sha256": lock_sha, "formal_generation_enabled": enabled,
            "authorized_methods": ["em_det"],
            "methods": {method: {"status": status, "planned_complete_outer_cycles": 30,
                                 "time_cap_hours": 12.0,
                                 "continue_after_time_cap": True,
                                 "noise_time_policy": "warning_only_continue_to_registered_schedule"}}}


class FidelityAdmissionLockTest(unittest.TestCase):
    def test_provisional_lock_cannot_launch(self):
        with self.assertRaisesRegex(RuntimeError, "admission-only"):
            validate_admission(lock(False), schedule(), lock_sha256="lock-sha",
                               method="em_det", max_cycles=30, time_cap_hours=12.0)

    def test_enabled_em_requires_frozen_cap_and_cycles(self):
        validate_admission(lock(True), schedule(), lock_sha256="lock-sha",
                           method="em_det", max_cycles=30, time_cap_hours=12.0,
                           continue_after_time_cap=True)
        with self.assertRaisesRegex(RuntimeError, "cycle count"):
            validate_admission(lock(True), schedule(), lock_sha256="lock-sha",
                               method="em_det", max_cycles=29, time_cap_hours=12.0,
                               continue_after_time_cap=True)
        with self.assertRaisesRegex(RuntimeError, "time cap"):
            validate_admission(lock(True), schedule(), lock_sha256="lock-sha",
                               method="em_det", max_cycles=30, time_cap_hours=8.0,
                               continue_after_time_cap=True)

    def test_entf_remains_hard_blocked_even_if_schedule_is_enabled(self):
        gate = schedule(method="entf_push_det", status="ready_to_launch", enabled=True)
        gate["methods"]["entf_push_det"]["planned_complete_outer_cycles"] = 1
        gate["methods"]["entf_push_det"]["time_cap_hours"] = 12.0
        with self.assertRaisesRegex(RuntimeError, "EntF remains blocked"):
            validate_admission(lock(True), gate, lock_sha256="lock-sha",
                               method="entf_push_det", max_cycles=1, time_cap_hours=12.0)

    def test_time_policy_must_match_lock_and_schedule(self):
        gate = schedule()
        gate["methods"]["em_det"]["continue_after_time_cap"] = False
        with self.assertRaisesRegex(RuntimeError, "time-limit behavior"):
            validate_admission(lock(True), gate, lock_sha256="lock-sha",
                               method="em_det", max_cycles=30, time_cap_hours=12.0,
                               continue_after_time_cap=True)


if __name__ == "__main__":
    unittest.main()
