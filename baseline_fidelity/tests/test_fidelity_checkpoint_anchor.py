import tempfile
import unittest
import random
from pathlib import Path

import numpy as np
import torch


def persistent_test_dir(prefix: str) -> Path:
    """Keep tiny test artifacts; avoid recursive temp-directory cleanup."""
    root = Path(__file__).resolve().parent / ".test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))

from scripts.generate_fidelity_baseline import (
    load_complete_cycle_checkpoint,
    optimizer_digest,
    save_complete_cycle_checkpoint,
)
from ue_framework.baselines_det.core import model_state_sha256
from ue_framework.baselines_det.fidelity import PersistentShuffledSampler


class FidelityRunStartCheckpointTest(unittest.TestCase):
    def test_run_start_anchor_replays_first_update_and_random_streams(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ids = ["person_a", "person_b", "person_c"]
        sampler = PersistentShuffledSampler(tuple(ids), seed=7)
        sampler_state = sampler.state_dict()
        generator = np.random.default_rng(7)
        spec = {
            "registered_config_sha256": "fixture-lock",
            "method_schedule": {"complete_outer_cycles": 0},
            "checkpoint_model_sha256": model_state_sha256(model),
            "checkpoint_optimizer_sha256": optimizer_digest(optimizer),
            "cycle_history": [], "state": "running",
        }
        path = persistent_test_dir("checkpoint-anchor-") / "anchor.pt.gz"
        save_complete_cycle_checkpoint(path, model=model, optimizer=optimizer,
                                       spec=spec, deltas={}, sampler=sampler,
                                       person_order=[0, 1, 2], rng=generator, centroid=None)
        self.assertTrue(path.is_file())
        checkpoint_hash_before = model_state_sha256(model)
        x = torch.tensor([[0.5, -0.25, 0.75]])
        y = torch.tensor([[0.0, 1.0]])

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        update()
        expected_model = model_state_sha256(model)
        expected_optimizer = optimizer_digest(optimizer)
        expected_batch = sampler.next_batch(2)
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(2)
        payload = load_complete_cycle_checkpoint(
            path, config_sha256="fixture-lock", sample_ids=ids,
            model=model, optimizer=optimizer, device=torch.device("cpu"))
        self.assertEqual(payload["checkpoint_boundary"], "run_start")
        self.assertEqual(payload["completed_cycles"], 0)
        self.assertEqual(payload["deltas"], {})
        sampler = PersistentShuffledSampler.from_state_dict(payload["sampler"], ids)
        generator = np.random.default_rng()
        generator.bit_generator.state = payload["numpy_generator_state"]
        self.assertEqual(sampler.next_batch(2), sampler_state["order"][:2])
        update()
        self.assertEqual(model_state_sha256(model), expected_model)
        self.assertEqual(optimizer_digest(optimizer), expected_optimizer)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(2), expected_torch)
        self.assertNotEqual(expected_model, checkpoint_hash_before)


if __name__ == "__main__":
    unittest.main()
