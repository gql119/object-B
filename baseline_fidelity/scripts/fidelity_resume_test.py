"""Tiny real-CUDA checkpoint round trip; no VOC data or detector weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.generate_fidelity_baseline import (
    ALLOWED_ROOT, load_complete_cycle_checkpoint, optimizer_digest,
    save_complete_cycle_checkpoint,
)
from ue_framework.baselines_det.core import model_state_sha256
from ue_framework.baselines_det.fidelity import PersistentShuffledSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ALLOWED_ROOT)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    device = torch.device("cuda:0")
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 3, device=device)).sum().backward()
    optimizer.step()
    sampler = PersistentShuffledSampler(("a", "b"), seed=23)
    sampler.next_batch(1)
    rng = np.random.default_rng(23)
    rng.random()
    spec = {"registered_config_sha256": "toy-fixed-hash",
            "method_schedule": {"complete_outer_cycles": 1},
            "checkpoint_model_sha256": model_state_sha256(model),
            "checkpoint_optimizer_sha256": optimizer_digest(optimizer)}
    path = out / "last_complete_cycle.pt.gz"
    save_complete_cycle_checkpoint(path, model=model, optimizer=optimizer, spec=spec,
                                   deltas={"a": torch.ones(1), "b": torch.zeros(1)},
                                   sampler=sampler, person_order=[1, 0], rng=rng,
                                   centroid=None)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1, device=device)
    other = torch.nn.Linear(3, 2).to(device)
    other_optimizer = torch.optim.Adam(other.parameters(), lr=1e-3)
    payload = load_complete_cycle_checkpoint(path, config_sha256="toy-fixed-hash",
                                             sample_ids=["a", "b"], model=other,
                                             optimizer=other_optimizer, device=device)
    if (random.random() != expected_python or np.random.random() != expected_numpy or
            not torch.equal(torch.rand(1, device=device), expected_torch)):
        raise RuntimeError("RNG continuation differs")
    if model_state_sha256(other) != model_state_sha256(model):
        raise RuntimeError("Model continuation differs")
    if optimizer_digest(other_optimizer) != optimizer_digest(optimizer):
        raise RuntimeError("Optimizer continuation differs")
    restored = PersistentShuffledSampler.from_state_dict(payload["sampler"], ("a", "b"))
    if restored.next_batch(1) != sampler.next_batch(1):
        raise RuntimeError("Sampler continuation differs")
    try:
        load_complete_cycle_checkpoint(path, config_sha256="incorrect",
                                       sample_ids=["a", "b"], model=other,
                                       optimizer=other_optimizer, device=device)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Config mismatch was not rejected")
    result = {"status": "passed", "model": True, "optimizer": True,
              "rng": True, "sampler": True, "config_rejection": True,
              "checkpoint_bytes": path.stat().st_size}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
