"""One-batch YOLO CUDA exact-resume check using only VOC training images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from scripts.generate_fidelity_baseline import (
    ALLOWED_ROOT, _assert_lock, load_complete_cycle_checkpoint,
    optimizer_digest, read_item, save_complete_cycle_checkpoint,
)
from ue_framework.baselines_det.core import model_state_sha256
from ue_framework.baselines_det.fidelity import PersistentShuffledSampler
from ue_framework.baselines_det.production import em_noise_batch, em_train_batch, new_counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ALLOWED_ROOT)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    _, labels, target_paths = _assert_lock(lock, args.surrogate, args.dataset_root)
    sample_ids = tuple(path.stem for path in target_paths[:2])
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    torch.cuda.manual_seed_all(19)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    model = YOLO(str(args.surrogate)).model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sampler = PersistentShuffledSampler(sample_ids, seed=19)
    items = [read_item(path, labels, device) for path in target_paths[:2]]
    counters = new_counters()
    em_train_batch(model, optimizer, items, imgsz=640, counters=counters)
    deltas = em_noise_batch(model, items, steps=1, step_size=0.8 / 255,
                            imgsz=640, counters=counters)
    delta_state = {item.stem: delta.detach().cpu() for item, delta in zip(items, deltas)}
    sampler.next_batch(1)
    rng = np.random.default_rng(19)
    rng.random()
    spec = {
        "registered_config_sha256": "real-detector-resume-v1",
        "method_schedule": {"complete_outer_cycles": 1},
        "checkpoint_model_sha256": model_state_sha256(model),
        "checkpoint_optimizer_sha256": optimizer_digest(optimizer),
    }
    checkpoint = output / "last_complete_cycle.pt.gz"
    save_complete_cycle_checkpoint(
        checkpoint, model=model, optimizer=optimizer, spec=spec,
        deltas=delta_state, sampler=sampler, person_order=[0, 1],
        rng=rng, centroid=None,
    )
    # Advance the original branch after checkpoint, then restore and replay it.
    continuation = [read_item(path, labels, device) for path in target_paths[:2]]
    for item in continuation:
        item.raw_delta = delta_state[item.stem].to(device)
    em_train_batch(model, optimizer, continuation, imgsz=640, counters=new_counters())
    expected_model = model_state_sha256(model)
    expected_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
    expected_optimizer = optimizer_digest(optimizer)
    expected_sampler = sampler.next_batch(1)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_cuda = torch.rand(1, device=device)
    payload = load_complete_cycle_checkpoint(
        checkpoint, config_sha256="real-detector-resume-v1",
        sample_ids=list(sample_ids), model=model, optimizer=optimizer, device=device,
    )
    if set(payload["deltas"]) != set(sample_ids):
        raise RuntimeError("Detector delta IDs did not survive checkpoint")
    restored_sampler = PersistentShuffledSampler.from_state_dict(payload["sampler"], sample_ids)
    if restored_sampler.next_batch(1) != expected_sampler:
        raise RuntimeError("Detector sampler continuation differs")
    em_train_batch(model, optimizer, continuation, imgsz=640, counters=new_counters())
    if model_state_sha256(model) != expected_model:
        differences = {}
        for name, value in model.state_dict().items():
            distance = (value.detach().cpu().float() - expected_state[name].float()).abs().max().item()
            if distance:
                differences[name] = distance
        largest = sorted(differences.items(), key=lambda entry: entry[1], reverse=True)[:8]
        raise RuntimeError(f"Detector model replay differs: {largest}; optimizer="
                           f"{optimizer_digest(optimizer) == expected_optimizer}")
    if optimizer_digest(optimizer) != expected_optimizer:
        raise RuntimeError("Detector optimizer replay differs")
    if (random.random() != expected_python or np.random.random() != expected_numpy
            or not torch.equal(torch.rand(1, device=device), expected_cuda)):
        raise RuntimeError("Detector RNG continuation differs")
    result = {
        "status": "passed", "train_samples": list(sample_ids),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_replay_exact": True, "optimizer_replay_exact": True,
        "sampler_replay_exact": True, "rng_replay_exact": True,
        "full_cycle_6095_resume": "NOT_RUN",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
