"""Training-only correctness and one complete inner-unit timing probe.

This never materializes a dataset or launches a victim. Output paths are new.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from scripts.generate_fidelity_baseline import (
    ALLOWED_ROOT, EM_NOISE_STEP, EM_NOISE_STEPS, EM_TRAIN_BATCHES,
    ENTF_STEP, ENTF_STEPS, IMG_SIZE, _assert_lock, read_item,
)
from ue_framework.baselines_det.core import EPS, file_sha256, model_state_sha256
from ue_framework.baselines_det.fidelity import fidelity_support_mask
from ue_framework.baselines_det.production import (
    BatchedPersonROIFeatures, em_noise_batch, em_train_batch,
    entf_centroid_batch, entf_push_batch, new_counters, sem_defender_batch,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("em_det", "sem_det", "entf_push_det"), required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--protocol-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--source-snapshot-sha256", required=True)
    p.add_argument("--full-inner", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint-footprint", action="store_true")
    p.add_argument("--spatial-protocol", choices=("old_overlap_half", "new_full_bbox8"),
                   default="new_full_bbox8")
    args = p.parse_args()
    output = args.output.resolve()
    output.relative_to(ALLOWED_ROOT)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,95}", args.run_id):
        raise ValueError("Smoke RunID must be a unique simple path component")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_snapshot_sha256):
        raise ValueError("Source snapshot must be identified by a lowercase SHA256")
    if output.exists():
        raise FileExistsError(output)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    _, labels, targets = _assert_lock(lock, args.surrogate, args.dataset_root)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Real GPU is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if args.batch_size < 1 or args.batch_size > len(targets):
        raise ValueError("Invalid probe batch size")
    items = [read_item(path, labels, device, args.spatial_protocol)
             for path in targets[:args.batch_size]]
    model = YOLO(str(args.surrogate)).model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model_state_sha256(model)
    counters = new_counters()
    losses: list[float] = []
    stats: dict = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.method == "em_det":
        train_steps = EM_TRAIN_BATCHES if args.full_inner else 1
        for _ in range(train_steps):
            losses.append(em_train_batch(model, optimizer, items,
                                         imgsz=IMG_SIZE, counters=counters))
        deltas = em_noise_batch(model, items, steps=EM_NOISE_STEPS if args.full_inner else 2,
                                step_size=EM_NOISE_STEP, imgsz=IMG_SIZE,
                                counters=counters, loss_trace=losses, stats=stats)
        expected_model_change = True
    elif args.method == "sem_det":
        deltas, loss = sem_defender_batch(
            model, optimizer, items, defender_steps=10 if args.full_inner else 1,
            stability_samples=5 if args.full_inner else 1,
            surrogate_pgd_steps=10 if args.full_inner else 1,
            imgsz=IMG_SIZE, seed=0, counters=counters, loss_trace=losses,
            random_start=True, stats=stats)
        if loss is None:
            raise RuntimeError("Training SEM omitted its surrogate update")
        expected_model_change = True
        frozen_sha = model_state_sha256(model)
        final, final_loss = sem_defender_batch(
            model, optimizer, items, defender_steps=10 if args.full_inner else 1,
            stability_samples=5 if args.full_inner else 1,
            surrogate_pgd_steps=10 if args.full_inner else 1,
            imgsz=IMG_SIZE, seed=1, counters=counters, loss_trace=losses,
            random_start=True, update_surrogate=False)
        if final_loss is not None or model_state_sha256(model) != frozen_sha:
            raise RuntimeError("SEM final regeneration mutated the final surrogate")
        deltas = final
    else:
        extractor = BatchedPersonROIFeatures(model, layer="model.15", imgsz=IMG_SIZE)
        try:
            centroid, object_count, n_batches = entf_centroid_batch(extractor, [items])
            counters["forward_calls"] += n_batches
            if object_count < 1:
                raise RuntimeError("No person centroid object")
            deltas = entf_push_batch(extractor, items, centroid,
                                     steps=ENTF_STEPS if args.full_inner else 2,
                                     step_size=ENTF_STEP, seed=0,
                                     counters=counters, loss_trace=losses, stats=stats)
        finally:
            extractor.close()
        expected_model_change = False
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    changed = before != model_state_sha256(model)
    if changed != expected_model_change:
        raise RuntimeError("Surrogate update/freeze contract failed")
    max_linf = 0.0
    overlap_pixels = changed_overlap_pixels = max_overlap_byte = 0
    for item, delta in zip(items, deltas):
        applied = (delta.clamp(-EPS, EPS) * item.support).detach()
        outside = applied * (item.support == 0)
        linf = applied.abs().max().item()
        if outside.abs().max().item() != 0 or linf > EPS + 1e-7:
            raise RuntimeError("Spatial support or epsilon budget failed")
        max_linf = max(max_linf, linf)
        legacy = fidelity_support_mask(item.clean.shape[-2], item.clean.shape[-1],
                                       item.annotations, "old_overlap_half")
        overlap = torch.from_numpy(legacy == 0.5).to(device)
        overlap_pixels += int(overlap.sum())
        clean_bytes = (item.clean[0] * 255).round().to(torch.int16)
        poison_bytes = ((item.clean[0] + applied[0]).clamp(0, 1) * 255).round().to(torch.int16)
        difference = (poison_bytes - clean_bytes).abs().amax(dim=0)
        if overlap.any():
            changed_overlap_pixels += int((difference[overlap] > 0).sum())
            max_overlap_byte = max(max_overlap_byte, int(difference[overlap].max()))
    if not losses or not all(torch.isfinite(torch.tensor(loss)).item() for loss in losses):
        raise RuntimeError("Non-finite loss trace")
    checkpoint_bytes = checkpoint_write_seconds = None
    if args.checkpoint_footprint:
        tensor_path = output.with_suffix(".deltas.pt.gz")
        if tensor_path.exists():
            raise FileExistsError(tensor_path)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        with gzip.open(tensor_path, "wb", compresslevel=1) as stream:
            torch.save({item.stem: delta.detach().cpu()
                        for item, delta in zip(items, deltas)}, stream)
        checkpoint_write_seconds = time.perf_counter() - write_started
        checkpoint_bytes = tensor_path.stat().st_size
    result = {"method": args.method, "full_inner": args.full_inner,
              "run_id": args.run_id, "seed": 0,
              "source_snapshot_sha256": args.source_snapshot_sha256,
              "production_entrypoint": "scripts/fidelity_smoke.py::main",
              "production_entrypoint_reached": True,
              "computation_kind": "training", "finite_loss": True,
              "official_metrics_disabled": True, "artifact_ingest_disabled": True,
              "cleanup_policy": "retain_all", "checkpoint_cleanup_completed": False,
              "deleted_paths": [],
              "checkpoint_paths_remaining": ([str(output.with_suffix(".deltas.pt.gz"))]
                                             if checkpoint_bytes is not None else []),
              "determinism": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                              "torch_deterministic_algorithms": True, "warn_only": True},
              "spatial_protocol": args.spatial_protocol,
              "sample_stems_sha256": hashlib.sha256(
                  "\n".join(item.stem for item in items).encode()).hexdigest(),
              "batch_size": len(items), "surrogate_sha256": file_sha256(args.surrogate),
              "model_changed": changed, "loss_first": losses[0], "loss_last": losses[-1],
              "loss_observations": len(losses), "duration_seconds": duration,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
              "delta_linf": max_linf,
              "support_pixel_count": sum(int((item.support > 0).sum().item()) for item in items),
              "overlap_pixels": overlap_pixels,
              "changed_overlap_pixels": changed_overlap_pixels,
              "max_overlap_delta_byte": max_overlap_byte,
              "counters": counters, "noise_diagnostics": stats, "status": "passed"}
    result["checkpoint_delta_bytes"] = checkpoint_bytes
    result["checkpoint_delta_write_seconds"] = checkpoint_write_seconds
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
