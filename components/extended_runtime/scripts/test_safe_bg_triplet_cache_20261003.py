"""Compare production Safe-BG online and cached paths on identical GPU inputs."""

import argparse
import json
import time

import torch

from ue_framework.methods.legacy_kproto_ret import kproto_losses
from ue_framework.methods.safebg import SafeBGTripletCache, safe_background_direction


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = torch.load(args.artifact, map_location="cpu", weights_only=True)
    bases = {int(c): item["basis"].to(device) for c, item in loaded["classes"].items()}
    cache = SafeBGTripletCache(bases, projector_rtol=1e-6, safe_rank=8)
    torch.manual_seed(173)
    inputs = []
    for index in range(100):
        n = 1 + index % 4
        clean = torch.randn(n, 24, device=device)
        background = clean + torch.randn(n, 24, device=device)
        poison = clean + 0.2 * torch.randn(n, 24, device=device)
        inputs.append((clean, background, poison))
    differences = {key: 0.0 for key in ("safe_bg", "L_BG", "L_rel", "retention",
                                        "projector_rank", "singular_values_top16")}
    active_mismatch = 0
    for clean, background, poison in inputs:
        online, online_diag = safe_background_direction(clean, background, bases,
            active_k=3, projector_rtol=1e-6, mode="low_rank", safe_rank=8)
        cached, cached_diag = safe_background_direction(clean, background, bases,
            active_k=3, projector_rtol=1e-6, mode="low_rank", safe_rank=8,
            triplet_cache=cache)
        active_mismatch += online_diag["active_classes"] != cached_diag["active_classes"]
        differences["safe_bg"] = max(differences["safe_bg"], float((online-cached).abs().max()))
        for name in ("retention", "projector_rank", "singular_values_top16"):
            differences[name] = max(differences[name],
                float((online_diag[name]-cached_diag[name]).abs().max()))
        zero = clean.new_zeros((0, 24))
        online_losses = kproto_losses(poison, clean, online, zero, zero, 0.2)
        cached_losses = kproto_losses(poison, clean, cached, zero, zero, 0.2)
        for key, idx in (("L_BG", 0), ("L_rel", 1)):
            differences[key] = max(differences[key],
                float((online_losses[idx]-cached_losses[idx]).abs()))
    sample = inputs[:30]
    for clean, background, _ in sample[:3]:
        safe_background_direction(clean, background, bases, active_k=3,
            projector_rtol=1e-6, mode="low_rank", safe_rank=8)
        safe_background_direction(clean, background, bases, active_k=3,
            projector_rtol=1e-6, mode="low_rank", safe_rank=8, triplet_cache=cache)
    times = {}
    for name, use_cache in (("online", False), ("cached", True)):
        sync(device)
        started = time.perf_counter()
        for clean, background, _ in sample:
            safe_background_direction(clean, background, bases, active_k=3,
                projector_rtol=1e-6, mode="low_rank", safe_rank=8,
                triplet_cache=cache if use_cache else None)
        sync(device)
        times[name] = (time.perf_counter() - started) / len(sample)
    output = {"device": str(device), "fixed_input_batches": len(inputs),
              "fixed_input_vectors": sum(x[0].shape[0] for x in inputs),
              "triplet_count": len(cache.entries),
              "projector_variants": sum(map(len, cache.entries.values())),
              "projector_sha256": cache.sha256,
              "cache_build_seconds": cache.precompute_seconds,
              "active_mismatch": active_mismatch, "max_abs_differences": differences,
              "safe_bg_call_seconds_mean": times,
              "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated() / 1048576
                                         if device.type == "cuda" else None)}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output))
    if active_mismatch or max(differences.values()) >= 1e-5:
        raise SystemExit("Cached Safe-BG is not equivalent within 1e-5")


if __name__ == "__main__":
    main()
