"""Compare the preregistered rank/attenuation operators on fixed tensors."""
import importlib.util
from pathlib import Path

import torch

from ue_framework.methods.safebg import SafeBGTripletCache, safe_background_direction


def main():
    torch.manual_seed(4)
    bases = {i: torch.linalg.qr(torch.randn(24, 4)).Q for i in range(19)}
    clean = torch.randn(3, 24)
    background = clean + torch.randn(3, 24)
    for rank, alpha in ((4, 1.0), (8, 0.5), (8, 1.0)):
        cache = SafeBGTripletCache(bases, safe_rank=rank)
        output, diag = safe_background_direction(
            clean, background, bases, mode="low_rank", safe_rank=rank,
            removal_alpha=alpha, triplet_cache=cache)
        assert torch.equal(diag["projector_rank"], torch.full((3,), rank))
        assert (diag["orthonormal_error"] < 1e-5).all()
        assert (diag["decomposition_error"] < 1e-5).all()
        assert (diag["retention"] > 0).all()
        assert torch.allclose(diag["removed_norm"], alpha * diag["projected_component_norm"], atol=1e-6)
        assert torch.allclose(diag["retained_projected_component_norm"],
                              (1-alpha) * diag["projected_component_norm"], atol=1e-6)
        for row, active in enumerate(diag["active_classes"]):
            q = cache.lookup(active)[0]
            d = background[row] - clean[row]
            projected = q @ (q.T @ d)
            raw = d - alpha * projected
            expected = clean[row] + d.norm() * raw / (raw.norm() + 1e-12)
            assert torch.allclose(output[row], expected, atol=2e-6)
        if rank == 8 and alpha == 1.0:
            ancestor = Path(__file__).resolve().parents[3]
            choices = (ancestor / "ue_framework/methods/safebg.py",
                       ancestor / "b2_to_safebg_20261002/ue_project/ue_framework/methods/safebg.py")
            source = next((path for path in choices if path.is_file()), None)
            if source is None:
                raise FileNotFoundError("Cannot bind the original Safe-LR rank-8 computation")
            spec = importlib.util.spec_from_file_location("parent_safebg", source)
            parent = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parent)
            old, _ = parent.safe_background_direction(clean, background, bases,
                mode="low_rank", safe_rank=8)
            assert torch.allclose(output, old, atol=2e-6)
    print("rank4, soft05, and parent rank8 projection checks passed")


if __name__ == "__main__":
    main()
