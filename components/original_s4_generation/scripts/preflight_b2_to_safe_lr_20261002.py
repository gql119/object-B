"""Read-only preflight for frozen P5 coordinates and the rank-8 projector."""
import json
from pathlib import Path

import torch

from ue_framework.config import load_config
from ue_framework.methods.safebg import safe_background_direction


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ue_framework/configs/exp_voc_person_b2_to_safe_lr_p5_rank8_eps8_seed0_20261002.yaml"
EXPECTED = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def main():
    cfg = load_config(str(CONFIG))
    reference = load_config(str(ROOT / "ue_framework/configs/exp_voc_person_b2_to_safe_ls_p5_eps8_seed0_20261002.yaml"))
    flat, old = flatten(cfg), flatten(reference)
    changed = {key for key in flat.keys() | old.keys() if flat.get(key) != old.get(key)}
    if changed != {"platform.run_root", "methods.legacy_kproto_ret.semantic.safe_bg.mode",
                   "methods.legacy_kproto_ret.semantic.safe_bg.rank"}:
        raise ValueError(f"Safe-LR config has unexpected changes: {sorted(changed)}")
    sem = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    artifact = Path(sem["artifact_dir"])
    status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete":
        raise ValueError("Frozen coordinates are incomplete")
    selected = torch.load(artifact / "target_specific_channels.pt", map_location="cpu", weights_only=True)
    if selected["indices"].tolist() != EXPECTED:
        raise ValueError("P5 Strict TSFL selected indices changed")
    layer = json.loads((artifact / "strict_tsfl_selected_layer_summary.json").read_text(encoding="utf-8"))
    prereg = json.loads(Path(sem["strict_tsfl"]["layer_selection_path"]).read_text(encoding="utf-8"))
    if (layer["selected_layer"] != "P5" or prereg["selected_layer"] != "P5"
            or layer["selected_count"] != 24):
        raise ValueError("P5/24-channel selection changed")
    base = torch.load(artifact / "safe_bg_selected_subspaces.pt", map_location="cpu", weights_only=True)
    bases = {int(c): item["basis"] for c, item in base["classes"].items()}
    g = torch.Generator().manual_seed(20261002)
    target = torch.randn(4, 24, generator=g)
    bg = target + torch.randn(4, 24, generator=g)
    result, diag = safe_background_direction(target, bg, bases, active_k=3,
                                             mode="low_rank", safe_rank=8)
    if (diag["projector_rank"] != 8).any() or (diag["projector_basis_columns"] != 8).any():
        raise ValueError("Rank-8 projector failed synthetic probe")
    if not torch.isfinite(result).all() or diag["orthonormal_error"].max() > 1e-5:
        raise ValueError("Low-rank projector is unstable")
    print(json.dumps({"state": "passed", "selected_layer": "P5", "selected_count": 24,
                      "indices_match_previous": True, "safe_bg_rank": 8,
                      "synthetic_projector_rank": diag["projector_rank"].tolist(),
                      "synthetic_retention": diag["retention"].tolist(),
                      "orthonormal_error_max": float(diag["orthonormal_error"].max())}))


if __name__ == "__main__":
    main()
