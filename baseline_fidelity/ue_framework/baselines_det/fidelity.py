"""Protocol helpers for the independent detection-baseline fidelity study.

These helpers never change the historical overlap-half baseline protocol.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .core import TARGET_ID, _xyxy, support_mask


def validate_spatial_protocol_lock(lock: dict, protocol: str) -> None:
    """Reject a run when its declared pixel protocol disagrees with its lock."""
    expected = {
        "new_full_bbox8": (1.0, "8/255"),
        "old_overlap_half": (0.5, "4/255"),
    }
    if protocol not in expected:
        raise ValueError(f"Unknown spatial protocol: {protocol}")
    expected_scale, expected_overlap = expected[protocol]
    poison = lock.get("poison", {})
    try:
        actual_scale = float(poison.get("overlap_scale"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Protocol lock is missing a numeric overlap_scale") from exc
    if poison.get("epsilon") != "8/255":
        raise ValueError("Protocol lock epsilon must be 8/255")
    if actual_scale != expected_scale:
        raise ValueError(
            f"Spatial protocol {protocol} requires overlap_scale={expected_scale}, got {actual_scale}"
        )
    if poison.get("overlap_effective_epsilon") != expected_overlap:
        raise ValueError(
            f"Spatial protocol {protocol} requires overlap epsilon {expected_overlap}"
        )
    if poison.get("outside_target_epsilon") not in (0, "0", "0/255"):
        raise ValueError("Protocol lock must keep perturbation outside Person boxes at zero")
    if protocol == "new_full_bbox8" and poison.get("support") != "person_box_union":
        raise ValueError("Full-BBox-8 lock must declare support=person_box_union")


def fidelity_support_mask(height: int, width: int, annotations: Sequence[dict],
                          protocol: str) -> np.ndarray:
    """Return the exact spatial multiplier for one registered protocol."""
    if protocol == "old_overlap_half":
        return support_mask(height, width, annotations)
    if protocol != "new_full_bbox8":
        raise ValueError(f"Unknown spatial protocol: {protocol}")
    person = np.zeros((height, width), dtype=np.float32)
    for ann in annotations:
        if int(ann["cls"]) == TARGET_ID:
            x1, y1, x2, y2 = _xyxy(ann["bbox"], height, width)
            person[y1:y2, x1:x2] = 1.0
    return person


def decoded_fidelity_pixel_audit(clean: np.ndarray, poison: np.ndarray,
                                 annotations: Sequence[dict], protocol: str) -> dict:
    """Audit decoded uint8 pixels in exclusive, overlap, and outside regions."""
    if clean.shape != poison.shape or clean.dtype != np.uint8 or poison.dtype != np.uint8:
        raise ValueError("Both decoded images must have the same uint8 RGB shape")
    height, width = clean.shape[:2]
    person = fidelity_support_mask(height, width, annotations, "new_full_bbox8") > 0
    other = np.zeros((height, width), dtype=bool)
    for ann in annotations:
        if int(ann["cls"]) != TARGET_ID:
            x1, y1, x2, y2 = _xyxy(ann["bbox"], height, width)
            other[y1:y2, x1:x2] = True
    diff = np.abs(poison.astype(np.int16) - clean.astype(np.int16))
    maxima = {"target_only": int(diff[person & ~other].max(initial=0)),
              "overlap": int(diff[person & other].max(initial=0)),
              "outside": int(diff[~person].max(initial=0))}
    overlap_limit = 8 if protocol == "new_full_bbox8" else 4 if protocol == "old_overlap_half" else None
    if overlap_limit is None:
        raise ValueError(f"Unknown spatial protocol: {protocol}")
    if maxima["target_only"] > 8 or maxima["overlap"] > overlap_limit or maxima["outside"] != 0:
        raise RuntimeError(f"Decoded fidelity budget failed: {maxima}")
    return {"max_abs_delta_byte": maxima,
            "changed_target_pixels": int(np.any(diff[person] > 0, axis=-1).sum()),
            "changed_overlap_pixels": int(np.any(diff[person & other] > 0, axis=-1).sum()),
            "changed_outside_pixels": int(np.any(diff[~person] > 0, axis=-1).sum())}


@dataclass
class PersistentShuffledSampler:
    """Continue an indexed train iterator across cycles; checkpoint exactly."""

    sample_ids: tuple[str, ...]
    seed: int
    order: list[int] = field(default_factory=list)
    cursor: int = 0
    pass_index: int = 0
    seen: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Sample IDs must be nonempty and unique")
        if not self.order:
            self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = np.random.default_rng(self.seed + self.pass_index).permutation(
            len(self.sample_ids)).tolist()
        self.cursor = 0

    def next_batch(self, batch_size: int) -> list[int]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.cursor >= len(self.order):
            self.pass_index += 1
            self._reshuffle()
        result = self.order[self.cursor:self.cursor + batch_size]
        self.cursor += len(result)
        for index in result:
            self.seen[self.sample_ids[index]] += 1
        return result

    def coverage(self) -> dict:
        return {
            "surrogate_seen_images": sum(self.seen.values()),
            "surrogate_unique_images": len(self.seen),
            "surrogate_coverage_ratio": len(self.seen) / len(self.sample_ids),
            "surrogate_full_passes": self.pass_index,
            "sample_revisit_distribution": dict(sorted(Counter(self.seen.values()).items())),
        }

    def state_dict(self) -> dict:
        return {"sample_ids": list(self.sample_ids), "seed": self.seed,
                "order": list(self.order), "cursor": self.cursor,
                "pass_index": self.pass_index, "seen": dict(self.seen)}

    @classmethod
    def from_state_dict(cls, state: dict, sample_ids: Sequence[str]) -> "PersistentShuffledSampler":
        if list(sample_ids) != state["sample_ids"]:
            raise ValueError("Sample inventory changed; exact resume is unsafe")
        sampler = cls(tuple(sample_ids), int(state["seed"]))
        sampler.order = list(state["order"])
        sampler.cursor = int(state["cursor"])
        sampler.pass_index = int(state["pass_index"])
        sampler.seen = Counter(state["seen"])
        if sorted(sampler.order) != list(range(len(sample_ids))) or not 0 <= sampler.cursor <= len(sample_ids):
            raise ValueError("Sampler checkpoint is invalid")
        return sampler
