"""Metadata-only guard for pixel-space purification outputs.

This contract checks identity and annotation invariants. It does not prove that
an image still depicts the same objects, and it does not implement a purifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
NUM_VOC_CLASSES = 20


class PairContractError(ValueError):
    """Raised when before/after records violate the frozen pixel-only contract."""


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_sha256: str
    labels_sha256: str
    width: int
    height: int
    # Each item is (VOC class id, x1, y1, x2, y2) in pixel coordinates.
    boxes_xyxy_px: tuple[tuple[int, float, float, float, float], ...]


def _validate_record(record: SampleRecord) -> None:
    if not record.image_id:
        raise PairContractError("image_id must be non-empty")
    if not _SHA256.fullmatch(record.image_sha256):
        raise PairContractError(f"invalid image SHA-256 for {record.image_id}")
    if not _SHA256.fullmatch(record.labels_sha256):
        raise PairContractError(f"invalid labels SHA-256 for {record.image_id}")
    if not isinstance(record.width, int) or not isinstance(record.height, int):
        raise PairContractError(f"image dimensions must be integers for {record.image_id}")
    if record.width <= 0 or record.height <= 0:
        raise PairContractError(f"image dimensions must be positive for {record.image_id}")

    for i, box in enumerate(record.boxes_xyxy_px):
        if len(box) != 5:
            raise PairContractError(f"box {i} must have class,x1,y1,x2,y2")
        cls, x1, y1, x2, y2 = box
        if not isinstance(cls, int) or not 0 <= cls < NUM_VOC_CLASSES:
            raise PairContractError(f"box {i} has an invalid VOC class id")
        if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2)):
            raise PairContractError(f"box {i} has a non-finite coordinate")
        if not (0 <= x1 < x2 <= record.width and 0 <= y1 < y2 <= record.height):
            raise PairContractError(f"box {i} lies outside image bounds or has zero area")


def validate_sample_pair(before: SampleRecord, after: SampleRecord) -> None:
    """Require unchanged image identity, dimensions, and exact GT metadata.

    The image hash is intentionally allowed to change. Label bytes and decoded
    boxes must remain unchanged for this pixel-only, no-geometry contract.
    """
    _validate_record(before)
    _validate_record(after)
    if before.image_id != after.image_id:
        raise PairContractError("image_id changed")
    if (before.width, before.height) != (after.width, after.height):
        raise PairContractError("image dimensions changed")
    if before.labels_sha256 != after.labels_sha256:
        raise PairContractError("label bytes changed")
    if before.boxes_xyxy_px != after.boxes_xyxy_px:
        raise PairContractError("decoded GT classes or coordinates changed")


def validate_dataset_pairs(
    before: Iterable[SampleRecord], after: Iterable[SampleRecord]
) -> None:
    """Require a one-to-one sample mapping and validate each pixel-only pair."""
    before_rows = list(before)
    after_rows = list(after)
    before_by_id = {row.image_id: row for row in before_rows}
    after_by_id = {row.image_id: row for row in after_rows}
    if len(before_by_id) != len(before_rows):
        raise PairContractError("duplicate image_id in before manifest")
    if len(after_by_id) != len(after_rows):
        raise PairContractError("duplicate image_id in after manifest")
    if before_by_id.keys() != after_by_id.keys():
        missing = sorted(before_by_id.keys() - after_by_id.keys())
        extra = sorted(after_by_id.keys() - before_by_id.keys())
        raise PairContractError(f"sample identity set changed; missing={missing}, extra={extra}")
    for image_id in sorted(before_by_id):
        validate_sample_pair(before_by_id[image_id], after_by_id[image_id])
