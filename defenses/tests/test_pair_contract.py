import unittest

from defenses.common.pair_contract import (
    PairContractError,
    SampleRecord,
    validate_dataset_pairs,
    validate_sample_pair,
)


def record(image_id, boxes, *, image_sha="a", labels_sha="b", width=640, height=480):
    return SampleRecord(
        image_id=image_id,
        image_sha256=image_sha * 64,
        labels_sha256=labels_sha * 64,
        width=width,
        height=height,
        boxes_xyxy_px=tuple(boxes),
    )


class PairContractTests(unittest.TestCase):
    def test_cooccurring_person_and_non_target_preserved(self):
        boxes = ((14, 2.0, 3.0, 24.0, 45.0), (6, 30.0, 5.0, 200.0, 220.0))
        validate_sample_pair(record("cooccur", boxes), record("cooccur", boxes, image_sha="c"))

    def test_person_only_preserved(self):
        boxes = ((14, 8.0, 12.0, 20.0, 70.0),)
        validate_sample_pair(record("person-only", boxes), record("person-only", boxes, image_sha="c"))

    def test_person_free_empty_gt_preserved(self):
        validate_sample_pair(record("person-free", ()), record("person-free", (), image_sha="c"))

    def test_small_object_is_not_filtered(self):
        boxes = ((14, 0.0, 0.0, 2.0, 3.0), (11, 30.0, 20.0, 34.0, 26.0))
        validate_sample_pair(record("small", boxes), record("small", boxes, image_sha="c"))

    def test_detects_box_change(self):
        before = record("x", ((14, 1.0, 1.0, 5.0, 8.0),))
        after = record("x", ((14, 1.0, 1.0, 5.0, 7.0),))
        with self.assertRaisesRegex(PairContractError, "decoded GT"):
            validate_sample_pair(before, after)

    def test_detects_label_byte_change(self):
        before = record("x", ())
        after = record("x", (), labels_sha="d")
        with self.assertRaisesRegex(PairContractError, "label bytes"):
            validate_sample_pair(before, after)

    def test_detects_geometry_change(self):
        before = record("x", ())
        after = record("x", (), height=640)
        with self.assertRaisesRegex(PairContractError, "dimensions"):
            validate_sample_pair(before, after)

    def test_detects_missing_sample_identity(self):
        with self.assertRaisesRegex(PairContractError, "identity set changed"):
            validate_dataset_pairs((record("a", ()), record("b", ())), (record("a", ()),))

    def test_detects_duplicate_sample_identity(self):
        with self.assertRaisesRegex(PairContractError, "duplicate image_id"):
            validate_dataset_pairs((record("a", ()), record("a", ())), (record("a", ()),))

    def test_rejects_out_of_bounds_or_nonfinite_boxes(self):
        for box in (
            (14, -1.0, 0.0, 5.0, 5.0),
            (14, 0.0, 0.0, float("nan"), 5.0),
        ):
            with self.subTest(box=box), self.assertRaises(PairContractError):
                validate_sample_pair(record("x", (box,)), record("x", (box,)))


if __name__ == "__main__":
    unittest.main()
