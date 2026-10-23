"""Compare the read-only VOC source bytes with the completed clean baseline lock."""
import argparse
import hashlib
import json
from pathlib import Path

from ue_framework.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_fingerprint(paths):
    # Same byte-level algorithm as the completed clean VOC baseline.
    lines = (f"{sha256_file(path)}  {path.resolve()}\n" for path in sorted(paths))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError("Fingerprint receipt must be a new file inside the isolated candidate")
    lock_path = Path(args.lock).resolve()
    clean_root = ROOT.parents[3] / "baseline_dev_20260928"
    if lock_path != (clean_root / "protocol_lock.json").resolve():
        raise ValueError("Use only the completed matched-clean protocol lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    dataset = Path(cfg["data"]["dataset_root"]).resolve()
    if dataset != Path(lock["dataset"]["root"]).resolve():
        raise ValueError("The ablation and clean baseline use different VOC source roots")
    found = {}
    for split in ("train", "val"):
        images = sorted(path for path in (dataset / "images" / split).iterdir() if path.is_file())
        labels = sorted(path for path in (dataset / "labels" / split).iterdir() if path.is_file())
        expected = int(lock["dataset"][f"{split}_images"])
        if (len(images) != expected or len(labels) != int(lock["dataset"][f"{split}_labels"])
                or {p.stem for p in images} != {p.stem for p in labels}):
            raise RuntimeError(f"{split} image/label count or stems differ")
        found[split] = split_fingerprint(images + labels)
        if found[split] != lock["dataset"][f"{split}_split_fingerprint"]:
            raise RuntimeError(f"{split} source bytes differ from matched-clean lock")
    combined = hashlib.sha256((found["train"] + "\n" + found["val"]).encode()).hexdigest()
    if combined != lock["dataset"]["fingerprint"]:
        raise RuntimeError("Combined source fingerprint differs from matched-clean lock")
    receipt = {"state": "passed", "source_root": str(dataset),
               "clean_protocol_lock_sha256": sha256_file(lock_path),
               "train_split_fingerprint": found["train"], "val_split_fingerprint": found["val"],
               "dataset_fingerprint": combined, "train_count": int(lock["dataset"]["train_images"]),
               "val_count": int(lock["dataset"]["val_images"])}
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
