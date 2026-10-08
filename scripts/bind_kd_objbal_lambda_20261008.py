"""Bind the one-shot gradient calibration SHA into a new immutable formal config."""

import argparse
import hashlib
import json
from pathlib import Path


PLACEHOLDER = "0" * 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.template).read_text(encoding="utf-8")
    artifact = Path(args.calibration).read_bytes()
    data = json.loads(artifact)
    if data.get("gate") != "KD_OBJECT_BALANCED_CALIBRATION_PASS":
        raise ValueError("Cannot bind a failing KD calibration")
    if source.count(PLACEHOLDER) != 1:
        raise ValueError("Config must contain exactly one unbound calibration hash")
    digest = hashlib.sha256(artifact).hexdigest()
    output = Path(args.output)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(source.replace(PLACEHOLDER, digest))
    print(json.dumps({"config": str(output), "calibration_sha256": digest}))


if __name__ == "__main__":
    main()
