"""Evaluate already-trained clean and B2-TO last.pt on the same clean VOC val."""
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

from scripts.evaluate_b2_to_safebg_checkpoints import summarize
from ue_framework.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clean-victim", required=True)
    parser.add_argument("--b2to-victim", required=True)
    parser.add_argument("--safe-lr-victim", required=True)
    parser.add_argument("--data-yaml", required=True,
                        help="The Safe-LR isolated victim data YAML with byte-identical clean validation split")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Reference evaluation output must be in Safe-LR isolation")
    output.mkdir(exist_ok=False)
    cfg = load_config(args.config)
    victim_cfg = cfg["victim"]
    result = {}
    data_yaml = Path(args.data_yaml).resolve()
    if not data_yaml.is_relative_to(ROOT.resolve()) or not data_yaml.is_file():
        raise ValueError("Reference evaluations must use the isolated clean validation copy")
    for name, root in (("clean", Path(args.clean_victim)),
                       ("b2to", Path(args.b2to_victim)),
                       ("safe_lr", Path(args.safe_lr_victim))):
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        spec = json.loads((root / "run_spec.json").read_text(encoding="utf-8"))
        if status.get("state") != "complete" or int(spec["train_args"]["epochs"]) != 200:
            raise RuntimeError(f"{name} reference victim is not complete")
        for key in ("batch", "imgsz", "workers", "optimizer", "cos_lr", "close_mosaic", "cache",
                    "amp", "lr0", "lrf", "momentum", "weight_decay", "seed", "pretrained"):
            expected = 0 if key == "seed" else False if key == "pretrained" else victim_cfg[key]
            if spec["train_args"][key] != expected:
                raise ValueError(f"{name} victim protocol mismatch: {key}")
        checkpoint = root / "victim/weights/last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name} last.pt missing")
        metrics = YOLO(str(checkpoint)).val(data=str(data_yaml), imgsz=int(victim_cfg["imgsz"]),
            batch=int(victim_cfg["batch"]), workers=int(victim_cfg["workers"]),
            device="0", verbose=False, plots=False, seed=0, project=str(output),
            name=f"{name}_last_validation", exist_ok=False)
        summary = summarize(metrics.box)
        summary.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                       checkpoint_role="last.pt", epoch=200, validation_split="clean VOC val")
        result[name] = summary
        (output / f"{name}_last_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in data.items()
                             if key in ("mAP50_target", "mAP50_non_target", "mAP50_all", "checkpoint_sha256")}
                      for name, data in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
