"""Audit the exact configuration delta for the preregistered P5 Safe-LS smoke."""
import argparse
import hashlib
import json
from pathlib import Path

from ue_framework.config import load_config


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-config", required=True)
    parser.add_argument("--ls-config", required=True)
    args = parser.parse_args()
    original = flatten(load_config(args.safe_config))
    candidate = flatten(load_config(args.ls_config))
    changed = {key for key in original.keys() | candidate.keys()
               if original.get(key) != candidate.get(key)}
    allowed = {
        "platform.run_root", "methods.legacy_kproto_ret.kproto.feature_layer",
        "methods.legacy_kproto_ret.adaptive.feature_layer",
        "methods.legacy_kproto_ret.semantic.artifact_dir",
        "methods.legacy_kproto_ret.semantic.strict_tsfl.target_candidate_fraction",
        "methods.legacy_kproto_ret.semantic.strict_tsfl.nt_safety_quantile",
        "methods.legacy_kproto_ret.semantic.strict_tsfl.strict_tsfl_feature_level",
        "methods.legacy_kproto_ret.semantic.strict_tsfl.layer_selection_path",
    }
    if changed != allowed:
        raise RuntimeError(f"Unexpected configuration delta: {sorted(changed ^ allowed)}")
    expected = {
        "methods.legacy_kproto_ret.kproto.feature_layer": "model.21",
        "methods.legacy_kproto_ret.adaptive.feature_layer": "model.15",
        "methods.legacy_kproto_ret.semantic.strict_tsfl.target_candidate_fraction": .4,
        "methods.legacy_kproto_ret.semantic.strict_tsfl.nt_safety_quantile": .6,
        "methods.legacy_kproto_ret.semantic.strict_tsfl.strict_tsfl_feature_level": "P5",
    }
    if any(candidate[key] != value for key, value in expected.items()):
        raise RuntimeError("Selected layer, gates, or adaptive layer differ from preregistration")
    print(json.dumps({"state": "passed", "changed_paths": sorted(changed),
                      "config_sha256": hashlib.sha256(Path(args.ls_config).read_bytes()).hexdigest(),
                      "lambda_orth": candidate["methods.legacy_kproto_ret.semantic.lambda_orth"],
                      "orth_worst_k": candidate["methods.legacy_kproto_ret.semantic.orth_worst_k"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
