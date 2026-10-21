"""Fail-closed configuration and queue admission for the independent ablation."""
import copy
import hashlib
import json
import time
from pathlib import Path
from ue_framework.config import load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_full_config(root, lock):
    root = Path(root)
    parent_path = root/lock["parent_config_path"]
    candidate_path = root/lock["candidate_config_path"]
    if sha(parent_path) != lock["parent_config_sha256"] or sha(candidate_path) != lock["candidate_config_sha256"]:
        raise ValueError("Parent/candidate config hash mismatch")
    parent, candidate = load_config(str(parent_path)), load_config(str(candidate_path))
    restored = copy.deepcopy(candidate)
    if restored["methods"]["legacy_kproto_ret"]["support_mode"] != "bbox_full_target":
        raise ValueError("Candidate support identity missing")
    restored["methods"]["legacy_kproto_ret"]["support_mode"] = "bbox_half_overlap"
    if restored != parent:
        raise ValueError("Changes beyond the single approved support variable")
    if lock["support_mode"] != "bbox_full_target" or lock["tfss"]["eta_U"] != .85:
        raise ValueError("Frozen support/TFSS configuration changed")
    return candidate


def verify_queue(lock, receipt, *, source_commit, config_sha256, now=None):
    now = time.time() if now is None else now
    state_path = Path(lock["queue"]["pipeline_state_path"])
    state = json.loads(state_path.read_text())
    if state.get("pipeline_run_id") != lock["queue"]["pipeline_id"]:
        raise ValueError("Wrong predecessor pipeline")
    if state.get("state") != "COMPLETED_AND_ACCEPTED" or state.get("active_job") is not None:
        raise ValueError("Existing robustness pipeline not fully accepted")
    if any(state.get("stages", {}).get(stage, {}).get("state") != "ACCEPTED"
           for stage in lock["queue"]["required_stages"]):
        raise ValueError("Predecessor stage acceptance missing")
    if receipt.get("pipeline_state_sha256") != sha(state_path):
        raise ValueError("Pipeline state changed since queue audit")
    if receipt.get("source_commit") != source_commit or receipt.get("config_sha256") != config_sha256:
        raise ValueError("Queue receipt source/config differs")
    checked = receipt.get("checked_at_unix", 0)
    if not 0 <= now-checked <= 900:
        raise ValueError("Fresh queue audit required")
    if receipt.get("queue_conflicts") != [] or receipt.get("planned_queue_checked") is not True:
        raise ValueError("Subsequent queue not verified clear")
    if receipt.get("own_active_training_roots") != 0 or receipt.get("observed_via") != "RRCTL":
        raise ValueError("Existing training root or unverified transport")
    return state
