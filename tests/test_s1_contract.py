import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.run_s1_csb_pipeline import summarize
from ue_framework.methods.csb_budget import digest

ROOT = Path(__file__).resolve().parents[1]


def test_original_parent_runtime_bytes_untouched():
    receipt = json.loads((ROOT / 'docs/s1_evidence/parent_remote_identity.json').read_text())
    for name, expected in receipt['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    old = json.loads((ROOT / 'configs/s1_csb_initial.json').read_text())
    new = json.loads((ROOT / 'configs/s1_csb_locked.json').read_text())
    assert new == {k: v for k, v in old.items() if k not in ['status', 'formal_generation_enabled', 'victim_training_enabled']}


def test_validator_ap75_column_and_role_metrics():
    values = np.tile(np.arange(10) / 10, (20, 1))
    box = SimpleNamespace(ap_class_index=np.arange(20), all_ap=values, r=np.ones(20)*.4, p=np.ones(20)*.6)
    result = summarize(box)
    assert result['mAP50_target'] == 0 and result['mAP75_target'] == .5
    assert result['Recall_target'] == .4 and result['Precision_target'] == .6
    box.all_ap[14, 0] = np.nan
    with pytest.raises(FloatingPointError): summarize(box)
