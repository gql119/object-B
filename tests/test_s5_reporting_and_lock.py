"""Historical optional-metric regression and unchanged-parent config guard."""
import copy
import json
from pathlib import Path
import pytest
from scripts.s5_report import comparison_tables
from scripts.run_s5_pipeline import verify_lock

ROOT=Path(__file__).resolve().parents[1]


def test_historical_missing_optional_aggregate_never_aborts_full_class_comparison():
    old=json.loads((ROOT/'docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json').read_text())['metrics']
    metrics={k:copy.deepcopy(old[k]) for k in ('Clean','S0','S1','S2','S3','S4')}
    # Schema-only regression: no synthetic experiment result is written.
    metrics['S5']=copy.deepcopy(old['S4']);metrics['S5_best_secondary']=copy.deepcopy(old['S4_best_secondary'])
    comparison,aggregate,perclass=comparison_tables(metrics)
    assert len(perclass)==20 and len(aggregate)==7
    assert comparison['delta']['S4']['mAP50_target']==0
    assert 'Recall_all' in comparison['delta_unavailable_historical_aggregates']['S0']
    assert comparison['metrics']['S4']['checkpoint_sha256']==old['S4']['checkpoint_sha256']


def test_s5_lock_preserves_parent_science_and_rejects_unauthorized_spatial_change():
    lock=json.loads((ROOT/'docs/S5_CUP_S4/S5_CONFIG_LOCK.json').read_text())
    assert verify_lock(lock)
    broken=copy.deepcopy(lock);broken['tfss']['eta_U']=.84
    with pytest.raises(ValueError,match='Original S4 science'):verify_lock(broken)
