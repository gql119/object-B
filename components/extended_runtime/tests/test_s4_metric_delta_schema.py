from scripts.run_s4_tfss_pipeline import comparable_metric_delta
import pytest


def test_historical_voc20_schema_without_aggregate_recall_precision_is_supported():
    candidate = {
        "mAP50_all": 0.50,
        "Recall_all": 0.60,
        "Precision_all": 0.70,
        "Recall_per_class": [0.1] * 20,
        "AP50_per_class": [0.2] * 20,
    }
    historical = {
        "mAP50_all": 0.45,
        "Recall_target": 0.01,
        "Precision_non_target": 0.8,
        "Recall_per_class": [0.1] * 20,
        "AP50_per_class": [0.2] * 20,
    }

    delta, unavailable = comparable_metric_delta(candidate, historical)

    assert delta.keys() == {"mAP50_all"}
    assert delta["mAP50_all"] == pytest.approx(0.05)
    assert unavailable == ["Precision_all", "Recall_all"]
