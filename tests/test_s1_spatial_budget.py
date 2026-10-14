import numpy as np
import pytest
from ue_framework.methods.csb_budget import construct_budget, png_project

SETTINGS = dict(eta=.25, gamma=1., epsilon_max=8 / 255, q_low=.1, q_high=.9, smooth_kernel=3)


def test_masked_map_limits_and_monotonic_mapping():
    mask = np.zeros((9, 9), bool); mask[1:8, 1:8] = True
    g = np.arange(81, dtype=np.float32).reshape(9, 9)
    budget, z, stats = construct_budget({11: g}, mask, SETTINGS)
    assert np.all(budget[~mask] == 0)
    assert budget[mask].min() >= 2 / 255 - 1e-9 and budget.max() <= 8 / 255 + 1e-9
    assert budget[mask][z[mask].argmax()] <= budget[mask][z[mask].argmin()]
    assert not stats[0]['degenerate']


def test_only_no_nt_can_fallback_and_constant_has_recorded_midpoint():
    mask = np.ones((5, 5), bool)
    b, _, _ = construct_budget({}, mask, SETTINGS, no_nt_gt=True)
    assert np.allclose(b, 8 / 255)
    with pytest.raises(ValueError): construct_budget({}, mask, SETTINGS)
    with pytest.raises(ValueError): construct_budget({11: np.zeros((5, 5))}, mask, SETTINGS)
    b, _, rows = construct_budget({11: np.ones((5, 5))}, mask, SETTINGS)
    assert rows[0]['degenerate'] and np.allclose(b, 5 / 255)


def test_png_continuous_budget_no_one_byte_slack():
    clean = np.full((2, 2, 3), 100, np.uint8)
    budget = np.array([[0, 2.9 / 255], [4.1 / 255, 8 / 255]], np.float32)
    poison = clean / 255 + .1
    result = png_project(clean, poison, budget)
    diff = abs(result.astype(int) - clean.astype(int)).max(2)
    assert np.all(diff / 255 <= budget + 1e-9)
    assert diff.tolist() == [[0, 2], [4, 8]]
