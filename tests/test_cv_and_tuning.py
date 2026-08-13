"""Grouped CV (§7) and the q sweep (§11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batteryswap.cv import grouped_folds, summarize_folds
from batteryswap.optimizer import Schedule
from batteryswap.tuning import best_q, default_q_grid, sweep_q

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def test_grouped_folds_never_split_a_building_across_sides():
    groups = pd.Series([f"bl{i // 4:02d}" for i in range(40)])
    for train_idx, val_idx in grouped_folds(groups, n_splits=5):
        train_b = set(groups.iloc[train_idx])
        val_b = set(groups.iloc[val_idx])
        assert not (train_b & val_b), "a building appeared on both sides of the split"
        assert len(val_b) > 0


def test_grouped_folds_rejects_more_splits_than_buildings():
    groups = pd.Series(["bl00"] * 10 + ["bl01"] * 10)
    with pytest.raises(ValueError, match="exceeds"):
        grouped_folds(groups, n_splits=5)


def test_summarize_folds_reports_spread_not_just_the_mean():
    folds = pd.DataFrame({"fold": [0, 1, 2], "mae_hours": [100.0, 200.0, 300.0]})
    summary = summarize_folds(folds)
    assert summary.loc["mae_hours", "mean"] == pytest.approx(200.0)
    assert summary.loc["mae_hours", "std"] > 0


def test_default_q_grid_contains_the_newsvendor_value():
    grid = default_q_grid(q_star=0.0909)
    assert 0.0909 in grid
    assert min(grid) >= 0.01 and max(grid) <= 0.95


def test_sweep_q_finds_the_cost_minimising_q():
    """A synthetic objective that prefers late days: the sweep must climb to a
    high q rather than sit at the theoretical value."""
    n = 5
    cal = pd.DataFrame({
        "q05": np.full(n, 100.0), "q25": np.full(n, 300.0),
        "q50": np.full(n, 500.0), "q75": np.full(n, 700.0),
        "q95": np.full(n, 900.0),
        "battery_id": [f"b{i}" for i in range(n)],
    })

    def objective(s: Schedule) -> float:
        # cheapest when every battery is scheduled as late as possible
        return -float(sum(s.assignments.values()))

    sweep = sweep_q(cal, QUANTILES, horizon_days=60, objective=objective,
                    q_grid=[0.05, 0.25, 0.5, 0.75, 0.95])
    q, cost = best_q(sweep)
    assert q == 0.95
    assert cost == sweep["final_cost"].min()
    assert list(sweep["q"]) == sorted(sweep["q"])


def test_sweep_q_applies_the_post_process_before_scoring():
    cal = pd.DataFrame({
        "q05": [100.0], "q25": [300.0], "q50": [500.0],
        "q75": [700.0], "q95": [900.0], "battery_id": ["b0"],
    })
    seen = {}

    def post(s: Schedule) -> Schedule:
        seen["called"] = True
        return Schedule(assignments={b: 1 for b in s.assignments})

    sweep = sweep_q(cal, QUANTILES, horizon_days=60,
                    objective=lambda s: float(sum(s.assignments.values())),
                    q_grid=[0.5], post_process=post)
    assert seen.get("called")
    assert sweep["final_cost"].iloc[0] == 1.0
