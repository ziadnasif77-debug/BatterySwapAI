"""Decision layer + opportunistic rule + SAA (§11, §12, §12.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from batteryswap.decision import (cost_curve, cost_curves_from_constants, days_at_q,
                                  newsvendor_q, quantile_grid_to_samples)
from batteryswap.optimizer import Schedule
from batteryswap.optimizer.opportunistic import opportunistic_upgrade
from batteryswap.saa import draw_failure_scenarios, scenario_cost_curves

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def _calibrated(n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = np.sort(rng.uniform(200, 4000, size=(n, len(QUANTILES))), axis=1)
    df = pd.DataFrame(grid, columns=[f"q{int(q * 100):02d}" for q in QUANTILES])
    df["battery_id"] = [f"b{i}" for i in range(n)]
    return df


def test_newsvendor_q_matches_the_identity():
    # lateness 10x costlier -> replace at the 0.09 quantile (§11)
    assert abs(newsvendor_q(1.0, 10.0) - 1 / 11) < 1e-12


def test_days_at_q_is_monotone_in_q():
    cal = _calibrated()
    days = [min(days_at_q(cal, QUANTILES, q, horizon_days=400).values())
            for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert days == sorted(days), "a higher q must never schedule earlier"


def test_days_at_q_respects_the_horizon():
    cal = _calibrated()
    for day in days_at_q(cal, QUANTILES, 0.95, horizon_days=30).values():
        assert 1 <= day <= 30


def test_cost_curve_minimum_sits_near_the_newsvendor_quantile():
    """The single-battery, no-travel optimum is q* — the identity §11 relies on."""
    c_early, c_late = 1.0, 9.0
    levels = np.asarray(QUANTILES, dtype=float)
    q_row = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
    samples = quantile_grid_to_samples(q_row, levels, n_samples=4000)
    days = np.arange(1, 45, dtype=float)
    g = cost_curve(samples, days, c_early, c_late)

    best_hours = days[int(np.argmin(g))] * 24.0
    target = float(np.quantile(samples, newsvendor_q(c_early, c_late)))
    assert abs(best_hours - target) < 48.0  # within one day-grid step either side


def test_opportunistic_upgrade_consolidates_but_does_not_collapse():
    """⛔ The amortization guard: a shared trip may pull batteries in, but the
    schedule must not degenerate into 'replace everything on the first day'."""
    horizon = 40
    curves = {}
    # four batteries in one building, optima spread across the horizon
    for i, opt in enumerate((5, 8, 25, 38)):
        d = np.arange(1, horizon + 1)
        curves[f"b{i}"] = np.abs(d - opt).astype(float) * 3.0
    battery_building = {b: "bl0" for b in curves}
    start = Schedule(assignments={f"b{i}": opt for i, opt in enumerate((5, 8, 25, 38))})

    # generous but amortized trip cost
    upgraded, moved = opportunistic_upgrade(start, curves, battery_building,
                                            {"bl0": 120.0})
    assert moved >= 1, "a nearby battery should join an existing visit"
    assert len(set(upgraded.assignments.values())) > 1, "schedule collapsed to one day"

    # zero trip value -> nothing may move
    unchanged, moved_zero = opportunistic_upgrade(start, curves, battery_building,
                                                  {"bl0": 0.0})
    assert moved_zero == 0
    assert unchanged.assignments == start.assignments


def test_opportunistic_upgrade_is_deterministic():
    curves = {f"b{i}": np.abs(np.arange(1, 31) - opt).astype(float)
              for i, opt in enumerate((3, 7, 20))}
    bb = {b: "bl0" for b in curves}
    s = Schedule(assignments={"b0": 3, "b1": 7, "b2": 20})
    first = opportunistic_upgrade(s, curves, bb, {"bl0": 60.0})[0].assignments
    second = opportunistic_upgrade(s, curves, bb, {"bl0": 60.0})[0].assignments
    assert first == second


def test_saa_scenarios_converge_to_the_analytic_curves():
    """With LINEAR penalties, scenario averaging must reproduce the analytic
    integral — the check that the SAA path is wired correctly (§12.5)."""
    cal = _calibrated(n=4, seed=3)
    c_early, c_late, horizon = 0.05, 0.5, 200

    analytic = {b: g.sort_values("day")["expected_penalty"].to_numpy()
                for b, g in cost_curves_from_constants(
                    cal, QUANTILES, c_early, c_late, horizon).groupby("battery_id")}
    ids, scen = draw_failure_scenarios(cal, QUANTILES, n_scenarios=20000, seed=1)
    sampled = scenario_cost_curves(ids, scen, horizon, c_early, c_late)

    scale = max(c.max() for c in analytic.values())
    dev = max(np.max(np.abs(sampled[b] - analytic[b])) for b in analytic) / scale
    assert dev < 0.02, f"scenario curves deviate {dev:.3%} from analytic"


def test_saa_scenarios_are_deterministic_given_a_seed():
    cal = _calibrated(n=3)
    _, a = draw_failure_scenarios(cal, QUANTILES, n_scenarios=50, seed=7)
    _, b = draw_failure_scenarios(cal, QUANTILES, n_scenarios=50, seed=7)
    assert np.array_equal(a, b)
