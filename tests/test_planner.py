"""The Planner must satisfy the official plan contract.

Every assertion here mirrors a check in
``batteryswap_public.evaluate.check_plan_valid``. They run without the official
package installed, so a contract violation is caught in CI rather than on the
submission machine.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from batteryswap.planner import SKIP_DATE, BatterySwapPlanner, daily_from_cut


class _StubModel:
    """Returns a fixed quantile grid per battery, in days."""

    def __init__(self, feature_names: list[str], days_by_row: list[list[float]]):
        self.feature_names_ = feature_names
        self._days = days_by_row

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = ["q05", "q10", "q25", "q50", "q75", "q90"]
        return pd.DataFrame(self._days[: len(X)], columns=cols, index=X.index)


class _Settings:
    planning_window_days = 42.0
    unobserved_eol_days = 30.0


def _series(battery: str, days: int = 400, v0: float = 3.0, v1: float = 2.5,
            end: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(end) - pd.Timedelta(days=days - 1),
                        periods=days, freq="D")
    return pd.DataFrame({"device_id": battery, "end_time": idx,
                         "voltage": np.linspace(v0, v1, days),
                         "temperature": 20.0})


def _problem(batteries: list[str], voltages: list[tuple[float, float]],
             buildings: list[str] | None = None, end: str = "2026-01-01"):
    buildings = buildings or ["b_1"] * len(batteries)
    frames = [_series(b, v0=v0, v1=v1, end=end)
              for b, (v0, v1) in zip(batteries, voltages)]
    data = pd.concat(frames, ignore_index=True).set_index(["device_id", "end_time"])
    locations = pd.DataFrame({
        "battery": batteries, "building": buildings,
        "room": [f"r_{i}" for i in range(len(batteries))],
        "start_time": pd.Timestamp("2024-01-01"),
        "end_time": pd.Timestamp(end),
    })
    travel = pd.DataFrame({"from": ["b_1"], "to": ["b_1"], "hours": [0.03]})
    return data, locations, travel


def _planner(n_rows: int, days: list[float], **kwargs) -> BatterySwapPlanner:
    feature_names = ["v_now"]
    model = _StubModel(feature_names, [days] * n_rows)
    planner = BatterySwapPlanner(model, offsets={}, quantiles=[0.05, 0.10, 0.25,
                                                              0.50, 0.75, 0.90],
                                 **kwargs)
    # the stub only knows one feature; keep the real selection off the matrix
    planner.model.feature_names_ = feature_names
    return planner


def test_daily_from_cut_collapses_to_one_row_per_device_day():
    data, _, _ = _problem(["d_a"], [(3.0, 2.5)])
    daily = daily_from_cut(data)
    assert list(daily.columns) == ["device_id", "timestamp", "voltage", "temperature"]
    assert daily["timestamp"].is_monotonic_increasing
    assert len(daily) == daily["timestamp"].nunique()


def test_plan_contains_every_battery_exactly_once():
    """check_plan_valid rejects missing OR duplicated batteries."""
    batteries = ["d_a", "d_b", "d_c"]
    data, locations, travel = _problem(batteries, [(3.0, 2.4)] * 3)
    plan = _planner(3, [5, 8, 12, 20, 30, 40], voltage_margin=None).plan(
        data, locations, travel, _Settings())
    assert sorted(plan["battery"]) == sorted(batteries)
    assert not plan["battery"].duplicated().any()


def test_plan_days_are_normalised_dates_and_non_decreasing():
    batteries = ["d_a", "d_b", "d_c"]
    data, locations, travel = _problem(batteries, [(3.0, 2.4)] * 3)
    plan = _planner(3, [3, 6, 10, 20, 30, 40], voltage_margin=None).plan(
        data, locations, travel, _Settings())
    days = plan["day"]
    assert pd.api.types.is_datetime64_any_dtype(days)
    assert (days.dt.normalize() == days).all(), "no fractional days allowed"
    assert (days.diff().dropna() >= pd.Timedelta(0)).all(), "days must not decrease"
    assert plan.index.equals(pd.RangeIndex(len(plan))), "range index required"


def test_plan_never_schedules_before_the_cutoff():
    batteries = ["d_a"]
    data, locations, travel = _problem(batteries, [(3.0, 2.3)])
    plan = _planner(1, [0, 0, 0, 0, 0, 0], voltage_margin=None).plan(
        data, locations, travel, _Settings())
    cutoff = pd.Timestamp("2026-01-01")
    assert (plan["day"] > cutoff).all(), "earliest usable slot is the day after"


def test_batteries_far_from_failure_are_pushed_past_the_window():
    """The skip mechanism: a date beyond end_time, which the evaluator cuts."""
    data, locations, travel = _problem(["d_a"], [(3.0, 3.0)])
    plan = _planner(1, [200, 220, 240, 260, 280, 300]).plan(
        data, locations, travel, _Settings())
    assert plan["day"].iloc[0] == SKIP_DATE


def test_voltage_gate_skips_a_healthy_battery_the_model_calls_urgent():
    """Physical gate beats the regression: a battery well above the failure
    voltage cannot earn a RECORDED end of life inside the window."""
    data, locations, travel = _problem(["d_a"], [(3.1, 3.0)])
    urgent = [1, 2, 3, 4, 5, 6]
    gated = _planner(1, urgent, voltage_margin=0.10).plan(
        data, locations, travel, _Settings())
    ungated = _planner(1, urgent, voltage_margin=None).plan(
        data, locations, travel, _Settings())
    assert gated["day"].iloc[0] == SKIP_DATE
    assert ungated["day"].iloc[0] < SKIP_DATE


def test_stale_batteries_are_skipped():
    """A device whose assumed EOL already passed must never be swapped: the
    penalty is unbounded late, while leaving it alone is free."""
    data, locations, travel = _problem(["d_a"], [(3.0, 2.3)], end="2026-01-01")
    locations["end_time"] = pd.Timestamp("2025-01-01")   # stopped reporting a year ago
    plan = _planner(1, [1, 2, 3, 4, 5, 6], voltage_margin=None).plan(
        data, locations, travel, _Settings())
    assert plan["day"].iloc[0] == SKIP_DATE


def test_stale_rule_can_be_disabled(monkeypatch):
    data, locations, travel = _problem(["d_a"], [(3.0, 2.3)], end="2026-01-01")
    locations["end_time"] = pd.Timestamp("2025-01-01")
    plan = _planner(1, [1, 2, 3, 4, 5, 6], voltage_margin=None,
                    skip_stale=False).plan(data, locations, travel, _Settings())
    assert plan["day"].iloc[0] < SKIP_DATE


def test_batching_pulls_a_building_onto_one_day_and_never_later():
    """Consolidation must move batteries EARLIER: late costs 20x early."""
    batteries = ["d_a", "d_b"]
    data, locations, travel = _problem(batteries, [(3.0, 2.4)] * 2,
                                       buildings=["b_1", "b_1"])
    planner = _planner(2, [5, 8, 12, 20, 30, 40], voltage_margin=None,
                       batch_window_days=30.0)
    # both stubs predict the same curve, so they already share a day
    plan = planner.plan(data, locations, travel, _Settings())
    assert plan["day"].nunique() == 1


def test_rows_within_a_day_are_grouped_by_building():
    """Row order IS the route - the evaluator charges a building change on
    every consecutive pair that differs."""
    batteries = ["d_a", "d_b", "d_c", "d_d"]
    data, locations, travel = _problem(batteries, [(3.0, 2.4)] * 4,
                                       buildings=["b_2", "b_1", "b_2", "b_1"])
    travel = pd.DataFrame({"from": ["b_1", "b_1", "b_2", "b_2"],
                           "to": ["b_1", "b_2", "b_1", "b_2"],
                           "hours": [0.03, 1.0, 1.0, 0.03]})
    plan = _planner(4, [5, 8, 12, 20, 30, 40], voltage_margin=None,
                    batch_window_days=30.0).plan(data, locations, travel, _Settings())
    building_of = dict(zip(locations["battery"], locations["building"]))
    for _, day_rows in plan.groupby("day"):
        sequence = [building_of[b] for b in day_rows["battery"]]
        transitions = sum(a != b for a, b in pairwise(sequence))
        assert transitions <= len(set(sequence)) - 1, "route revisits a building"


def test_planner_is_picklable():
    """It is shipped to the competition as a pickle."""
    import pickle
    planner = _planner(1, [5, 8, 12, 20, 30, 40])
    assert isinstance(pickle.loads(pickle.dumps(planner)), BatterySwapPlanner)


@pytest.mark.parametrize("q", [0.05, 0.20, 0.50])
def test_higher_q_never_schedules_earlier(q):
    """The day is the q-quantile of predicted EOL, so it must be monotone."""
    data, locations, travel = _problem(["d_a"], [(3.0, 2.4)])
    days = [5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
    plan = _planner(1, days, q=q, voltage_margin=None).plan(
        data, locations, travel, _Settings())
    expected = np.interp(q, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90], days)
    assert plan["day"].iloc[0] == pd.Timestamp("2026-01-01") + pd.Timedelta(
        days=int(np.clip(np.floor(expected), 1, 42)))
