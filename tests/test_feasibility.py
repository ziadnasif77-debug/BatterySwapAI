"""§16.5 — every emitted schedule respects worker count, hour cap, and travel
times. Full assertions activate once cost-model constants are resolved; the
routing arithmetic is testable now on synthetic geometry."""

import numpy as np
import pandas as pd
import pytest

from batteryswap.config import load_config
from batteryswap.optimizer.routing import plan_day, route_length_minutes, two_opt


@pytest.fixture()
def travel() -> pd.DataFrame:
    ids = ["A", "B", "C", "D"]
    m = np.array([
        [0, 10, 20, 30],
        [10, 0, 15, 25],
        [20, 15, 0, 12],
        [30, 25, 12, 0],
    ], dtype=float)
    return pd.DataFrame(m, index=ids, columns=ids)


def test_two_opt_never_worsens(travel):
    order = ["A", "D", "B", "C"]
    improved = two_opt(order, travel)
    assert route_length_minutes(improved, travel) <= route_length_minutes(order, travel)
    assert sorted(improved) == sorted(order)  # a permutation, nothing dropped


def test_plan_day_accounting(travel):
    plan = plan_day(["A", "B", "C"], {"A": 30.0, "B": 45.0, "C": 15.0},
                    travel, day_cap_minutes=480.0, start="A")
    assert plan.service_minutes == 90.0
    assert plan.total_minutes == plan.travel_minutes + plan.service_minutes
    assert plan.overtime_minutes == max(0.0, plan.total_minutes - 480.0)
    assert set(plan.order) == {"A", "B", "C"}


def test_official_hour_caps_are_loadable_and_the_weekly_one_binds():
    """The release settles the workforce limits in HOURS. With a 42-day window
    the weekly cap is what actually constrains the plan, so it is asserted
    explicitly — a regression here would silently relax feasibility."""
    cm = load_config("cost_model")
    assert cm["workforce"]["overtime_start_hours"] == 8.0
    assert cm["workforce"]["overtime_penalty_factor"] == 2.0
    weekly = cm["workforce"]["worker_limit_weekly_hours"]
    daily = cm["workforce"]["worker_limit_daily_hours"]
    assert weekly == 24.0 and daily == 24.0
    # 6 weeks x 24 h is the entire crew budget for one scenario
    assert cm["horizon"]["planning_days"] / 7 * weekly == 144.0
