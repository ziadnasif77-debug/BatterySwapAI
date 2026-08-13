"""§16.5 — every emitted schedule respects worker count, hour cap, and travel
times. Full assertions activate once cost-model constants are resolved; the
routing arithmetic is testable now on synthetic geometry."""

import numpy as np
import pandas as pd
import pytest

from batteryswap.config import UnknownValueError, load_config
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


def test_full_feasibility_check_blocked_until_constants_resolved():
    """The real feasibility test needs worker_count and the day cap from the
    official evaluator; asserting with invented values would be worse than
    not asserting. This test documents (and enforces) that state."""
    cm = load_config("cost_model")
    with pytest.raises(UnknownValueError):
        _ = cm["workforce"]["worker_count"]
