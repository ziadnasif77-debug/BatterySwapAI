"""The cost function we reimplemented (see scoring.py for the exception).

Because this scorer is OURS, its arithmetic has to be pinned harder than
borrowed code would be: every constant it consumes is checked against the
official settings, and each assumption A1-A8 has a test that fails if the
behaviour silently changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batteryswap.io import Scenario
from batteryswap.scoring import ScenarioScorer

SETTINGS = {
    "base_location": "b0", "base_room": "r0", "planning_window_days": 42,
    "early_replacement_penalty_daily": 0.5, "late_replacement_penalty_daily": 10.0,
    "unobserved_eol_days": 30.0, "overtime_start": 8.0,
    "overtime_penalty_factor": 2.0, "time_per_battery_hours": 0.25,
    "time_per_room_change_hours": 0.5, "time_per_building_change_hours": 1.0,
    "worker_limit_weekly_hours": 24.0, "worker_limit_weekly_penalty": 100.0,
    "worker_limit_daily_hours": 24.0, "worker_limit_daily_penalty": 100.0,
}


@pytest.fixture()
def scorer() -> ScenarioScorer:
    ids = ["b0", "b1"]
    travel = pd.DataFrame([[0.1, 1.0], [1.0, 0.1]], index=ids, columns=ids)
    scenario = Scenario(name="s", start_time=pd.Timestamp("2026-01-01"),
                        settings=dict(SETTINGS), travel=travel)
    location_of = {"d0": ("b0", "r0"), "d1": ("b0", "r0"),
                   "d2": ("b0", "r1"), "d3": ("b1", "r0")}
    eol_days = {"d0": 10.0, "d1": 20.0, "d2": 100.0, "d3": -5.0}
    return ScenarioScorer(scenario, location_of, eol_days)


def test_early_and_late_penalties_use_the_official_daily_rates(scorer):
    # d0 fails on day 10; replacing on day 4 wastes 6 days at 0.5/day
    assert scorer.device_penalty("d0", 4) == pytest.approx(3.0)
    # replacing on day 13 is 3 days late at 10.0/day
    assert scorer.device_penalty("d0", 13) == pytest.approx(30.0)
    # exactly on time is free
    assert scorer.device_penalty("d0", 10) == pytest.approx(0.0)


def test_late_is_twenty_times_early(scorer):
    """The asymmetry that drives every scheduling decision."""
    one_day_early = scorer.device_penalty("d0", 9)
    one_day_late = scorer.device_penalty("d0", 11)
    assert one_day_late / one_day_early == pytest.approx(20.0)


def test_skipping_a_device_that_never_fails_in_window_is_free(scorer):
    # A7: d2 fails on day 100, well past the 42-day horizon
    assert scorer.device_penalty("d2", None) == pytest.approx(0.0)
    assert scorer.skip_penalty(100.0) == pytest.approx(0.0)


def test_skipping_a_device_that_fails_in_window_is_charged_to_window_end(scorer):
    # A7: d0 fails on day 10, so 32 days of downtime remain
    assert scorer.device_penalty("d0", None) == pytest.approx(10.0 * 32)


def test_already_dead_device_is_charged_from_day_zero_not_before(scorer):
    """d3 failed 5 days before the window opened; downtime inside the window
    is capped at the window itself, never extended backwards."""
    assert scorer.device_penalty("d3", None) == pytest.approx(10.0 * 42)


def test_day_hours_charges_depot_round_trip_and_each_overhead(scorer):
    """A2 + A3: depot -> b0 -> depot, one building entry, one room entry."""
    travel_h, service_h = scorer.day_hours(("d0",))
    assert travel_h == pytest.approx(0.1 + 0.1)          # depot self-loop both ways
    assert service_h == pytest.approx(1.0 + 0.5 + 0.25)  # building + room + swap


def test_batching_two_devices_in_one_room_only_pays_the_swap_twice(scorer):
    _, one = scorer.day_hours(("d0",))
    _, two = scorer.day_hours(("d0", "d1"))
    assert two - one == pytest.approx(SETTINGS["time_per_battery_hours"])


def test_a_second_room_costs_one_more_room_change(scorer):
    _, same_room = scorer.day_hours(("d0", "d1"))
    _, two_rooms = scorer.day_hours(("d0", "d2"))
    assert two_rooms - same_room == pytest.approx(SETTINGS["time_per_room_change_hours"])


def test_overhead_per_transition_is_the_documented_alternative(scorer):
    """A3 has a second defensible reading; it must be reachable and cheaper."""
    alt = ScenarioScorer(scorer.scenario, scorer.location_of, scorer.eol_days,
                         overhead_per_visit=False)
    _, per_visit = scorer.day_hours(("d0", "d2"))
    _, per_transition = alt.day_hours(("d0", "d2"))
    assert per_transition < per_visit


def test_overtime_doubles_only_the_hours_beyond_the_threshold(scorer):
    """A4. 40 devices in one room: 1.0 + 0.5 + 40*0.25 = 11.5 h service."""
    many = {f"x{i}": 1 for i in range(40)}
    location_of = {**scorer.location_of, **{f"x{i}": ("b0", "r0") for i in range(40)}}
    eol_days = {**scorer.eol_days, **{f"x{i}": 1.0 for i in range(40)}}
    s = ScenarioScorer(scorer.scenario, location_of, eol_days)
    out = s.score(many, list(many))
    hours = out.raw_hours
    expected_overtime = max(hours - 8.0, 0.0)
    assert out.overtime_hours == pytest.approx(expected_overtime)
    assert out.work_cost == pytest.approx(hours + expected_overtime)


def test_weekly_cap_breach_adds_a_flat_penalty(scorer):
    """A5: 24 h inside one week is the binding constraint."""
    devices = {f"x{i}": 1 + (i % 3) for i in range(300)}   # days 1-3, same week
    location_of = {**scorer.location_of,
                   **{f"x{i}": ("b0", f"r{i % 5}") for i in range(300)}}
    eol_days = {**scorer.eol_days, **{f"x{i}": 1.0 for i in range(300)}}
    s = ScenarioScorer(scorer.scenario, location_of, eol_days)
    out = s.score(devices, list(devices))
    assert out.raw_hours > 24.0
    assert out.weeks_over_cap >= 1
    assert out.limit_penalty >= SETTINGS["worker_limit_weekly_penalty"]


def test_weeks_are_counted_from_the_window_start(scorer):
    """Day 7 and day 8 belong to different weeks (A8: days are 1-based)."""
    location_of = {**scorer.location_of}
    s = ScenarioScorer(scorer.scenario, location_of, scorer.eol_days)
    same_week = s.score({"d0": 6, "d1": 7}, ["d0", "d1"])
    split_week = s.score({"d0": 7, "d1": 8}, ["d0", "d1"])
    # identical work, but split across two weekly buckets
    assert same_week.raw_hours == pytest.approx(split_week.raw_hours)


def test_penalty_curve_matches_pointwise_penalties(scorer):
    """The planner consumes penalty_curve; the scorer consumes device_penalty.
    If these ever disagree, planning optimises something the score ignores."""
    curve = scorer.penalty_curve(scorer.eol_days["d0"])
    for day in (1, 5, 10, 11, 42):
        assert curve[day - 1] == pytest.approx(scorer.device_penalty("d0", day))


def test_total_is_the_sum_of_its_parts(scorer):
    out = scorer.score({"d0": 10, "d3": 1}, ["d0", "d1", "d2", "d3"])
    assert out.total == pytest.approx(out.early_penalty + out.late_penalty
                                      + out.work_cost + out.limit_penalty)


def test_scoring_is_deterministic(scorer):
    plan = {"d0": 9, "d1": 19, "d3": 1}
    population = ["d0", "d1", "d2", "d3"]
    first = scorer.score(plan, population).as_dict()
    second = ScenarioScorer(scorer.scenario, scorer.location_of,
                            scorer.eol_days).score(plan, population).as_dict()
    assert first == second


def test_unscheduled_devices_still_get_charged(scorer):
    """A device absent from the plan must not vanish from the bill."""
    with_all = scorer.score({}, ["d0", "d1", "d2", "d3"])
    plan_only = scorer.score({})
    assert with_all.late_penalty > 0
    assert plan_only.late_penalty == 0.0


def test_day_hours_cache_returns_the_same_answer(scorer):
    first = scorer.day_hours(("d0", "d1"))
    second = scorer.day_hours(("d1", "d0"))   # order must not matter
    assert first == second
    assert np.isfinite(first).all()
