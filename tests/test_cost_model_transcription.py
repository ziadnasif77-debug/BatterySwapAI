"""configs/cost_model.yaml must match the official source, exactly.

The constants live in data/raw/scenarios.json (licensed, never committed), so
this test self-activates when the data is present and skips otherwise. It is
the guard against a transcription drifting from the official release.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

SCENARIOS = Path("data/raw/scenarios.json")

# yaml key path -> official JSON settings key
MAPPING = {
    ("time", "swap_hours_per_battery"): "time_per_battery_hours",
    ("time", "room_change_overhead_hours"): "time_per_room_change_hours",
    ("time", "building_change_overhead_hours"): "time_per_building_change_hours",
    ("workforce", "overtime_start_hours"): "overtime_start",
    ("workforce", "overtime_penalty_factor"): "overtime_penalty_factor",
    ("workforce", "worker_limit_daily_hours"): "worker_limit_daily_hours",
    ("workforce", "worker_limit_daily_penalty"): "worker_limit_daily_penalty",
    ("workforce", "worker_limit_weekly_hours"): "worker_limit_weekly_hours",
    ("workforce", "worker_limit_weekly_penalty"): "worker_limit_weekly_penalty",
    ("penalties", "early_replacement_per_day"): "early_replacement_penalty_daily",
    ("penalties", "late_replacement_per_day"): "late_replacement_penalty_daily",
    ("horizon", "planning_days"): "planning_window_days",
    ("horizon", "unobserved_eol_days"): "unobserved_eol_days",
}

pytestmark = pytest.mark.skipif(not SCENARIOS.exists(),
                                reason="official scenarios.json not present")


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cost_model() -> dict:
    return yaml.safe_load(Path("configs/cost_model.yaml").read_text(encoding="utf-8"))


def test_every_transcribed_constant_matches_the_official_file(cost_model, scenarios):
    settings = scenarios[0]["settings"]
    for (section, key), official_key in MAPPING.items():
        assert cost_model[section][key] == settings[official_key], (
            f"cost_model.yaml {section}.{key} disagrees with "
            f"scenarios.json settings.{official_key}")


def test_constants_are_identical_across_every_scenario(scenarios):
    """If a constant ever varies per scenario it is an INPUT, not a constant,
    and must not live in cost_model.yaml."""
    first = scenarios[0]["settings"]
    for official_key in MAPPING.values():
        values = {s["settings"][official_key] for s in scenarios}
        assert values == {first[official_key]}, (
            f"settings.{official_key} varies across scenarios: {sorted(values)}")


def test_derived_ratio_and_newsvendor_follow_from_the_penalties(cost_model):
    early = cost_model["penalties"]["early_replacement_per_day"]
    late = cost_model["penalties"]["late_replacement_per_day"]
    assert cost_model["derived"]["late_over_early_ratio"] == pytest.approx(late / early)
    assert cost_model["derived"]["newsvendor_q_star"] == pytest.approx(
        early / (early + late), rel=1e-5)
