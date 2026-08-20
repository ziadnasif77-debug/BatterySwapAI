"""Thin wrapper around the OFFICIAL scorer (section 5.3).

This module NEVER reimplements the cost function. It imports
``batteryswap_public`` - the package the competition ships and runs on its own
submission machine - and calls it directly.

The package was not in the data release; it lives on PyPI and is pulled in by
the official example repository's ``requirements.txt``. Install it with:

    pip install batteryswap_public fastparquet structlog pydantic-settings

Dataset layout the official loader expects (one directory per split):

    <root>/<split>/devices.csv
    <root>/<split>/eol_times.csv
    <root>/<split>/battery_metrics.parquet
    <root>/<split>/scenarios.json

Historical note: before the official package was located, this project ran on
its own reimplementation of the cost function. That module is retained as
``scoring.py`` and is now used only by the synthetic dry run - nothing on the
official path may import it, and `tests/test_evaluator_agreement.py` measures
how far it drifts from the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


class OfficialEvaluatorMissingError(RuntimeError):
    def __init__(self, detail: str):
        super().__init__(
            f"The official evaluator is unavailable: {detail}\n"
            f"Install it with: pip install batteryswap_public fastparquet "
            f"structlog pydantic-settings\n"
            f"Reimplementing the scorer is forbidden (section 18)."
        )


def _official():
    """Import the official modules, or fail with an actionable message."""
    try:
        from batteryswap_public import evaluate as official_evaluate
        from batteryswap_public import utils as official_utils
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OfficialEvaluatorMissingError(str(exc)) from exc
    return official_evaluate, official_utils


def is_available() -> bool:
    try:
        _official()
        return True
    except OfficialEvaluatorMissingError:
        return False


@dataclass
class ScenarioProblem:
    """One planning problem exactly as the official harness hands it over."""
    scenario: dict
    locations: pd.DataFrame      # alive devices only: battery, building, room, ...
    timeseries: pd.DataFrame     # truncated at the scenario start
    eol_times: pd.Series         # NaT where unobserved; NOT visible to a planner

    @property
    def name(self) -> str:
        return self.scenario["name"]

    @property
    def settings(self):
        return self.scenario["settings"]

    @property
    def travel_costs(self) -> pd.DataFrame:
        return self.scenario["travel_costs"]

    @property
    def start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.scenario["start_time"])

    @property
    def end_time(self) -> pd.Timestamp:
        return self.start_time + pd.Timedelta(days=self.settings.planning_window_days)


def load_split(split_path: str | Path):
    """Official loader. Returns (locations, timeseries, eol_times, scenarios)."""
    _, utils = _official()
    return utils.load_dataset(Path(split_path))


def iter_problems(split_path: str | Path) -> Iterator[ScenarioProblem]:
    """Official scenario iterator: drops already-dead devices and truncates the
    series at each scenario start, so leakage is prevented by their code."""
    _, utils = _official()
    locations, timeseries, eol_times, scenarios = utils.load_dataset(Path(split_path))
    for scenario, locs, cut, not_dead in utils.iterate_scenarios(
            locations, timeseries, eol_times, scenarios):
        yield ScenarioProblem(scenario=scenario, locations=locs,
                              timeseries=cut, eol_times=not_dead)


def score_plan(plan: pd.DataFrame, problem: ScenarioProblem) -> pd.Series:
    """Score one plan with the official evaluator. Returns its cost Series
    (``total_cost`` plus every component)."""
    official, _ = _official()
    _, _, scores = official.evaluate_plan(
        plan.reset_index(drop=True), problem.locations, problem.travel_costs,
        problem.settings, eol_times=problem.eol_times, start_time=problem.start_time)
    return scores


def check_plan(plan: pd.DataFrame, problem: ScenarioProblem) -> None:
    """Official validity check - raises InvalidPlan. Cheap; call it before
    scoring so a malformed plan fails loudly rather than mid-evaluation."""
    official, _ = _official()
    official.check_plan_valid(plan.reset_index(drop=True), problem.locations,
                              start_time=problem.start_time)


def invalid_plan_error():
    """The official InvalidPlan exception type, for callers that catch it."""
    official, _ = _official()
    return official.InvalidPlan
