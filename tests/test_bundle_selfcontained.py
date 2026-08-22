"""The bundle must run with nothing but itself, the runtime packages, and the
dataset the competition mounts.

The container proved this empirically once. These tests lock the property so a
later edit cannot quietly reintroduce a dependency on something outside the
submission directory - which would fail only on the submission machine, after
the deadline.

The vendored package DOES contain filesystem access (io.load_raw reads the raw
dataset, config.load_config reads configs/*.yaml). None of it is reachable from
plan(), and that is exactly what is asserted here rather than assumed.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from batteryswap import config as config_module
from batteryswap import io as io_module
from batteryswap.planner import BatterySwapPlanner


class _Flat:
    """Model stub: fixed quantiles, so the test exercises plan() not the model."""

    feature_names_: ClassVar[list[str]] = []

    def predict_quantiles(self, X):
        cols = ["q05", "q10", "q25", "q50", "q75", "q90"]
        grid = np.tile(np.array([5.0, 8.0, 15.0, 30.0, 60.0, 120.0]), (len(X), 1))
        return pd.DataFrame(grid, columns=cols, index=X.index)


class _Settings:
    planning_window_days = 42.0
    unobserved_eol_days = 30.0


def _problem(n_devices: int = 4, days: int = 400):
    end = pd.Timestamp("2026-01-01")
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), periods=days, freq="D")
    frames = []
    for i in range(n_devices):
        frames.append(pd.DataFrame({
            "device_id": f"d_{i}", "end_time": idx,
            "voltage": np.linspace(3.0, 2.40 + 0.02 * i, days),
            "temperature": 18.0 + 4.0 * np.sin(np.arange(days) / 30.0),
        }))
    data = pd.concat(frames, ignore_index=True).set_index(["device_id", "end_time"])
    locations = pd.DataFrame({
        "battery": [f"d_{i}" for i in range(n_devices)],
        "building": ["b_1", "b_1", "b_2", "b_2"][:n_devices],
        "room": [f"r_{i}" for i in range(n_devices)],
        "start_time": pd.Timestamp("2024-01-01"), "end_time": end,
    })
    travel = pd.DataFrame({"from": ["b_1", "b_1", "b_2", "b_2"],
                           "to": ["b_1", "b_2", "b_1", "b_2"],
                           "hours": [0.03, 0.5, 0.5, 0.03]})
    return data, locations, travel


def _planner() -> BatterySwapPlanner:
    data, _, _ = _problem()
    from batteryswap.features import compute_features
    from batteryswap.planner import daily_from_cut

    daily = daily_from_cut(data)
    sample = daily[daily["device_id"] == daily["device_id"].iloc[0]]
    model = _Flat()
    model.feature_names_ = list(compute_features(sample, daily["timestamp"].max(),
                                                 2.42).keys())
    return BatterySwapPlanner(model, offsets={}, quantiles=[0.05, 0.10, 0.25,
                                                           0.50, 0.75, 0.90])


def test_plan_never_reads_a_config_file(monkeypatch):
    """configs/ does not exist in the competition image. If plan() ever reaches
    load_config the submission dies there and nowhere else."""
    def explode(*args, **kwargs):
        raise AssertionError("plan() reached load_config - configs/ is absent "
                             "from the competition image")

    monkeypatch.setattr(config_module, "load_config", explode)
    # features.py imported the symbol directly, so patch it there too
    import batteryswap.features as features_module
    monkeypatch.setattr(features_module, "load_config", explode)

    data, locations, travel = _problem()
    plan = _planner().plan(data, locations, travel, _Settings())
    assert len(plan) == len(locations)


def test_plan_never_loads_the_raw_dataset(monkeypatch):
    """The planner is handed its data as arguments. Reaching for the dataset on
    disk would mean depending on a layout the competition does not provide."""
    def explode(*args, **kwargs):
        raise AssertionError("plan() reached load_raw")

    monkeypatch.setattr(io_module, "load_raw", explode)
    data, locations, travel = _problem()
    plan = _planner().plan(data, locations, travel, _Settings())
    assert sorted(plan["battery"]) == sorted(locations["battery"])


def test_plan_opens_no_files_at_all(monkeypatch):
    """The strongest form: no file is opened for reading during planning."""
    import builtins

    real_open = builtins.open
    opened: list[str] = []

    def watched_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watched_open)
    data, locations, travel = _problem()
    _planner().plan(data, locations, travel, _Settings())
    assert opened == [], f"plan() opened files: {opened}"


def test_planning_is_deterministic():
    """Two runs must give the identical plan - the submission is scored once,
    but reproducibility is a competition requirement."""
    data, locations, travel = _problem()
    first = _planner().plan(data, locations, travel, _Settings())
    second = _planner().plan(data, locations, travel, _Settings())
    pd.testing.assert_frame_equal(first, second)


def test_plan_survives_a_device_with_no_history():
    """locations may list a battery the truncated series does not cover; it
    still has to appear in the plan exactly once or the plan is rejected."""
    data, locations, travel = _problem()
    extra = pd.DataFrame({"battery": ["d_ghost"], "building": ["b_1"],
                          "room": ["r_9"], "start_time": pd.Timestamp("2024-01-01"),
                          "end_time": pd.Timestamp("2026-01-01")})
    locations = pd.concat([locations, extra], ignore_index=True)
    plan = _planner().plan(data, locations, travel, _Settings())
    assert sorted(plan["battery"]) == sorted(locations["battery"])
    assert not plan["battery"].duplicated().any()
