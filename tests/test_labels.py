"""EOL machinery: robust rolling statistic, sustained crossing, censoring."""

import numpy as np
import pandas as pd
import pytest

from batteryswap.labels import (
    FAILURE_VOLTAGE,
    compute_eol,
    eol_days_relative,
    ground_truth_eol,
    time_column,
)


def _series(voltages: list[float], device="x", start="2024-01-01",
            column="timestamp") -> pd.DataFrame:
    t = pd.date_range(start, periods=len(voltages), freq="h")
    return pd.DataFrame({column: t, "device_id": device,
                         "voltage": voltages, "temperature": 20.0})


def test_transient_dip_does_not_trigger_eol():
    v = [3.0] * 200 + [2.0] * 3 + [3.0] * 200  # 3-hour dip, recovers fully
    assert compute_eol(_series(v), threshold=2.5) is None


def test_sustained_decline_triggers_eol():
    v = [3.0] * 300 + list(np.linspace(3.0, 1.8, 200)) + [1.8] * 100
    eol = compute_eol(_series(v), threshold=2.5)
    assert eol is not None
    assert eol > pd.Timestamp("2024-01-01") + pd.Timedelta(hours=300)


def test_healthy_series_is_censored():
    assert compute_eol(_series([3.0] * 500), threshold=2.5) is None


def test_time_column_accepts_both_conventions():
    assert time_column(_series([3.0] * 10, column="timestamp")) == "timestamp"
    assert time_column(_series([3.0] * 10, column="end_time")) == "end_time"
    with pytest.raises(KeyError):
        time_column(pd.DataFrame({"voltage": [3.0]}))


def _fixture_frames():
    """One labelled device, one that only crosses the threshold, one healthy."""
    dead = _series(list(np.linspace(3.0, 2.0, 400)), device="d_dead", column="end_time")
    crossing = _series(list(np.linspace(3.0, 2.0, 400)), device="d_cross", column="end_time")
    healthy = _series([3.0] * 400, device="d_ok", column="end_time")
    metrics = pd.concat([dead, crossing, healthy], ignore_index=True)
    last = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=399)
    devices = pd.DataFrame({"device_id": ["d_dead", "d_cross", "d_ok"],
                            "end_time": [last] * 3})
    eol = pd.DataFrame({"device_id": ["d_dead", "d_cross", "d_ok"],
                        "eol_time": [pd.Timestamp("2024-01-10", tz="UTC"),
                                     pd.NaT, pd.NaT]})
    return metrics, devices, eol


def test_ground_truth_prefers_the_recorded_label_over_the_threshold():
    metrics, devices, eol = _fixture_frames()
    truth = ground_truth_eol(metrics, devices, eol, threshold=2.5).set_index("device_id")
    assert truth.loc["d_dead", "source"] == "label"
    assert truth.loc["d_dead", "eol_time"] == pd.Timestamp("2024-01-10", tz="UTC")
    assert truth.loc["d_cross", "source"] == "threshold"
    assert truth.loc["d_ok", "source"] == "censored"
    assert bool(truth.loc["d_ok", "censored"])


def test_censored_devices_are_placed_beyond_the_horizon():
    """A device with no evidence of failure must be free to skip (its EOL sits
    past the horizon) yet still finite, so replacing it costs early penalty."""
    metrics, devices, eol = _fixture_frames()
    truth = ground_truth_eol(metrics, devices, eol, threshold=2.5)
    days = eol_days_relative(truth, pd.Timestamp("2024-01-05"),
                             horizon_days=42, grace_days=30.0)
    assert days["d_ok"] >= 42
    assert np.isfinite(days["d_ok"])
    assert days["d_dead"] == pytest.approx(5.0)


def test_calibrated_threshold_is_the_documented_value():
    """FAILURE_VOLTAGE is a measured estimate, not an official constant — pin
    it so a silent edit cannot change every label downstream."""
    assert FAILURE_VOLTAGE == 2.42
