"""EOL machinery: robust rolling statistic, sustained crossing, censoring."""

import numpy as np
import pandas as pd
import pytest

from batteryswap.config import UnknownValueError
from batteryswap.labels import compute_eol, label_batteries


def _series(voltages: list[float], start="2024-01-01") -> pd.DataFrame:
    t = pd.date_range(start, periods=len(voltages), freq="h")
    return pd.DataFrame({"timestamp": t, "battery_id": "x",
                         "voltage": voltages, "temperature": 20.0})


def test_transient_dip_does_not_trigger_eol():
    v = [3.0] * 200 + [2.0] * 3 + [3.0] * 200  # 3-hour dip, recovers fully
    assert compute_eol(_series(v), threshold=2.5) is None


def test_sustained_decline_triggers_eol():
    v = [3.0] * 300 + list(np.linspace(3.0, 1.8, 200)) + [1.8] * 100
    eol = compute_eol(_series(v), threshold=2.5)
    assert eol is not None
    # crossing happens inside the decline segment
    assert eol > pd.Timestamp("2024-01-01") + pd.Timedelta(hours=300)


def test_healthy_series_is_censored():
    v = [3.0] * 500
    assert compute_eol(_series(v), threshold=2.5) is None


def test_label_batteries_requires_official_threshold(synth_ts):
    """label_batteries pulls the threshold from cost_model.yaml and must raise
    while it is UNKNOWN — never guess (§18)."""
    with pytest.raises(UnknownValueError):
        label_batteries(synth_ts)
