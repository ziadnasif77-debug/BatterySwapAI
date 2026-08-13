"""End-of-life definition, RUL computation, and censoring flags.

⛔ The EOL definition is owned by the competition. This module implements the
MACHINERY (robust rolling statistic + threshold crossing + censoring) but the
threshold/rule itself comes from configs/cost_model.yaml and raises until it
has been transcribed from the official evaluator. Never define EOL on a raw
sample (§1.4 of the brief): noise and temperature-coupled recovery make raw
crossings meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import load_config

ROLLING_WINDOW = "24h"  # robust statistic window; verify against official definition


@dataclass
class BatteryLabel:
    battery_id: str
    eol_time: pd.Timestamp | None   # None => right-censored
    censored: bool
    last_observed: pd.Timestamp


def rolling_voltage(series: pd.DataFrame, window: str = ROLLING_WINDOW) -> pd.Series:
    """24-hour rolling median of voltage, indexed by timestamp."""
    s = series.set_index("timestamp")["voltage"].sort_index()
    return s.rolling(window, min_periods=6).median()


def compute_eol(series: pd.DataFrame, threshold: float,
                window: str = ROLLING_WINDOW) -> pd.Timestamp | None:
    """First time the rolling-median voltage falls to/below `threshold` and
    never recovers above it for the remainder of the series (sustained
    crossing, so a transient dip does not create a false EOL)."""
    rv = rolling_voltage(series, window).dropna()
    below = rv <= threshold
    if not below.any():
        return None
    # Last index where voltage is above threshold; EOL is the first sustained
    # crossing after it. If the series ends below threshold from some point on,
    # that point is EOL.
    above_idx = rv.index[~below]
    if len(above_idx) == 0:
        return rv.index[0]
    last_above = above_idx[-1]
    after = rv.index[(rv.index > last_above) & below]
    return after[0] if len(after) else None


def label_batteries(train: pd.DataFrame) -> pd.DataFrame:
    """Per-battery EOL and censoring flags for the whole training set.

    Raises UnknownValueError until the official EOL rule is transcribed.
    Right-censored batteries are KEPT and flagged — dropping them biases the
    lifetime distribution short (§7).
    """
    cost_model = load_config("cost_model")
    threshold = cost_model["failure"]["voltage_threshold"]  # raises if UNKNOWN

    records = []
    for battery_id, group in train.groupby("battery_id", sort=True):
        eol = compute_eol(group, threshold)
        records.append({
            "battery_id": battery_id,
            "eol_time": eol,
            "censored": eol is None,
            "last_observed": group["timestamp"].max(),
        })
    return pd.DataFrame.from_records(records)


def rul_hours(labels: pd.DataFrame, battery_id: str, at: pd.Timestamp) -> float | None:
    """True remaining useful life in hours at time `at`; None if censored."""
    row = labels.loc[labels["battery_id"] == battery_id]
    if row.empty or bool(row["censored"].iloc[0]):
        return None
    return (row["eol_time"].iloc[0] - at) / pd.Timedelta(hours=1)
