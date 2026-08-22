"""End-of-life ground truth, RUL computation, and censoring flags.

The official release supplies `eol_times.csv`, but only 82 of 461 devices
(17.8 %) carry a recorded EOL - the other 379 are right-censored. Dropping
them would bias the lifetime distribution short, and assuming they
all fail would invent failures the data does not show. So ground truth is
assembled in three tiers, most authoritative first:

    1. label     - the recorded EOL from eol_times.csv
    2. threshold - first SUSTAINED crossing of the failure voltage on a 24 h
                   rolling median (never on a raw sample: noise and
                   temperature-coupled recovery make raw crossings meaningless)
    3. censored  - no evidence of failure; the device is treated as not
                   failing inside any planning window (see A6 in scoring.py)

Tier 2 is calibrated against tier 1: `threshold_agreement()` reports how well
a candidate threshold reproduces the 82 known labels, and that measurement -
not intuition - picks the constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLLING_WINDOW = "24h"

# Chosen by threshold_agreement() against the 82 recorded labels, swept over
# 2.30-2.50: 2.42 is the least biased (median error -0.7 d) and the tightest
# (MAD 10.1 d, 41% within 7 d), while declaring only 13 extra unlabelled
# devices dead. See reports/data_audit.md.
# NOT an official constant - the official EOL rule was never published, so
# this is a data-derived ESTIMATE of it.
FAILURE_VOLTAGE = 2.42


@dataclass
class BatteryLabel:
    device_id: str
    eol_time: pd.Timestamp | None   # None => no observed failure
    censored: bool
    source: str                     # label | threshold | censored
    last_observed: pd.Timestamp


def time_column(series: pd.DataFrame) -> str:
    """The official metrics frame names its timestamp `end_time` (the end of
    each one-hour averaging window); the synthetic harness uses `timestamp`."""
    for candidate in ("end_time", "timestamp"):
        if candidate in series.columns:
            return candidate
    raise KeyError(f"no time column in {sorted(series.columns)}")


def rolling_voltage(series: pd.DataFrame, window: str = ROLLING_WINDOW) -> pd.Series:
    """Rolling median of voltage indexed by time. Robust by construction."""
    s = series.set_index(time_column(series))["voltage"].sort_index()
    return s.rolling(window, min_periods=6).median()


def sustained_crossing(rv: pd.Series, threshold: float) -> pd.Timestamp | None:
    """First time the rolling median falls to/below `threshold` and never
    recovers above it, so a transient dip cannot create a false EOL."""
    rv = rv.dropna()
    below = rv <= threshold
    if not below.any():
        return None
    above_idx = rv.index[~below]
    if len(above_idx) == 0:
        return rv.index[0]
    after = rv.index[(rv.index > above_idx[-1]) & below]
    return after[0] if len(after) else None


def compute_eol(series: pd.DataFrame, threshold: float,
                window: str = ROLLING_WINDOW) -> pd.Timestamp | None:
    return sustained_crossing(rolling_voltage(series, window), threshold)


def _as_utc(t) -> pd.Timestamp | None:
    if t is None or pd.isna(t):
        return None
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def ground_truth_eol(metrics: pd.DataFrame, devices: pd.DataFrame,
                     eol: pd.DataFrame, threshold: float = FAILURE_VOLTAGE,
                     window: str = ROLLING_WINDOW) -> pd.DataFrame:
    """Per-device EOL ground truth across the three tiers.

    Returns columns: device_id, eol_time (NaT when censored), censored,
    source, last_observed.
    """
    labelled = {r.device_id: r.eol_time for r in eol.itertuples()}
    last_seen = dict(zip(devices["device_id"], devices["end_time"]))

    records = []
    for device_id, group in metrics.groupby("device_id", sort=True):
        recorded = _as_utc(labelled.get(device_id))
        if recorded is not None:
            t, source = recorded, "label"
        else:
            crossed = _as_utc(compute_eol(group, threshold, window))
            t, source = (crossed, "threshold") if crossed is not None else (None, "censored")
        records.append({"device_id": device_id, "eol_time": t,
                        "censored": t is None, "source": source,
                        "last_observed": _as_utc(last_seen.get(device_id))})
    out = pd.DataFrame.from_records(records)
    out["eol_time"] = pd.to_datetime(out["eol_time"], utc=True)
    out["last_observed"] = pd.to_datetime(out["last_observed"], utc=True)
    return out


def threshold_agreement(metrics: pd.DataFrame, eol: pd.DataFrame,
                        thresholds=(2.30, 2.35, 2.40, 2.45, 2.50),
                        window: str = ROLLING_WINDOW) -> pd.DataFrame:
    """How well does each candidate threshold reproduce the recorded labels?

    This is the measurement that picks FAILURE_VOLTAGE. Reported columns:
    matched devices, median/MAD day error against the label, share within
    7 and 30 days, and how many unlabelled devices the threshold would
    additionally declare dead.
    """
    labelled = {r.device_id: _as_utc(r.eol_time) for r in eol.itertuples()
                if pd.notna(r.eol_time)}
    rows = []
    cache = {d: rolling_voltage(g, window) for d, g in metrics.groupby("device_id", sort=True)}
    for thr in thresholds:
        errors, extra, missed = [], 0, 0
        for device_id, rv in cache.items():
            crossed = _as_utc(sustained_crossing(rv, thr))
            label = labelled.get(device_id)
            if label is not None:
                if crossed is None:
                    missed += 1
                else:
                    errors.append((crossed - label) / pd.Timedelta(days=1))
            elif crossed is not None:
                extra += 1
        e = np.asarray(errors, dtype=float)
        rows.append({
            "threshold": thr,
            "matched": len(e),
            "label_missed": missed,
            "extra_unlabelled": extra,
            "median_err_days": float(np.median(e)) if len(e) else np.nan,
            "mad_days": float(np.median(np.abs(e - np.median(e)))) if len(e) else np.nan,
            "within_7d": float(np.mean(np.abs(e) < 7)) if len(e) else np.nan,
            "within_30d": float(np.mean(np.abs(e) < 30)) if len(e) else np.nan,
        })
    return pd.DataFrame(rows)


def eol_days_relative(truth: pd.DataFrame, start: pd.Timestamp, horizon_days: int,
                      grace_days: float) -> dict[str, float]:
    """Failure day of every device relative to a scenario start.

    Censored devices (no label, no crossing) get a value guaranteed to sit
    beyond the horizon, so skipping them is free, while still being finite so
    that replacing one carries a real early penalty (A6 in scoring.py).
    """
    start = _as_utc(start)
    out: dict[str, float] = {}
    for row in truth.itertuples():
        if pd.notna(row.eol_time):
            out[row.device_id] = (row.eol_time - start) / pd.Timedelta(days=1)
        else:
            bound = (row.last_observed + pd.Timedelta(days=grace_days) - start)
            out[row.device_id] = max(float(horizon_days) + grace_days,
                                     bound / pd.Timedelta(days=1))
    return out


def rul_days(truth: pd.DataFrame, device_id: str, at: pd.Timestamp) -> float | None:
    """True remaining useful life in days at `at`; None if censored."""
    row = truth.loc[truth["device_id"] == device_id]
    if row.empty or bool(row["censored"].iloc[0]):
        return None
    return (row["eol_time"].iloc[0] - _as_utc(at)) / pd.Timedelta(days=1)
