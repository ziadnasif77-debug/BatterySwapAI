"""Deterministic, leak-free feature engineering (§8).

Contract:
- Every feature is computed from `history[history.timestamp <= cutoff]` only.
  The no-future test (tests/test_no_future.py) asserts byte-identical output
  whether or not post-cutoff rows exist in the frame.
- ⛔ Identifier columns never enter the matrix — assemble_matrix() raises if
  they do, and tests/test_no_leakage.py enforces it.
- Features needing the failure threshold are emitted only when the official
  threshold has been resolved (never guessed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import UnknownValueError, load_config
from .io import ID_COLUMNS

WINDOWS_DAYS = (7, 30, 90, 180)
HOURS = pd.Timedelta(hours=1)


def _theil_sen_slope(t_hours: np.ndarray, v: np.ndarray) -> float:
    """Robust slope in volts/hour; NaN if under 8 points."""
    if len(v) < 8:
        return np.nan
    # subsample for O(n^2) safety on long windows
    if len(v) > 500:
        idx = np.linspace(0, len(v) - 1, 500).astype(int)
        t_hours, v = t_hours[idx], v[idx]
    slope, _, _, _ = stats.theilslopes(v, t_hours)
    return float(slope)


def _window(history: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = cutoff - pd.Timedelta(days=days)
    return history[(history["timestamp"] > lo) & (history["timestamp"] <= cutoff)]


def compute_features(history: pd.DataFrame, cutoff: pd.Timestamp,
                     failure_threshold: float | None = None) -> dict[str, float]:
    """Feature vector for one battery at one cutoff. Pure function of the
    pre-cutoff history; returns a flat dict of floats."""
    h = history[history["timestamp"] <= cutoff].sort_values("timestamp")
    if h.empty:
        raise ValueError("no observations before cutoff")

    t0 = h["timestamp"].iloc[0]
    t_hours = ((h["timestamp"] - t0) / HOURS).to_numpy()
    v = h["voltage"].to_numpy()
    temp = h["temperature"].to_numpy()

    f: dict[str, float] = {}
    f["age_hours"] = float((cutoff - t0) / HOURS)
    f["obs_span_hours"] = float((h["timestamp"].iloc[-1] - t0) / HOURS)
    f["n_obs"] = float(len(h))
    f["month_sin"] = float(np.sin(2 * np.pi * cutoff.month / 12))
    f["month_cos"] = float(np.cos(2 * np.pi * cutoff.month / 12))

    # --- voltage levels -----------------------------------------------------
    recent = _window(h, cutoff, 7)["voltage"]
    f["v_now"] = float(recent.median()) if len(recent) else float(np.median(v[-24:]))
    init = h.head(min(len(h), 24 * 7))["voltage"]
    f["v_init"] = float(init.median())
    f["v_drop_total"] = f["v_init"] - f["v_now"]

    if failure_threshold is not None:
        f["v_above_threshold"] = f["v_now"] - failure_threshold

    # --- multi-scale slopes, spread, extremes -------------------------------
    slopes: dict[int, float] = {}
    for d in WINDOWS_DAYS:
        w = _window(h, cutoff, d)
        wt = ((w["timestamp"] - t0) / HOURS).to_numpy()
        wv = w["voltage"].to_numpy()
        s = _theil_sen_slope(wt, wv)
        slopes[d] = s
        f[f"v_slope_{d}d"] = s
        f[f"v_std_{d}d"] = float(np.std(wv)) if len(wv) > 2 else np.nan
        f[f"v_iqr_{d}d"] = float(np.subtract(*np.percentile(wv, [75, 25]))) if len(wv) > 4 else np.nan
        f[f"v_min_{d}d"] = float(np.min(wv)) if len(wv) else np.nan
        f[f"v_p10_{d}d"] = float(np.percentile(wv, 10)) if len(wv) > 4 else np.nan
        f[f"t_mean_{d}d"] = float(np.mean(w["temperature"])) if len(w) else np.nan
        f[f"t_std_{d}d"] = float(np.std(w["temperature"])) if len(w) > 2 else np.nan

    # acceleration: short-scale slope vs long-scale slope
    f["v_accel_30_180"] = slopes[30] - slopes[180]
    f["v_accel_7_90"] = slopes[7] - slopes[90]

    # --- knee indicator group (§8, physics-derived) --------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        f["knee_slope_ratio"] = (slopes[30] / slopes[180]
                                 if slopes[180] not in (0.0,) and np.isfinite(slopes[180])
                                 else np.nan)
    plateau = float(np.percentile(v, 60)) if len(v) > 50 else f["v_init"]
    f["dist_below_plateau"] = plateau - f["v_now"]

    # cumulative charge proxy: integral of (V0 - V) dt over the observed window
    if len(v) > 2:
        f["cum_charge_proxy"] = float(np.trapezoid(np.maximum(f["v_init"] - v, 0.0), t_hours))
    else:
        f["cum_charge_proxy"] = np.nan

    # --- temperature confound removal (§8, "the important group") ------------
    mask = np.isfinite(v) & np.isfinite(temp)
    if mask.sum() > 24:
        beta, alpha, *_ = stats.linregress(temp[mask], v[mask])
        resid = v - (alpha + beta * temp)
        f["temp_coeff"] = float(beta)
        f["v_resid_now"] = float(np.median(resid[-24 * 7:]))
        rw = t_hours >= (t_hours[-1] - 30 * 24)
        f["v_resid_slope_30d"] = _theil_sen_slope(t_hours[rw & mask], resid[rw & mask])
    else:
        f["temp_coeff"] = np.nan
        f["v_resid_now"] = np.nan
        f["v_resid_slope_30d"] = np.nan

    return f


def build_feature_table(ts: pd.DataFrame, cutoffs: pd.DataFrame,
                        failure_threshold: float | None = None,
                        id_column: str = "battery_id",
                        target_columns: tuple[str, ...] = ("rul_hours", "rul_days"),
                        ) -> pd.DataFrame:
    """Feature rows for (id, cutoff) pairs. Emits threshold-distance features
    only if a threshold is available — passed explicitly or resolved from the
    official cost model.

    `id_column` lets the same builder serve the synthetic harness
    (``battery_id``) and the official data (``device_id``); the series frame
    must carry a ``timestamp`` column either way.
    """
    threshold = failure_threshold
    if threshold is None:
        try:
            threshold = load_config("cost_model")["failure"]["voltage_threshold"]
        except UnknownValueError:
            threshold = None

    grouped = dict(tuple(ts.groupby(id_column)))
    rows = []
    for _, row in cutoffs.iterrows():
        key = row[id_column]
        feats = compute_features(grouped[key], row["cutoff"], threshold)
        feats[id_column] = key           # carried as INDEX metadata, stripped below
        feats["cutoff"] = row["cutoff"]
        for target in target_columns:
            if target in row.index:
                feats[target] = row[target]
        rows.append(feats)
    return pd.DataFrame(rows)


def assemble_matrix(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Split a feature table into (X, y, meta).

    ⛔ Raises if any identifier column would reach X. meta carries ids/cutoffs
    for grouping and joining — never for prediction.
    """
    meta_cols = [c for c in (*ID_COLUMNS, "cutoff", "rul_hours", "rul_days",
                             "censored", "scenario") if c in table.columns]
    meta = table[meta_cols].copy()
    target = next((c for c in ("rul_days", "rul_hours") if c in table.columns), None)
    y = table[target].copy() if target else None
    X = table.drop(columns=meta_cols, errors="ignore")

    leaked = [c for c in X.columns if c in ID_COLUMNS or c.endswith("_id")]
    if leaked:
        raise ValueError(f"Identifier columns must never reach the feature matrix: {leaked}")
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    return X, y, meta
