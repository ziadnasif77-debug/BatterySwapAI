"""Phase 1 — data audit (§6). Produces reports/data_audit.md.

Covers: schema/dtypes as loaded; row counts per battery/room/building;
sampling regularity and gaps; missing values; distributions; lifetime
distribution; installation clustering; evaluation truncation shape; plus the
two protective audits — knee detection and the seasonal (voltage~temperature)
confound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .io import RawData


def gap_distribution(ts: pd.DataFrame) -> pd.Series:
    """Distribution of sampling gaps in hours across all batteries."""
    gaps = (ts.sort_values(["battery_id", "timestamp"])
              .groupby("battery_id")["timestamp"].diff()
              .dropna() / pd.Timedelta(hours=1))
    return gaps.describe(percentiles=[0.5, 0.9, 0.99, 0.999])

def temperature_confound(ts: pd.DataFrame) -> pd.DataFrame:
    """Per-battery OLS of voltage on temperature. If the coefficient
    distribution is materially non-zero, every raw-voltage feature is partly a
    thermometer (§6)."""
    rows = []
    for b, g in ts.groupby("battery_id"):
        mask = g["voltage"].notna() & g["temperature"].notna()
        if mask.sum() < 48:
            continue
        res = stats.linregress(g.loc[mask, "temperature"], g.loc[mask, "voltage"])
        rows.append({"battery_id": b, "coef_v_per_degC": res.slope,
                     "r2": res.rvalue ** 2, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def knee_report(ts: pd.DataFrame, labels: pd.DataFrame,
                short_days: int = 30, long_days: int = 180,
                ratio_threshold: float = 3.0) -> pd.DataFrame:
    """Locate, per non-censored battery, when the 30d/180d slope ratio first
    exceeds `ratio_threshold` (decay acceleration), and report
    time_from_knee_to_EOL. This is the window the model must resolve."""
    from .features import _theil_sen_slope

    rows = []
    eol = labels.set_index("battery_id")["eol_time"]
    for b, g in ts.groupby("battery_id"):
        if b not in eol.index or pd.isna(eol[b]):
            continue
        g = g.sort_values("timestamp")
        t0 = g["timestamp"].iloc[0]
        th = ((g["timestamp"] - t0) / pd.Timedelta(hours=1)).to_numpy()
        v = g["voltage"].to_numpy()
        knee_time = None
        # weekly scan is enough resolution for a months-long knee window
        for cut in pd.date_range(t0 + pd.Timedelta(days=long_days),
                                 g["timestamp"].iloc[-1], freq="7D"):
            cut_h = (cut - t0) / pd.Timedelta(hours=1)
            recent = (th > cut_h - short_days * 24) & (th <= cut_h)
            longw = (th > cut_h - long_days * 24) & (th <= cut_h)
            s_short = _theil_sen_slope(th[recent], v[recent])
            s_long = _theil_sen_slope(th[longw], v[longw])
            if (np.isfinite(s_short) and np.isfinite(s_long) and s_long < 0
                    and s_short / s_long > ratio_threshold):
                knee_time = cut
                break
        rows.append({
            "battery_id": b,
            "knee_time": knee_time,
            "knee_to_eol_days": ((eol[b] - knee_time) / pd.Timedelta(days=1)
                                 if knee_time is not None else np.nan),
        })
    return pd.DataFrame(rows)


def write_audit_report(data: RawData, out_path: str | Path | None = None) -> Path:
    base = load_config("base")
    out = Path(out_path or Path(base["paths"]["reports_dir"]) / "data_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    train, ev, loc = data.train, data.evaluation, data.locations
    lines = ["# Data Audit", ""]

    lines += ["## Schema as loaded", "",
              f"- train: {dict(train.dtypes.astype(str))}",
              f"- evaluation: {dict(ev.dtypes.astype(str))}",
              f"- locations: {dict(loc.dtypes.astype(str))}",
              f"- travel matrix shape: {data.travel_minutes.shape}", ""]
    if data.issues:
        lines += ["## Structural issues", ""] + [f"- {i}" for i in data.issues] + [""]

    lines += ["## Counts", "",
              f"- train batteries: {train['battery_id'].nunique()}",
              f"- evaluation batteries: {ev['battery_id'].nunique()}",
              f"- buildings: {loc['building_id'].nunique()}",
              f"- rooms: {loc['room_id'].nunique()}",
              f"- batteries per room: {loc.groupby(['building_id', 'room_id']).size().describe().to_dict()}",
              ""]

    lines += ["## Sampling gaps (hours)", "", "```",
              gap_distribution(train).to_string(), "```", ""]

    spans = (ev.groupby('battery_id')['timestamp'].agg(['min', 'max'])
               .assign(span_days=lambda d: (d['max'] - d['min']) / pd.Timedelta(days=1)))
    lines += ["## Evaluation truncation shape (observed span, days)", "", "```",
              spans['span_days'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string(),
              "```", ""]

    confound = temperature_confound(train)
    if not confound.empty:
        lines += ["## Seasonal confound: voltage ~ temperature (per battery)", "", "```",
                  confound[["coef_v_per_degC", "r2"]].describe().to_string(), "```", "",
                  "If coef is materially non-zero, raw-voltage features are partly a "
                  "thermometer — residualized features are mandatory.", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
