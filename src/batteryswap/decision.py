"""Phase 6 — the decision layer bridging predictions and optimization (§11).

g_b(d) = c_early * E[max(0, t_fail - d)] + c_late * E[max(0, d - t_fail)]

computed numerically from calibrated quantiles. Output: an
[n_batteries x n_days] expected-penalty matrix, fully determined before any
scheduling happens.

c_early / c_late come from configs/cost_model.yaml, transcribed from the
official release (scenarios.json): 0.5 and 10.0 cost per DAY respectively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def newsvendor_q(c_early: float, c_late: float) -> float:
    """The single-battery optimal quantile. Only correct with no travel cost —
    treat as an initializer for the tunable q, not the answer (§11)."""
    return c_early / (c_early + c_late)


def failure_hours_at_q(q_row: np.ndarray, quantile_levels: np.ndarray, q: float) -> float:
    """Calibrated failure time at level q, by interpolation on the quantile grid."""
    return float(np.interp(q, quantile_levels, q_row))


def days_at_q(calibrated: pd.DataFrame, quantile_levels: list[float], q: float,
              horizon_days: int) -> dict[str, int]:
    """The q-rule: replace each battery at the q-quantile of its calibrated
    failure-time distribution, floored to a whole day (replace on or before)
    and clipped into the horizon.

    ⛔ q is a TUNABLE scalar, not the newsvendor value (§11). Initialize it at
    newsvendor_q() and sweep it against the evaluator — once trips are shared
    across a building, grouping already pulls batteries early, so the true
    optimum sits materially above the theoretical q*.
    """
    levels = np.asarray(quantile_levels, dtype=float)
    cols = [f"q{int(round(x * 100)):02d}" for x in quantile_levels]
    out = {}
    for _, row in calibrated.iterrows():
        hours = failure_hours_at_q(row[cols].to_numpy(dtype=float), levels, q)
        day = int(np.floor(hours / 24.0))
        out[str(row["battery_id"])] = int(np.clip(day, 1, horizon_days))
    return out


def quantile_grid_to_samples(q_row: np.ndarray, quantile_levels: np.ndarray,
                             n_samples: int = 400) -> np.ndarray:
    """Approximate the failure-time distribution by inverse-CDF interpolation
    over the calibrated quantile grid, with linear tails clamped at zero."""
    u = (np.arange(n_samples) + 0.5) / n_samples
    return np.maximum(np.interp(u, quantile_levels, q_row), 0.0)


def cost_curve(fail_hours_samples: np.ndarray, candidate_days: np.ndarray,
               c_early: float, c_late: float) -> np.ndarray:
    """Expected penalty per candidate replacement day for one battery.

    `fail_hours_samples` and the day grid are in hours/days; `c_early` and
    `c_late` are therefore per-HOUR rates (callers divide the official
    per-day constants by 24). The official penalties are linear on both
    sides — confirmed only as far as the constant names go; the exact
    functional form still needs the official scorer.
    """
    d_hours = candidate_days[:, None] * 24.0
    t = fail_hours_samples[None, :]
    early = np.maximum(t - d_hours, 0.0).mean(axis=1)
    late = np.maximum(d_hours - t, 0.0).mean(axis=1)
    return c_early * early + c_late * late


def cost_curves_from_constants(calibrated: pd.DataFrame, quantile_levels: list[float],
                               c_early: float, c_late: float,
                               horizon_days: int) -> pd.DataFrame:
    """Cost curves for every battery: long frame (battery_id, day, expected_penalty).

    Pure function of its arguments — the config-guarded entry point is
    build_cost_curves(); this form also serves the synthetic dry-run harness.
    """
    levels = np.asarray(quantile_levels, dtype=float)
    q_cols = [f"q{int(round(q * 100)):02d}" for q in quantile_levels]
    days = np.arange(1, horizon_days + 1, dtype=float)

    frames = []
    for _, row in calibrated.iterrows():
        samples = quantile_grid_to_samples(row[q_cols].to_numpy(dtype=float), levels)
        g = cost_curve(samples, days, c_early, c_late)
        frames.append(pd.DataFrame({"battery_id": row["battery_id"],
                                    "day": days.astype(int),
                                    "expected_penalty": g}))
    return pd.concat(frames, ignore_index=True)


def build_cost_curves(calibrated: pd.DataFrame, quantile_levels: list[float],
                      horizon_days: int | None = None) -> pd.DataFrame:
    """Config-guarded entry point: penalties/horizon come from the official
    cost model and raise until transcribed (no-invention rule)."""
    cm = load_config("cost_model")
    # Official units are cost per DAY (scenarios.json), so the curves below are
    # built on a per-day grid; cost_curve() converts the hour-based failure
    # samples accordingly.
    c_early = cm["penalties"]["early_replacement_per_day"] / 24.0
    c_late = cm["penalties"]["late_replacement_per_day"] / 24.0
    if horizon_days is None:
        horizon_days = cm["horizon"]["planning_days"]            # raises if UNKNOWN
    return cost_curves_from_constants(calibrated, quantile_levels, c_early, c_late,
                                      horizon_days)
