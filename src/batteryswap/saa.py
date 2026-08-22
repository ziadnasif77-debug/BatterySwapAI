"""Sample Average Approximation over lifetime scenarios.

Draw 50-200 failure-time scenarios per battery from the CALIBRATED
distributions and optimize expected cost across them, instead of against one
point deadline.

Why this exists even though decision.cost_curves already integrates over the
quantile grid: that integration assumes the penalty is a known analytic
function of (day - t_fail). The official penalty may be piecewise, capped, or
otherwise non-linear, and scenarios evaluate ANY penalty shape — including the
official scorer applied scenario by scenario. With linear penalties the two
agree in the limit, which is exactly the check `scenario_curves_match_analytic`
performs in the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import q_col


def draw_failure_scenarios(calibrated: pd.DataFrame, quantile_levels: list[float],
                           n_scenarios: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Inverse-CDF sampling over the calibrated quantile grid.

    Returns (battery_ids, hours[n_batteries, n_scenarios]). Deterministic given
    `seed` — a submission-path requirement.
    """
    rng = np.random.default_rng(seed)
    levels = np.asarray(quantile_levels, dtype=float)
    cols = [q_col(q) for q in quantile_levels]

    battery_ids = list(calibrated["battery_id"])
    u = rng.uniform(0.0, 1.0, size=(len(battery_ids), n_scenarios))
    grid = calibrated[cols].to_numpy(dtype=float)

    out = np.empty_like(u)
    for i in range(len(battery_ids)):
        out[i] = np.interp(u[i], levels, grid[i])
    return battery_ids, np.maximum(out, 0.0)


def scenario_cost_curves(battery_ids: list[str], scenarios: np.ndarray,
                         horizon_days: int, c_early: float, c_late: float
                         ) -> dict[str, np.ndarray]:
    """Per-battery expected penalty per day, averaged over scenarios.

    Swap this dict for the analytic one to make any optimizer scenario-based:
    the objective signature does not change.
    """
    days_hours = np.arange(1, horizon_days + 1, dtype=float) * 24.0
    curves = {}
    for i, b in enumerate(battery_ids):
        t = scenarios[i][None, :]
        d = days_hours[:, None]
        early = np.maximum(t - d, 0.0).mean(axis=1)
        late = np.maximum(d - t, 0.0).mean(axis=1)
        curves[b] = c_early * early + c_late * late
    return curves


def scenario_curves_frame(curves: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long frame (battery_id, day, expected_penalty) — same contract as
    decision.build_cost_curves, so downstream code is indifferent."""
    frames = [pd.DataFrame({"battery_id": b,
                            "day": np.arange(1, len(g) + 1),
                            "expected_penalty": g})
              for b, g in curves.items()]
    return pd.concat(frames, ignore_index=True)
