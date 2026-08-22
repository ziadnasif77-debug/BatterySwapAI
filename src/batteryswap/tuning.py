"""The q sweep — the single highest-value tuning knob in the system.

The newsvendor identity q* = c_early/(c_early+c_late)
is only optimal for ONE battery with NO travel cost. Once trips are shared
across a building, grouping already pulls batteries early; pushing deadlines
earlier on top of that is pure waste. So q is swept, not derived.

The objective passed in must be the official evaluator (or the
tolerance-verified surrogate). This module never scores anything itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .decision import days_at_q, newsvendor_q
from .optimizer import Schedule

ObjectiveFn = Callable[[Schedule], float]


def default_q_grid(q_star: float | None = None, n: int = 19) -> list[float]:
    """A grid spanning the decision-relevant range, with q* inserted so the
    theoretical value is always among the candidates it is compared against."""
    grid = list(np.round(np.linspace(0.05, 0.95, n), 4))
    if q_star is not None and np.isfinite(q_star):
        grid.append(round(float(np.clip(q_star, 0.01, 0.99)), 4))
    return sorted(set(grid))


def sweep_q(calibrated: pd.DataFrame, quantile_levels: list[float],
            horizon_days: int, objective: ObjectiveFn,
            q_grid: list[float] | None = None,
            post_process: Callable[[Schedule], Schedule] | None = None,
            c_early: float | None = None, c_late: float | None = None) -> pd.DataFrame:
    """Score the q-rule schedule at every q on the grid.

    `post_process` lets the caller apply the opportunistic-swap rule (or any
    optimizer stage) before scoring, so the sweep measures q as the full
    pipeline actually uses it — sweeping a rule you do not ship is a
    measurement of nothing.

    Returns a frame sorted by q with the final cost per q. ⛔ Final cost is the
    only selection column here.
    """
    if q_grid is None:
        q_star = (newsvendor_q(c_early, c_late)
                  if c_early is not None and c_late is not None else None)
        q_grid = default_q_grid(q_star)

    rows = []
    for q in q_grid:
        schedule = Schedule(assignments=days_at_q(calibrated, quantile_levels, q,
                                                  horizon_days))
        if post_process is not None:
            schedule = post_process(schedule)
        rows.append({"q": q, "final_cost": float(objective(schedule)),
                     "n_days_used": len(set(schedule.assignments.values()))})
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


def best_q(sweep: pd.DataFrame) -> tuple[float, float]:
    """(q, final_cost) at the sweep minimum."""
    row = sweep.loc[sweep["final_cost"].idxmin()]
    return float(row["q"]), float(row["final_cost"])
