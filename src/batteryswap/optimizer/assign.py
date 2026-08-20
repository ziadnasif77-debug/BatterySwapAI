"""Building -> day -> worker assignment via CP-SAT (§12.3), used when the
subproblem is small enough to solve exactly.

The model is deliberately deferred: its constraints (worker count, day cap,
overtime shape) are all UNKNOWN cost-model constants, and building it before
Phase 0 would mean inventing them. The function signature and decomposition
are fixed now so ALNS can call into it once the constants exist.
"""

from __future__ import annotations

import pandas as pd

from ..config import load_config


def cpsat_assign(buildings: list[str], candidate_days: list[int],
                 building_penalty: pd.DataFrame, travel: pd.DataFrame,
                 time_limit_seconds: int = 60) -> dict[str, tuple[int, int]]:
    """Exact assignment: building -> (day, worker).

    `building_penalty`: buildings x days aggregated expected penalty (from the
    decision layer). Raises UnknownValueError until workforce constants are
    transcribed — this is intentional (no-invention rule).
    """
    cm = load_config("cost_model")
    n_workers = cm["workforce"]["worker_count"]
    day_cap = cm["workforce"]["worker_limit_daily_hours"]
    week_cap = cm["workforce"]["worker_limit_weekly_hours"]  # the binding one

    from ortools.sat.python import cp_model  # imported late: heavy dependency

    raise NotImplementedError(
        f"CP-SAT model construction is deferred (§20 cut order). Constants load "
        f"OK: workers={n_workers}, day_cap={day_cap}h, week_cap={week_cap}h. "
        f"cp_model={cp_model.__name__} available."
    )
