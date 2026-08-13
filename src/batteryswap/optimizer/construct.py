"""Construction heuristic: regret-k insertion (§12.1).

Regret-k: at each step, for every unassigned battery compute the cost of its
best insertion and its k-th best; insert the battery with the LARGEST regret
(best minus k-th best) at its best position. Substantially better than greedy
because it commits scarce slots to the batteries that lose the most by
waiting.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import Schedule

# insertion_cost(schedule, battery_id, day) -> float
InsertionCostFn = Callable[[Schedule, str, int], float]


def regret_k_construct(battery_ids: list[str], candidate_days: list[int],
                       insertion_cost: InsertionCostFn, k: int = 3) -> Schedule:
    """Build a schedule by repeated regret-k insertion.

    `insertion_cost` must be backed by the official evaluator or the
    tolerance-verified surrogate — it is the ONLY place cost enters.
    """
    schedule = Schedule()
    unassigned = list(battery_ids)

    while unassigned:
        best_battery, best_day = None, None
        best_regret, best_cost = -np.inf, np.inf

        for b in unassigned:
            costs = sorted((insertion_cost(schedule, b, d), d) for d in candidate_days)
            top_cost, top_day = costs[0]
            kth_cost = costs[min(k - 1, len(costs) - 1)][0]
            regret = kth_cost - top_cost
            if (regret, -top_cost) > (best_regret, -best_cost):
                best_regret, best_cost = regret, top_cost
                best_battery, best_day = b, top_day

        schedule.assignments[best_battery] = best_day
        unassigned.remove(best_battery)

    return schedule
