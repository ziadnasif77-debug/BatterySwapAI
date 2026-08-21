"""Per-day sequencing under the daily hour cap (§12.4).

Given the buildings to visit on one day and per-building service time
(swaps + room overheads), sequence them per worker: nearest-neighbor
construction + 2-opt improvement on the travel matrix, then feasibility
accounting against the working-day cap (overtime = cost, not prohibition,
per the official rules once transcribed).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pandas as pd


@dataclass
class DayPlan:
    order: list[str]                  # building visit order
    travel_minutes: float
    service_minutes: float
    total_minutes: float
    overtime_minutes: float


def nearest_neighbor_order(buildings: list[str], travel: pd.DataFrame,
                           start: str | None = None) -> list[str]:
    """Depot semantics are UNKNOWN until the official evaluator is read —
    if there is a depot, `start` must be set to it in Phase 0."""
    if not buildings:
        return []
    remaining = list(buildings)
    order = []
    current = start if start in remaining else remaining[0]
    remaining.remove(current)
    order.append(current)
    while remaining:
        row = travel.loc[current, remaining]
        current = row.idxmin()
        remaining.remove(current)
        order.append(current)
    return order


def two_opt(order: list[str], travel: pd.DataFrame, max_passes: int = 20) -> list[str]:
    def leg(a: str, b: str) -> float:
        return float(travel.loc[a, b])

    best = list(order)
    for _ in range(max_passes):
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                before = leg(best[i - 1], best[i]) + (leg(best[j], best[j + 1])
                                                      if j + 1 < len(best) else 0.0)
                after = leg(best[i - 1], best[j]) + (leg(best[i], best[j + 1])
                                                     if j + 1 < len(best) else 0.0)
                if after + 1e-12 < before:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
        if not improved:
            break
    return best


def route_length_minutes(order: list[str], travel: pd.DataFrame) -> float:
    return float(sum(travel.loc[a, b] for a, b in pairwise(order)))


def plan_day(buildings: list[str], service_minutes: dict[str, float],
             travel: pd.DataFrame, day_cap_minutes: float,
             start: str | None = None) -> DayPlan:
    order = two_opt(nearest_neighbor_order(buildings, travel, start), travel)
    travel_min = route_length_minutes(order, travel)
    service_min = float(sum(service_minutes[b] for b in order))
    total = travel_min + service_min
    return DayPlan(order=order, travel_minutes=travel_min, service_minutes=service_min,
                   total_minutes=total, overtime_minutes=max(0.0, total - day_cap_minutes))
