"""The opportunistic-swap rule.

When a worker is already in building B on day d, also replace every battery
there whose loss from being serviced early is smaller than its share of a
future dedicated trip:

    g_b(d) - min_d' g_b(d')  <  amortized marginal cost of a future visit to B

⛔ The amortization is the whole rule. Charging the FULL trip cost to a single
battery makes the threshold so generous that the schedule collapses into
"replace everything at the first visit". The cost is therefore divided across
the batteries that would otherwise share that future trip.

⛔ The trip cost is PER BUILDING: a visit to the far cluster justifies
sacrificing far more remaining life than a visit next door. Callers pass a
per-building dict measured from the evaluator sensitivity study.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import Schedule


def opportunistic_upgrade(schedule: Schedule, curves: dict[str, np.ndarray],
                          battery_building: dict[str, str],
                          trip_cost: dict[str, float]) -> tuple[Schedule, int]:
    """One deterministic pass over the (building, day) visits already planned.

    Returns (upgraded schedule, number of batteries moved). Single pass by
    design: each move changes the amortization base, and iterating to a fixed
    point re-introduces exactly the collapse the rule guards against. The gain
    is measured against the evaluator, never assumed.
    """
    assignments = dict(schedule.assignments)

    members: dict[str, list[str]] = defaultdict(list)
    for b in assignments:
        members[battery_building[b]].append(b)

    moved = 0
    for building in sorted(members):
        crew = sorted(members[building])
        for day in sorted({assignments[b] for b in crew}):
            if all(assignments[b] != day for b in crew):
                continue  # this visit was emptied by an earlier move
            # batteries in this building not served on this day — they are the
            # ones that would otherwise share a future dedicated trip
            candidates = [b for b in crew if assignments[b] != day]
            if not candidates:
                continue
            threshold = trip_cost.get(building, 0.0) / len(candidates)
            for b in candidates:
                curve = curves[b]
                # extra penalty from being serviced on this visit instead of
                # where it currently sits; negative means strictly better
                delta = float(curve[day - 1] - curve[assignments[b] - 1])
                if delta < threshold:
                    assignments[b] = day
                    moved += 1

    return Schedule(assignments=assignments), moved


def per_building_trip_cost(buildings: list[str], travel: object,
                           building_overhead_minutes: float,
                           minute_cost: float) -> dict[str, float]:
    """Proxy for the marginal cost of one additional dedicated visit.

    ⛔ PROXY ONLY: building overhead plus a round trip at the building's mean
    travel leg. Phase 0 replaces this with the MEASURED marginal cost per
    building from the evaluator sensitivity study — add and remove a
    trip, read the curve.
    """
    out = {}
    for bl in buildings:
        legs = travel.loc[bl].drop(index=bl)
        out[bl] = minute_cost * (building_overhead_minutes + 2.0 * float(legs.mean()))
    return out
