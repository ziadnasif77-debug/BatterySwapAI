"""ALNS — adaptive large neighborhood search, the primary improvement engine
. Ruin-and-recreate with adaptive operator weights and simulated-
annealing acceptance.

The objective callback must be the official evaluator or the verified
surrogate. Destroy/repair operators receive and return complete Schedules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import Schedule

ObjectiveFn = Callable[[Schedule], float]
DestroyFn = Callable[[Schedule, np.random.Generator, float], tuple[Schedule, list[str]]]
RepairFn = Callable[[Schedule, list[str], np.random.Generator], Schedule]


@dataclass
class ALNSConfig:
    iterations: int = 20000
    segment_length: int = 100
    reaction_factor: float = 0.2
    destroy_fraction: tuple[float, float] = (0.05, 0.3)
    initial_temperature_ratio: float = 0.05
    cooling_rate: float = 0.9997
    seed: int = 0


# --- destroy operators -------------------------------------------------------
# Each removes a subset of assignments and returns (partial schedule, removed).

def random_removal(schedule: Schedule, rng: np.random.Generator,
                   fraction: float) -> tuple[Schedule, list[str]]:
    batteries = list(schedule.assignments)
    n = max(1, int(len(batteries) * fraction))
    removed = list(rng.choice(batteries, size=n, replace=False))
    out = Schedule(assignments={b: d for b, d in schedule.assignments.items()
                                if b not in set(removed)})
    return out, removed


def day_removal(schedule: Schedule, rng: np.random.Generator,
                fraction: float) -> tuple[Schedule, list[str]]:
    days = sorted({d for d in schedule.assignments.values()})
    if not days:
        return schedule, []
    day = int(rng.choice(days))
    removed = [b for b, d in schedule.assignments.items() if d == day]
    out = Schedule(assignments={b: d for b, d in schedule.assignments.items() if d != day})
    return out, removed


def make_building_removal(battery_to_building: dict[str, str]) -> DestroyFn:
    """Building removal needs the location mapping — factory keeps operators
    signature-uniform."""
    def building_removal(schedule: Schedule, rng: np.random.Generator,
                         fraction: float) -> tuple[Schedule, list[str]]:
        buildings = sorted({battery_to_building[b] for b in schedule.assignments})
        if not buildings:
            return schedule, []
        target = rng.choice(buildings)
        removed = [b for b in schedule.assignments if battery_to_building[b] == target]
        out = Schedule(assignments={b: d for b, d in schedule.assignments.items()
                                    if battery_to_building[b] != target})
        return out, removed
    return building_removal


def make_worst_cost_removal(objective: ObjectiveFn) -> DestroyFn:
    """Remove the assignments whose individual removal improves cost most.
    O(n) objective calls per invocation — use the fast surrogate."""
    def worst_cost_removal(schedule: Schedule, rng: np.random.Generator,
                           fraction: float) -> tuple[Schedule, list[str]]:
        batteries = list(schedule.assignments)
        base = objective(schedule)
        deltas = []
        for b in batteries:
            trial = Schedule(assignments={k: v for k, v in schedule.assignments.items() if k != b})
            deltas.append((base - objective(trial), b))
        deltas.sort(reverse=True)
        n = max(1, int(len(batteries) * fraction))
        removed = [b for _, b in deltas[:n]]
        out = Schedule(assignments={b: d for b, d in schedule.assignments.items()
                                    if b not in set(removed)})
        return out, removed
    return worst_cost_removal


# --- search loop -------------------------------------------------------------

def alns_search(initial: Schedule, objective: ObjectiveFn,
                destroy_ops: list[DestroyFn], repair_ops: list[RepairFn],
                config: ALNSConfig) -> tuple[Schedule, float]:
    rng = np.random.default_rng(config.seed)
    current, best = initial, initial
    current_cost = best_cost = objective(initial)

    temperature = max(config.initial_temperature_ratio * abs(best_cost), 1e-9)
    d_weights = np.ones(len(destroy_ops))
    r_weights = np.ones(len(repair_ops))
    d_scores = np.zeros(len(destroy_ops))
    r_scores = np.zeros(len(repair_ops))
    d_uses = np.zeros(len(destroy_ops))
    r_uses = np.zeros(len(repair_ops))

    for it in range(config.iterations):
        di = rng.choice(len(destroy_ops), p=d_weights / d_weights.sum())
        ri = rng.choice(len(repair_ops), p=r_weights / r_weights.sum())
        fraction = rng.uniform(*config.destroy_fraction)

        partial, removed = destroy_ops[di](current, rng, fraction)
        candidate = repair_ops[ri](partial, removed, rng)
        cand_cost = objective(candidate)

        d_uses[di] += 1
        r_uses[ri] += 1

        accept = cand_cost < current_cost or \
            rng.random() < np.exp(-(cand_cost - current_cost) / temperature)
        if accept:
            current, current_cost = candidate, cand_cost
            reward = 2.0
            if cand_cost < best_cost:
                best, best_cost = candidate, cand_cost
                reward = 5.0
            d_scores[di] += reward
            r_scores[ri] += reward

        temperature *= config.cooling_rate

        if (it + 1) % config.segment_length == 0:
            for weights, scores, uses in ((d_weights, d_scores, d_uses),
                                          (r_weights, r_scores, r_uses)):
                used = uses > 0
                weights[used] = ((1 - config.reaction_factor) * weights[used]
                                 + config.reaction_factor * scores[used] / uses[used])
                weights += 1e-6
                scores[:] = 0
                uses[:] = 0

    return best, best_cost
