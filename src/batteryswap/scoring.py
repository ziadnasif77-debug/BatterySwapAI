"""SUPERSEDED: our reimplementation of the cost function. SYNTHETIC USE ONLY.

⛔ Do not import this on the official path. The real scorer was located after
this module was written - it is the PyPI package ``batteryswap_public``, pulled
in by the official example repository's requirements - and ``evaluate.py`` now
calls it directly, as section 5.3 always required. This module survives only
because the synthetic dry run (``dryrun.py``) needs *a* cost model and must
keep running without the licensed dataset.

Where it was right and wrong, checked line by line against the official source
(``batteryswap_public/evaluate.py``):

    A1 currency (1 unit per technician-hour)      CORRECT - the official field
                                                  docs say "In hour-equivalent"
    A2 each day starts and ends at the depot      CORRECT
    A6 unobserved EOL = last data + grace days    CORRECT
    A4 overtime                                   WRONG - the official scorer
                                                  adds factor*overtime ON TOP
                                                  of the hours; this adds 1x
    A3 room/building overheads                    WRONG - official charges per
                                                  TRANSITION, not per visit
    A8 day indexing                               WRONG - days are DATES, not
                                                  1-based integers
    A9 late penalty capped at the window          WRONG - the official penalty
                                                  is uncapped in both
                                                  directions

``tests/test_evaluator_agreement.py`` measures the resulting divergence rather
than assuming it away.

CONSTANTS (certain - read verbatim from data/raw/scenarios.json)
    time_per_battery_hours 0.25, time_per_room_change_hours 0.5,
    time_per_building_change_hours 1.0, travel_costs[].hours,
    overtime_start 8.0 h/day, overtime_penalty_factor 2.0,
    worker_limit_daily_hours 24.0 (penalty 100),
    worker_limit_weekly_hours 24.0 (penalty 100),
    early 0.5/day, late 10.0/day, window 42 d, unobserved_eol_days 30.0,
    base_location + base_room = depot.

ASSUMPTIONS (uncertain - these are OURS, and each one can move the score)
    A1 CURRENCY. Work is charged at 1 cost unit per technician-hour, in the
       same currency as the daily penalties. The official file never states
       the exchange rate between hours and penalty units.
    A2 DAY SHAPE. Each working day starts and ends at the depot building.
    A3 OVERHEADS. building_change is charged once per building entered per
       day, room_change once per room entered. The alternative reading is
       once per transition (n-1); toggle with overhead_per_visit=False.
    A4 OVERTIME. Hours beyond overtime_start are charged at
       overtime_penalty_factor: cost = min(h,8) + factor * max(h-8, 0).
    A5 LIMITS. worker_limit_*_penalty is a flat charge per breached day or
       week, not per excess hour.
    A6 CENSORED DEVICES. A device with no observed EOL is assumed to fail
       unobserved_eol_days after its last observation.
    A7 NOT REPLACED. A device whose EOL falls inside the window but which is
       never replaced is charged late from its EOL to the end of the window.
    A8 DAY INDEXING. Days are 1..planning_window_days inclusive.
    A9 DOWNTIME IS IN-WINDOW ONLY. A device that failed before the window
       opened is charged only for the downtime inside it, never for the
       history that preceded it. Without this, replacing a long-dead device
       on day 1 would cost more than the entire window is worth, and no plan
       could beat doing nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import Scenario
from .optimizer.routing import nearest_neighbor_order, two_opt


@dataclass
class CostBreakdown:
    """Every component separately (section 3: reports/score_breakdown.json)."""
    early_penalty: float = 0.0
    late_penalty: float = 0.0
    work_cost: float = 0.0            # regular + overtime-weighted hours
    limit_penalty: float = 0.0        # daily + weekly cap breaches
    raw_hours: float = 0.0            # unweighted technician hours
    travel_hours: float = 0.0
    service_hours: float = 0.0
    overtime_hours: float = 0.0
    days_used: int = 0
    devices_replaced: int = 0
    devices_late: int = 0
    days_over_cap: int = 0
    weeks_over_cap: int = 0

    @property
    def total(self) -> float:
        return self.early_penalty + self.late_penalty + self.work_cost + self.limit_penalty

    def as_dict(self) -> dict:
        out = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in self.__dict__.items()}
        out["total"] = round(self.total, 4)
        return out


@dataclass
class ScenarioScorer:
    """Scores replacement plans for ONE scenario against known EOL times.

    eol_days maps device -> true failure day relative to the scenario start.
    It may be negative (already dead when the window opens) or exceed the
    horizon (will not fail during it).
    """
    scenario: Scenario
    location_of: dict[str, tuple[str, str]]
    eol_days: dict[str, float]
    overhead_per_visit: bool = True
    _work_cache: dict = field(default_factory=dict, repr=False)

    @property
    def s(self) -> dict:
        return self.scenario.settings

    @property
    def horizon(self) -> int:
        return self.scenario.horizon_days

    # ---- work time ---------------------------------------------------------
    def day_hours(self, devices: tuple[str, ...]) -> tuple[float, float]:
        """(travel_hours, service_hours) for one day's device set."""
        if not devices:
            return 0.0, 0.0
        key = frozenset(devices)
        hit = self._work_cache.get(key)
        if hit is not None:
            return hit

        by_building: dict[str, set] = defaultdict(set)
        for d in devices:
            b, r = self.location_of[d]
            by_building[b].add(r)

        travel = self.scenario.travel
        depot = self.scenario.depot
        buildings = sorted(by_building)
        # A2: depot -> ... -> depot, sequenced by NN + 2-opt over the visit set
        order = two_opt(nearest_neighbor_order(buildings, travel), travel)
        legs = [depot, *order, depot]
        travel_h = float(sum(travel.loc[a, b] for a, b in zip(legs[:-1], legs[1:])))

        n_b = len(buildings)
        n_r = sum(len(rooms) for rooms in by_building.values())
        if not self.overhead_per_visit:          # A3 alternative reading
            n_b = max(n_b - 1, 0)
            n_r = sum(max(len(rooms) - 1, 0) for rooms in by_building.values())
        service_h = (n_b * self.s["time_per_building_change_hours"]
                     + n_r * self.s["time_per_room_change_hours"]
                     + len(devices) * self.s["time_per_battery_hours"])

        self._work_cache[key] = (travel_h, float(service_h))
        return self._work_cache[key]

    # ---- penalties ---------------------------------------------------------
    def device_penalty(self, device: str, day: int | None) -> float:
        """Penalty for replacing device on day (None = never replaced)."""
        eol = self.eol_days[device]
        if day is None:                                     # A7
            if eol >= self.horizon:
                return 0.0
            return self.s["late_replacement_penalty_daily"] * (self.horizon - max(eol, 0.0))
        if day <= eol:
            return self.s["early_replacement_penalty_daily"] * (eol - day)
        # Downtime accrues only INSIDE the planning window: a device that died
        # before it opened was already down, and this window cannot be charged
        # for that history (A9).
        return self.s["late_replacement_penalty_daily"] * (day - max(eol, 0.0))

    def penalty_curve(self, device_eol_days: float) -> np.ndarray:
        """Penalty for every day 1..horizon (A8), given a failure day.

        The decision layer consumes exactly this shape with a PREDICTED
        failure day, so planning and scoring cannot drift apart.
        """
        days = np.arange(1, self.horizon + 1, dtype=float)
        early = self.s["early_replacement_penalty_daily"] * np.maximum(
            device_eol_days - days, 0.0)
        late = self.s["late_replacement_penalty_daily"] * np.where(
            days > device_eol_days, days - max(device_eol_days, 0.0), 0.0)
        return early + late

    def skip_penalty(self, device_eol_days: float) -> float:
        """Cost of never replacing a device with this failure day (A7)."""
        if device_eol_days >= self.horizon:
            return 0.0
        return self.s["late_replacement_penalty_daily"] * (
            self.horizon - max(device_eol_days, 0.0))

    # ---- full score --------------------------------------------------------
    def score(self, assignments: dict[str, int],
              all_devices: list[str] | None = None) -> CostBreakdown:
        """Score a plan mapping device -> day (1..horizon).

        all_devices is the population being scored; devices in it but absent
        from assignments are charged the never-replaced penalty (A7).
        """
        out = CostBreakdown()
        population = all_devices if all_devices is not None else list(assignments)

        for d in population:
            day = assignments.get(d)
            penalty = self.device_penalty(d, day)
            if day is None:
                out.late_penalty += penalty
                if penalty > 0:
                    out.devices_late += 1
            else:
                out.devices_replaced += 1
                if day > self.eol_days[d]:
                    out.late_penalty += penalty
                    out.devices_late += 1
                else:
                    out.early_penalty += penalty

        by_day: dict[int, list[str]] = defaultdict(list)
        for d, day in assignments.items():
            by_day[day].append(d)

        weekly: dict[int, float] = defaultdict(float)
        for day, devices in by_day.items():
            travel_h, service_h = self.day_hours(tuple(sorted(devices)))
            hours = travel_h + service_h
            overtime = max(hours - self.s["overtime_start"], 0.0)          # A4
            out.work_cost += (hours - overtime
                              + self.s["overtime_penalty_factor"] * overtime)
            out.raw_hours += hours
            out.travel_hours += travel_h
            out.service_hours += service_h
            out.overtime_hours += overtime
            if hours > self.s["worker_limit_daily_hours"]:                 # A5
                out.limit_penalty += self.s["worker_limit_daily_penalty"]
                out.days_over_cap += 1
            weekly[(day - 1) // 7] += hours

        for hours in weekly.values():
            if hours > self.s["worker_limit_weekly_hours"]:                # A5
                out.limit_penalty += self.s["worker_limit_weekly_penalty"]
                out.weeks_over_cap += 1

        out.days_used = len(by_day)
        return out


def eol_days_for_scenario(scenario: Scenario, eol: pd.DataFrame,
                          devices: pd.DataFrame) -> dict[str, float]:
    """True failure day of every device, relative to the scenario start.

    A6: censored devices are assumed to fail unobserved_eol_days after their
    last observation.
    """
    grace = float(scenario.settings["unobserved_eol_days"])
    start = scenario.start_time
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    last_seen = dict(zip(devices["device_id"], devices["end_time"]))
    out = {}
    for row in eol.itertuples():
        t = row.eol_time
        if pd.isna(t):
            t = last_seen[row.device_id] + pd.Timedelta(days=grace)
        out[row.device_id] = (t - start) / pd.Timedelta(days=1)
    return out
