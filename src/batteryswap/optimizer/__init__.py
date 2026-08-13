"""Multi-period inventory-routing optimizer (§12).

Shared data model:

- ``Assignment``: battery -> replacement day (the decision variable).
- ``Schedule``: assignments + per-day worker routes, the object the official
  evaluator scores.

⛔ The objective inside every search component is the official evaluator (or
the tolerance-verified surrogate from Phase 0) — never an invented proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Visit:
    building_id: str
    room_id: str
    battery_id: str


@dataclass
class DayRoute:
    day: int
    worker_id: int
    visits: list[Visit] = field(default_factory=list)


@dataclass
class Schedule:
    """assignments: battery_id -> day. routes: realized sequencing."""
    assignments: dict[str, int] = field(default_factory=dict)
    routes: list[DayRoute] = field(default_factory=list)

    def batteries_on_day(self, day: int) -> list[str]:
        return [b for b, d in self.assignments.items() if d == day]
