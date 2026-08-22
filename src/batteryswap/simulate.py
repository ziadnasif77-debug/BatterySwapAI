"""Phase 8 — offline end-to-end harness.

Takes held-out BUILDINGS (never random rows), applies realistic truncations
drawn from the estimated evaluation distribution, runs the full pipeline
(features -> model -> calibration -> decision -> optimizer), and scores the
resulting schedule against known outcomes with the official evaluator.

Primary output column: FINAL COST. MAE/RMSE/pinball/coverage are adjacent
diagnostic columns whose job is to explain the cost, not to compete with it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HarnessResult:
    config_name: str
    final_cost: float
    mae_hours: float
    pinball_loss: float
    coverage_q10: float
    kpis: dict


def run_harness(holdout_buildings: list[str], config_overrides: dict | None = None
                ) -> HarnessResult:
    """Full-pipeline simulation on held-out buildings.

    Wiring is completed in Phase 8 — it needs the official evaluator (Phase 0)
    and real data (Phase 1) first. The signature and the result contract are
    fixed now so the tuning matrix

        {models} x {calibration on/off} x {q values} x {optimizer configs}

    can be scripted against it unchanged.
    """
    raise NotImplementedError(
        "run_harness is wired in Phase 8. Prerequisites: official evaluator wrapped "
        "(Phase 0), data audited (Phase 1), labels+sampling (Phase 2), features "
        "(Phase 3), models (Phase 4), calibration (Phase 5), decision layer "
        "(Phase 6), optimizer (Phase 7)."
    )
