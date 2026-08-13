"""Integration: the whole pipeline holds together on synthetic data.

Small fleet + short lives + few ALNS iterations to keep runtime test-friendly;
the point is wiring, not quality.
"""

from __future__ import annotations

import numpy as np

from batteryswap.dryrun import run_dry_run


def test_dry_run_end_to_end():
    res = run_dry_run(seed=0, n_buildings=4, n_cutoffs_per_battery=3,
                      alns_iterations=250, life_days_range=(220, 280))

    assert np.isfinite(res.final_cost)
    # ALNS starts from the construction — it can only hold or improve it
    assert res.final_cost <= res.construct_cost + 1e-6

    assert res.kpis["all_assigned"]
    days = list(res.schedule.assignments.values())
    assert all(1 <= d <= res.kpis["horizon_days"] for d in days)

    # coverage table exists for every nominal level and is a valid fraction
    assert len(res.coverage) == 5
    assert res.coverage["empirical"].between(0.0, 1.0).all()
