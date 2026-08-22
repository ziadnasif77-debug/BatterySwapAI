"""Determinism — two runs from identical state produce identical outputs.

The full-submission determinism test activates when `make submit` is wired;
until then the deterministic building blocks are held to the same standard.
"""

import pandas as pd

from batteryswap.features import build_feature_table
from batteryswap.optimizer import Schedule
from batteryswap.optimizer.alns import ALNSConfig, alns_search, day_removal, random_removal
from batteryswap.sampling import sample_cutoffs


def test_feature_table_deterministic(synth_ts):
    cutoffs = pd.DataFrame({
        "battery_id": ["b000", "b001", "b002"],
        "cutoff": pd.to_datetime(["2024-08-01", "2024-09-01", "2024-10-01"]),
    })
    a = build_feature_table(synth_ts, cutoffs)
    b = build_feature_table(synth_ts, cutoffs)
    pd.testing.assert_frame_equal(a, b)


def test_cutoff_sampling_deterministic_under_seed(synth_ts):
    import numpy as np
    labels = pd.DataFrame({
        "battery_id": ["b000", "b001"],
        "eol_time": pd.to_datetime(["2025-06-01", "2025-07-01"]),
        "censored": [False, False],
        "last_observed": pd.to_datetime(["2025-06-01", "2025-07-01"]),
    })
    fracs = np.array([0.3, 0.5, 0.7, 0.9])
    a = sample_cutoffs(labels, synth_ts, fracs, n_cutoffs_per_battery=5, seed=7)
    b = sample_cutoffs(labels, synth_ts, fracs, n_cutoffs_per_battery=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_alns_deterministic_under_seed():
    batteries = [f"b{i}" for i in range(20)]
    initial = Schedule(assignments={b: (i % 10) + 1 for i, b in enumerate(batteries)})

    def objective(s: Schedule) -> float:
        # toy separable objective — enough to exercise the search machinery
        return sum((d - 5) ** 2 for d in s.assignments.values()) \
            + 100 * (len(batteries) - len(s.assignments))

    def repair(partial: Schedule, removed: list[str], rng) -> Schedule:
        out = Schedule(assignments=dict(partial.assignments))
        for b in removed:
            out.assignments[b] = min(range(1, 11), key=lambda d: (d - 5) ** 2)
        return out

    cfg = ALNSConfig(iterations=300, seed=42)
    r1 = alns_search(initial, objective, [random_removal, day_removal], [repair], cfg)
    r2 = alns_search(initial, objective, [random_removal, day_removal], [repair], cfg)
    assert r1[1] == r2[1]
    assert r1[0].assignments == r2[0].assignments
    assert r1[1] <= objective(initial)
