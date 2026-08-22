"""Cutoff generation for training-example construction.

⛔ The cutoff distribution is a first-class design decision: it must MATCH the
evaluation truncation distribution, estimated from the evaluation series
lengths — never uniform in absolute time by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def observed_span_hours(ts: pd.DataFrame) -> pd.Series:
    """Observed span per battery in hours (max - min timestamp)."""
    g = ts.groupby("battery_id")["timestamp"]
    return (g.max() - g.min()) / pd.Timedelta(hours=1)


def estimate_truncation_fractions(eval_ts: pd.DataFrame,
                                  expected_life_hours: float) -> pd.Series:
    """Rough per-battery truncation fraction: observed span / expected life.

    `expected_life_hours` should come from the TRAINING lifetime distribution
    (e.g. its median), documented in reports/data_audit.md. This is an
    estimate — the true installed age at truncation is hidden.
    """
    spans = observed_span_hours(eval_ts)
    return (spans / expected_life_hours).clip(upper=1.0).rename("truncation_fraction")


def sample_cutoffs(train_labels: pd.DataFrame,
                   train_ts: pd.DataFrame,
                   truncation_fractions: np.ndarray,
                   n_cutoffs_per_battery: int,
                   seed: int) -> pd.DataFrame:
    """Draw cutoff timestamps per training battery matching the evaluation
    truncation distribution (sampled with replacement from the estimated
    fractions). Returns columns: battery_id, cutoff, rul_hours (NaN if the
    cutoff falls in a censored region).
    """
    rng = np.random.default_rng(seed)
    first_seen = train_ts.groupby("battery_id")["timestamp"].min()

    rows = []
    for _, lab in train_labels.iterrows():
        b = lab["battery_id"]
        start = first_seen[b]
        end = lab["last_observed"] if lab["censored"] else lab["eol_time"]
        life_hours = (end - start) / pd.Timedelta(hours=1)
        if life_hours <= 0:
            continue
        fracs = rng.choice(truncation_fractions, size=n_cutoffs_per_battery)
        for f in fracs:
            cutoff = start + pd.Timedelta(hours=float(f) * life_hours)
            rul = None if lab["censored"] else (lab["eol_time"] - cutoff) / pd.Timedelta(hours=1)
            rows.append({"battery_id": b, "cutoff": cutoff,
                         "rul_hours": rul, "censored": lab["censored"]})
    return pd.DataFrame(rows)
