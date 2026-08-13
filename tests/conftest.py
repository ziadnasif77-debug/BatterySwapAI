"""Shared fixtures: synthetic battery time-series shaped like the documented
schema (plateau -> knee -> collapse, temperature-coupled, noisy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_battery(battery_id: str, seed: int, n_days: int = 600,
                      life_days: int = 700) -> pd.DataFrame:
    """Hourly series: flat plateau near 3.0 V, knee at ~85% of life, seasonal
    temperature coupling, gaussian noise."""
    rng = np.random.default_rng(seed)
    hours = n_days * 24
    t = pd.date_range("2024-01-01", periods=hours, freq="h")
    age_frac = np.arange(hours) / (life_days * 24)

    plateau = 3.0 - 0.1 * age_frac
    knee_frac = 0.85
    collapse = np.where(age_frac > knee_frac,
                        -2.0 * (age_frac - knee_frac) ** 2 * 10, 0.0)
    day_of_year = t.dayofyear.to_numpy()
    temperature = 20 + 8 * np.sin(2 * np.pi * (day_of_year - 100) / 365) \
        + rng.normal(0, 1, hours)
    v = plateau + collapse + 0.004 * (temperature - 20) + rng.normal(0, 0.01, hours)

    return pd.DataFrame({
        "timestamp": t,
        "battery_id": battery_id,
        "voltage": v,
        "temperature": temperature,
    })


@pytest.fixture(scope="session")
def synth_ts() -> pd.DataFrame:
    frames = [synthetic_battery(f"b{i:03d}", seed=i) for i in range(4)]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synth_locations() -> pd.DataFrame:
    return pd.DataFrame({
        "building_id": ["bl01", "bl01", "bl02", "bl02"],
        "room_id": ["r1", "r2", "r1", "r1"],
        "battery_id": [f"b{i:03d}" for i in range(4)],
    })
