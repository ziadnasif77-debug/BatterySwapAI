"""Shared fixtures: synthetic battery time-series shaped like the documented
schema (plateau -> knee -> collapse, temperature-coupled, noisy)."""

from __future__ import annotations

import pandas as pd
import pytest

from batteryswap.synthetic import synthetic_battery  # noqa: F401 — shared generator


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
