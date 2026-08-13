"""Synthetic fleet generator — plumbing rehearsal data, NEVER competition data.

Shapes match the documented schema (§2 of the brief): hourly averages of
voltage/temperature, plateau -> knee -> collapse discharge, seasonal
temperature coupling, per-device lifetime spread, install-date clustering
within buildings, and a building x building travel matrix in minutes.

Every constant in here is an arbitrary SYNTHETIC choice used only to exercise
code paths before the official release. Nothing from this module may feed the
real pipeline: the no-invention rule (§0) still holds because these values
never touch configs/cost_model.yaml or any submission artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def synthetic_battery(battery_id: str, seed: int, n_days: int = 600,
                      life_days: int = 700,
                      start: str | pd.Timestamp = "2024-01-01") -> pd.DataFrame:
    """Hourly series: flat plateau near 3.0 V, knee at ~85% of life, seasonal
    temperature coupling, gaussian noise."""
    rng = np.random.default_rng(seed)
    hours = n_days * 24
    t = pd.date_range(start, periods=hours, freq="h")
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


@dataclass
class SyntheticFleet:
    """Full series (to end-of-life) per battery, plus locations and travel."""
    ts: pd.DataFrame              # full series for every battery
    locations: pd.DataFrame       # building_id, room_id, battery_id
    travel: pd.DataFrame          # building x building, minutes
    life_days: dict[str, int]     # ground-truth synthetic lifetime per battery


def synthetic_fleet(n_buildings: int = 8, seed: int = 0,
                    rooms_per_building: tuple[int, int] = (1, 3),
                    batteries_per_room: tuple[int, int] = (2, 4),
                    life_days_range: tuple[int, int] = (550, 750),
                    past_eol_days: int = 30) -> SyntheticFleet:
    """Fleet with install-date clustering per building and lifetime spread.

    Each battery's series runs `past_eol_days` beyond its synthetic lifetime so
    a threshold-based EOL is always observable (training-set semantics).
    """
    rng = np.random.default_rng(seed)
    frames, loc_rows, life = [], [], {}
    idx = 0
    for bi in range(n_buildings):
        building = f"bl{bi:03d}"
        # installations cluster within a building (§2): shared base + small jitter
        base_install = pd.Timestamp("2023-06-01") + pd.Timedelta(days=int(rng.integers(0, 120)))
        n_rooms = int(rng.integers(rooms_per_building[0], rooms_per_building[1] + 1))
        for ri in range(n_rooms):
            room = f"r{ri}"
            n_bat = int(rng.integers(batteries_per_room[0], batteries_per_room[1] + 1))
            for _ in range(n_bat):
                b = f"b{idx:04d}"
                idx += 1
                life_d = int(rng.integers(*life_days_range))
                install = base_install + pd.Timedelta(days=int(rng.integers(0, 14)))
                frames.append(synthetic_battery(
                    b, seed=1000 + idx, n_days=life_d + past_eol_days,
                    life_days=life_d, start=install))
                loc_rows.append({"building_id": building, "room_id": room, "battery_id": b})
                life[b] = life_d

    buildings = [f"bl{bi:03d}" for bi in range(n_buildings)]
    m = rng.integers(10, 90, size=(n_buildings, n_buildings)).astype(float)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    travel = pd.DataFrame(m, index=buildings, columns=buildings)

    return SyntheticFleet(ts=pd.concat(frames, ignore_index=True),
                          locations=pd.DataFrame(loc_rows),
                          travel=travel,
                          life_days=life)
