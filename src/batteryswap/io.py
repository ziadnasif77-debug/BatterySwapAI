"""Loading, schema validation, and dtype enforcement for the official files.

Schemas below are the REAL ones, verified field-by-field against the
2026-08-14 release (not the provisional guesses the brief sketched):

    battery_metrics.parquet  end_time, voltage, temperature, device_id
    devices.csv              device_id, building_id, room_id, start_time, end_time
    eol_times.csv            device_id, end_time            (blank => censored)
    scenarios.json           48 scenarios: settings, start_time, travel_costs

Note the naming the brief got wrong: the identifier is ``device_id`` (not
``battery_id``) and the timestamp column is ``end_time`` — the END of each
one-hour averaging window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

# Identifier columns. ⛔ Never features — enforced again in features.py and
# by tests/test_no_leakage.py.
ID_COLUMNS = ("device_id", "building_id", "room_id", "battery_id")

METRICS_SCHEMA = {
    "end_time": "datetime64[ns]",
    "device_id": "string",
    "voltage": "float64",
    "temperature": "float64",
}

DEVICES_SCHEMA = {
    "device_id": "string",
    "building_id": "string",
    "room_id": "string",
    "start_time": "datetime64[ns, UTC]",
    "end_time": "datetime64[ns, UTC]",
}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    """One planning problem: a start date, a depot, and a travel matrix."""
    name: str
    start_time: pd.Timestamp
    settings: dict
    travel: pd.DataFrame          # building x building, HOURS

    @property
    def depot(self) -> str:
        return self.settings["base_location"]

    @property
    def horizon_days(self) -> int:
        return int(self.settings["planning_window_days"])

    @property
    def end_time(self) -> pd.Timestamp:
        return self.start_time + pd.Timedelta(days=self.horizon_days)


@dataclass
class RawData:
    metrics: pd.DataFrame          # hourly series, all devices
    devices: pd.DataFrame          # device -> building/room, install + last-seen
    eol: pd.DataFrame              # device_id, eol_time (NaT => right-censored)
    scenarios: list[Scenario]
    issues: list[str] = field(default_factory=list)

    @property
    def location_of(self) -> dict[str, tuple[str, str]]:
        return {r.device_id: (r.building_id, r.room_id)
                for r in self.devices.itertuples()}


def _enforce(df: pd.DataFrame, schema: dict[str, str], name: str) -> pd.DataFrame:
    missing = set(schema) - set(df.columns)
    if missing:
        raise SchemaError(f"{name}: missing expected columns {sorted(missing)}; "
                          f"got {sorted(df.columns)}")
    out = df.copy()
    for col, dtype in schema.items():
        out[col] = out[col].astype(dtype)
    return out


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Parse scenarios.json into Scenario objects with square travel matrices."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = []
    for s in raw:
        tc = pd.DataFrame(s["travel_costs"])
        travel = tc.pivot(index="from", columns="to", values="hours")
        travel = travel.sort_index().sort_index(axis=1)
        if list(travel.index) != list(travel.columns):
            raise SchemaError(f"{s['name']}: travel matrix is not square")
        scenarios.append(Scenario(name=s["name"],
                                  start_time=pd.Timestamp(s["start_time"]),
                                  settings=dict(s["settings"]),
                                  travel=travel.astype(float)))
    return scenarios


def load_raw(raw_dir: str | Path | None = None) -> RawData:
    """Load and validate every official input file."""
    if raw_dir is None:
        raw_dir = Path(load_config("base")["paths"]["raw_dir"])
    raw_dir = Path(raw_dir)

    metrics = pd.read_parquet(raw_dir / "battery_metrics.parquet")
    metrics = metrics.reset_index(drop=True)
    metrics["device_id"] = metrics["device_id"].astype("string")
    metrics = _enforce(metrics, METRICS_SCHEMA, "battery_metrics.parquet")
    metrics = metrics.sort_values(["device_id", "end_time"], ignore_index=True)

    devices = pd.read_csv(raw_dir / "devices.csv", index_col=0)
    devices = _enforce(devices, DEVICES_SCHEMA, "devices.csv").reset_index(drop=True)

    eol = pd.read_csv(raw_dir / "eol_times.csv")
    eol = eol.rename(columns={"end_time": "eol_time"})
    eol["device_id"] = eol["device_id"].astype("string")
    eol["eol_time"] = pd.to_datetime(eol["eol_time"], utc=True)
    eol["censored"] = eol["eol_time"].isna()

    scenarios = load_scenarios(raw_dir / "scenarios.json")

    issues = validate(metrics, devices, eol, scenarios)
    return RawData(metrics=metrics, devices=devices, eol=eol,
                   scenarios=scenarios, issues=issues)


def validate(metrics: pd.DataFrame, devices: pd.DataFrame, eol: pd.DataFrame,
             scenarios: list[Scenario]) -> list[str]:
    """Cross-file structural checks; returns human-readable issues."""
    issues: list[str] = []

    m_dev = set(metrics["device_id"].unique())
    d_dev = set(devices["device_id"])
    e_dev = set(eol["device_id"])
    if m_dev != d_dev:
        issues.append(f"metrics/devices device sets differ "
                      f"(metrics-only {len(m_dev - d_dev)}, devices-only {len(d_dev - m_dev)})")
    if d_dev != e_dev:
        issues.append("devices/eol device sets differ")

    if metrics[["voltage", "temperature"]].isna().any().any():
        n = int(metrics[["voltage", "temperature"]].isna().any(axis=1).sum())
        issues.append(f"{n} metric rows have a null voltage or temperature")

    d_bld = set(devices["building_id"])
    for s in scenarios:
        s_bld = set(s.travel.index)
        if not d_bld <= s_bld:
            issues.append(f"{s.name}: travel matrix misses buildings {sorted(d_bld - s_bld)}")
        if s.depot not in s_bld:
            issues.append(f"{s.name}: depot {s.depot} absent from its travel matrix")
        v = s.travel.to_numpy(dtype=float)
        if np.isnan(v).any():
            issues.append(f"{s.name}: travel matrix contains NaN")
        if (v < 0).any():
            issues.append(f"{s.name}: travel matrix contains negative times")
    return issues


def validate_data(raw_dir: str | Path | None = None) -> RawData:
    """`make data` entry point: load, validate, and report."""
    data = load_raw(raw_dir)
    print(f"metrics : {len(data.metrics):,} rows, "
          f"{data.metrics['device_id'].nunique()} devices, "
          f"{data.metrics['end_time'].min()} .. {data.metrics['end_time'].max()}")
    print(f"devices : {len(data.devices)} devices, "
          f"{data.devices['building_id'].nunique()} buildings, "
          f"{data.devices['room_id'].nunique()} rooms")
    n_obs = int((~data.eol["censored"]).sum())
    print(f"eol     : {n_obs} observed / {len(data.eol)} "
          f"({n_obs / len(data.eol):.1%}), {int(data.eol['censored'].sum())} censored")
    s = data.scenarios[0]
    print(f"scenarios: {len(data.scenarios)}, window={s.horizon_days}d, "
          f"travel {s.travel.shape} in hours, "
          f"starts {data.scenarios[0].start_time.date()} .. "
          f"{data.scenarios[-1].start_time.date()}")
    for issue in data.issues:
        print(f"WARNING: {issue}")
    return data
