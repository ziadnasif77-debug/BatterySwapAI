"""Loading, schema validation, and dtype enforcement for the official files.

⛔ The schemas below follow §2 of the project brief and are PROVISIONAL until
verified field-by-field against the released dataset documentation. Any
mismatch must be fixed here first — every downstream module trusts these
frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

# Identifier columns. ⛔ Never features — enforced again in features.py and
# by tests/test_no_leakage.py.
ID_COLUMNS = ("battery_id", "building_id", "room_id")

TIMESERIES_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "battery_id": "string",
    "voltage": "float64",
    "temperature": "float64",
}

LOCATIONS_SCHEMA = {
    "building_id": "string",
    "room_id": "string",
    "battery_id": "string",
}


class SchemaError(ValueError):
    pass


@dataclass
class RawData:
    train: pd.DataFrame
    evaluation: pd.DataFrame
    locations: pd.DataFrame
    travel_minutes: pd.DataFrame  # building x building, index == columns
    issues: list[str] = field(default_factory=list)


def _enforce(df: pd.DataFrame, schema: dict[str, str], name: str) -> pd.DataFrame:
    missing = set(schema) - set(df.columns)
    if missing:
        raise SchemaError(f"{name}: missing expected columns {sorted(missing)}. "
                          f"Verify io.py schemas against the official documentation.")
    extra = set(df.columns) - set(schema)
    out = df.copy()
    for col, dtype in schema.items():
        out[col] = out[col].astype(dtype)
    if extra:
        # Extra columns are kept but flagged — they may be documented fields
        # the provisional schema does not know about yet.
        out.attrs["extra_columns"] = sorted(extra)
    return out


def _find_one(raw_dir: Path, patterns: list[str], what: str) -> Path:
    for pattern in patterns:
        hits = sorted(raw_dir.glob(pattern))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(
                f"Ambiguous {what}: {[h.name for h in hits]} all match {pattern} in {raw_dir}. "
                f"Pin the exact filename in io.py once the official release is inspected.")
    raise FileNotFoundError(
        f"Could not locate the {what} in {raw_dir} (tried {patterns}). "
        f"Place the official competition files under data/raw/ first.")


def load_raw(raw_dir: str | Path | None = None) -> RawData:
    """Load and validate every official input file.

    Filename patterns are guesses to be pinned on first contact with the real
    release; the schema validation is the part that matters.
    """
    if raw_dir is None:
        base = load_config("base")
        raw_dir = Path(base["paths"]["raw_dir"])
    raw_dir = Path(raw_dir)

    train_path = _find_one(raw_dir, ["*train*.parquet"], "training time-series")
    eval_path = _find_one(raw_dir, ["*eval*.parquet", "*test*.parquet"], "evaluation time-series")
    loc_path = _find_one(raw_dir, ["*location*.parquet", "*location*.csv"], "locations table")
    travel_path = _find_one(raw_dir, ["*travel*.parquet", "*travel*.csv", "*distance*.csv"],
                            "travel-time matrix")

    def read_any(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

    train = _enforce(read_any(train_path), TIMESERIES_SCHEMA, "train")
    evaluation = _enforce(read_any(eval_path), TIMESERIES_SCHEMA, "evaluation")
    locations = _enforce(read_any(loc_path), LOCATIONS_SCHEMA, "locations")

    travel = read_any(travel_path)
    if travel.columns[0].lower() in ("building_id", "building", "id", "unnamed: 0"):
        travel = travel.set_index(travel.columns[0])
    travel.index = travel.index.astype(str)
    travel.columns = travel.columns.astype(str)
    issues = validate_travel_matrix(travel)

    return RawData(train=train, evaluation=evaluation, locations=locations,
                   travel_minutes=travel, issues=issues)


def validate_travel_matrix(travel: pd.DataFrame) -> list[str]:
    """Structural checks; returns human-readable issues (empty == clean)."""
    issues: list[str] = []
    if list(travel.index) != list(travel.columns):
        issues.append("travel matrix index != columns (not building x building?)")
    values = travel.to_numpy(dtype=float)
    if np.isnan(values).any():
        issues.append("travel matrix contains NaN")
    if (values < 0).any():
        issues.append("travel matrix contains negative times")
    if not np.allclose(np.diag(values), 0, atol=1e-9):
        issues.append("travel matrix diagonal is not zero")
    if not np.allclose(values, values.T, rtol=1e-6, atol=1e-6):
        issues.append("travel matrix is asymmetric — routing must treat it as directed")
    return issues


def validate_data(raw_dir: str | Path | None = None) -> RawData:
    """`make data` entry point: load, validate, and report."""
    data = load_raw(raw_dir)
    n_train = data.train["battery_id"].nunique()
    n_eval = data.evaluation["battery_id"].nunique()
    print(f"train: {len(data.train):,} rows, {n_train} batteries")
    print(f"evaluation: {len(data.evaluation):,} rows, {n_eval} batteries")
    print(f"locations: {len(data.locations):,} rows, "
          f"{data.locations['building_id'].nunique()} buildings")
    print(f"travel matrix: {data.travel_minutes.shape}")
    for issue in data.issues:
        print(f"WARNING: {issue}")
    return data
