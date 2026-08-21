"""§16.1 — fails if any identifier column can reach a model's feature matrix."""

import pandas as pd
import pytest

from batteryswap.features import assemble_matrix, build_feature_table
from batteryswap.io import ID_COLUMNS


def test_assemble_matrix_strips_all_identifiers(synth_ts):
    cutoffs = pd.DataFrame({
        "battery_id": ["b000", "b001"],
        "cutoff": [pd.Timestamp("2024-10-01"), pd.Timestamp("2024-11-01")],
        "rul_hours": [500.0, 300.0],
    })
    table = build_feature_table(synth_ts, cutoffs)
    X, y, meta = assemble_matrix(table)

    for col in ID_COLUMNS:
        assert col not in X.columns, f"identifier {col} leaked into feature matrix"
    assert "cutoff" not in X.columns
    assert "battery_id" in meta.columns  # kept as metadata for grouping only
    assert y is not None and len(y) == len(X)


def test_assemble_matrix_rejects_smuggled_identifier(synth_ts):
    """A feature table that carries an *_id column under a new name must fail
    loudly, not silently train on it."""
    table = pd.DataFrame({"v_now": [3.0], "sneaky_building_id": ["bl01"]})
    with pytest.raises(ValueError, match=r"[Ii]dentifier"):
        assemble_matrix(table)


def test_assemble_matrix_rejects_non_numeric():
    table = pd.DataFrame({"v_now": [3.0], "note": ["plateau"]})
    with pytest.raises(ValueError, match="Non-numeric"):
        assemble_matrix(table)
