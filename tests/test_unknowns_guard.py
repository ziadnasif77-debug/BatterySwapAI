"""The no-invention rule is only as strong as its enforcement point."""

import pytest

from batteryswap.config import GuardedConfig, UnknownValueError, load_config


def test_unknown_leaf_raises():
    cfg = GuardedConfig({"a": {"b": "UNKNOWN"}})
    with pytest.raises(UnknownValueError, match=r"a\.b"):
        _ = cfg["a"]["b"]


def test_known_leaf_passes():
    cfg = GuardedConfig({"a": {"b": 3}})
    assert cfg["a"]["b"] == 3


def test_still_unresolved_constants_are_guarded():
    """Constants the official release did NOT settle must still raise. These
    are the real remaining blockers, so the guard has to keep holding."""
    cm = load_config("cost_model")
    for section, key in (("failure", "eol_definition"),
                         ("failure", "voltage_threshold"),
                         ("submission", "columns"),
                         ("horizon", "day_indexing_base")):
        with pytest.raises(UnknownValueError):
            _ = cm[section][key]


def test_transcribed_penalties_now_load_with_the_official_values():
    """The mirror of the guard: values read from scenarios.json must be live,
    in the official per-DAY units, with late 20x early."""
    cm = load_config("cost_model")
    early = cm["penalties"]["early_replacement_per_day"]
    late = cm["penalties"]["late_replacement_per_day"]
    assert (early, late) == (0.5, 10.0)
    assert cm["horizon"]["planning_days"] == 42
    assert late / early == 20.0


def test_decision_layer_runs_on_the_official_constants():
    """build_cost_curves was blocked pre-release; with the constants resolved
    it must now produce a curve over the official 42-day window."""
    import pandas as pd

    from batteryswap.decision import build_cost_curves

    calibrated = pd.DataFrame({"battery_id": ["x"], "q05": [100.0], "q50": [500.0],
                               "q95": [900.0]})
    curves = build_cost_curves(calibrated, [0.05, 0.50, 0.95])
    assert len(curves) == 42
    assert curves["day"].min() == 1 and curves["day"].max() == 42
    assert (curves["expected_penalty"] >= 0).all()
