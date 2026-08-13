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


def test_cost_model_constants_are_guarded():
    """Every unresolved constant in cost_model.yaml must raise on access."""
    cm = load_config("cost_model")
    with pytest.raises(UnknownValueError):
        _ = cm["penalties"]["late_replacement_per_device_hour"]


def test_decision_layer_blocked_without_official_constants(synth_ts):
    """decision.build_cost_curves must refuse to run on guessed penalties."""
    import pandas as pd

    from batteryswap.decision import build_cost_curves

    calibrated = pd.DataFrame({"battery_id": ["x"], "q05": [100.0], "q50": [500.0],
                               "q95": [900.0]})
    with pytest.raises(UnknownValueError):
        build_cost_curves(calibrated, [0.05, 0.50, 0.95])
