"""Model ladder behaviour (§9) — especially the monotone-constraint contract.

B2 is where ⛔ monotonic constraints actually apply: LightGBM accepts them
with the L2 objective and REJECTS them with the built-in quantile objective.
Both halves of that statement are asserted here, so a LightGBM upgrade that
changes either one fails loudly instead of silently dropping the constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batteryswap.models.gbm_point import GBMPointModel
from batteryswap.models.gbm_quantile import (
    GBMQuantileModel,
    enforce_non_crossing,
    monotone_constraints,
)

PARAMS = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 60,
          "min_child_samples": 5, "verbosity": -1}


def _training_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """RUL falls as age rises and as voltage drops — the physical directions
    the constraints encode."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 15000, n)
    v_now = 3.0 - age / 15000 * 0.4 + rng.normal(0, 0.01, n)
    X = pd.DataFrame({"age_hours": age, "v_now": v_now,
                      "noise": rng.normal(0, 1, n)})
    y = pd.Series(np.maximum(16000 - age + rng.normal(0, 300, n), 0.0))
    return X, y


def test_monotone_constraint_table_maps_the_documented_directions():
    names = ["age_hours", "v_now", "noise", "cum_charge_proxy"]
    assert monotone_constraints(names) == [-1, +1, 0, -1]


def test_b2_fits_with_monotone_constraints_and_respects_them():
    X, y = _training_frame()
    model = GBMPointModel(dict(PARAMS), seed=0, monotone=True).fit(X, y)

    # sweep age upward with everything else fixed: prediction must not rise
    probe = pd.DataFrame({"age_hours": np.linspace(0, 15000, 40),
                          "v_now": np.full(40, 2.8),
                          "noise": np.zeros(40)})
    preds = model.predict(probe)
    assert (np.diff(preds) <= 1e-9).all(), "RUL rose with age despite the constraint"
    assert (preds >= 0).all()

    # sweep voltage upward: prediction must not fall
    probe2 = pd.DataFrame({"age_hours": np.full(40, 7000.0),
                           "v_now": np.linspace(2.6, 3.0, 40),
                           "noise": np.zeros(40)})
    assert (np.diff(model.predict(probe2)) >= -1e-9).all()


def test_lightgbm_still_rejects_constraints_with_the_quantile_objective():
    """The constraint that forced B3's post-hoc design. If a LightGBM release
    ever lifts this, revisit gbm_quantile.py — that is why this test exists."""
    import lightgbm as lgb

    X, y = _training_frame(n=120)
    with pytest.raises(lgb.basic.LightGBMError, match="monotone"):
        lgb.train({"objective": "quantile", "alpha": 0.5, "verbosity": -1,
                   "monotone_constraints": [-1, 1, 0]},
                  lgb.Dataset(X, label=y), num_boost_round=5)


def test_b3_predictions_never_cross():
    X, y = _training_frame()
    quantiles = [0.05, 0.5, 0.95]
    pred = GBMQuantileModel(quantiles, dict(PARAMS), seed=0).fit(X, y).predict_quantiles(X)
    values = pred[["q05", "q50", "q95"]].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (values >= 0).all()


def test_enforce_non_crossing_is_idempotent():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 100, size=(20, 5))
    once = enforce_non_crossing(raw)
    assert np.array_equal(once, enforce_non_crossing(once))
