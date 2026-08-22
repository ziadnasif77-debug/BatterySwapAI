"""B3 — LightGBM quantile regression, the primary model.

Hard requirements implemented here:
- ⛔ Non-crossing quantiles: enforced by isotonic sorting across the quantile
  axis after prediction (tests/test_quantile_ordering.py).
- Monotonic constraints: LightGBM REJECTS ``monotone_constraints``
  combined with the built-in quantile objective (its per-leaf renormalization
  is incompatible) — discovered in the synthetic dry run, raises
  LightGBMError. The constraint table below is therefore applied to point-
  objective models (B2) only; for B3 the candidate mechanism is a smoothed-
  pinball custom objective WITH constraints, gated on a measured final-cost
  win once real data exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # allows the rest of the package to import without lightgbm
    lgb = None

# Feature-name -> monotone direction of RUL w.r.t. that feature.
# +1: RUL non-decreasing in the feature; -1: non-increasing; 0: unconstrained.
_MONOTONE_RULES: list[tuple[str, int]] = [
    ("v_now", +1),               # lower voltage -> less life
    ("v_resid_now", +1),
    ("v_above_threshold", +1),
    ("age_hours", -1),           # older -> less life
    ("v_drop_total", -1),
    ("dist_below_plateau", -1),
    ("cum_charge_proxy", -1),
]


def monotone_constraints(feature_names: list[str]) -> list[int]:
    out = []
    for name in feature_names:
        direction = 0
        for prefix, d in _MONOTONE_RULES:
            if name == prefix:
                direction = d
                break
        out.append(direction)
    return out


def enforce_non_crossing(q_matrix: np.ndarray) -> np.ndarray:
    """Sort each row across the quantile axis — the isotonic-regression
    solution for monotone rearrangement of crossing quantiles."""
    return np.sort(np.asarray(q_matrix, dtype=float), axis=1)


class GBMQuantileModel:
    """One LightGBM booster per quantile level, shared feature contract."""

    def __init__(self, quantiles: list[float], params: dict, seed: int):
        if lgb is None:
            raise ImportError("lightgbm is required for GBMQuantileModel")
        self.quantiles = list(quantiles)
        self.params = dict(params)
        self.seed = seed
        self.boosters_: dict[float, lgb.Booster] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> GBMQuantileModel:
        self.feature_names_ = list(X.columns)
        early_stopping = (int(self.params.pop("early_stopping_rounds", 0))
                          if X_val is not None else 0)
        n_estimators = int(self.params.pop("n_estimators", 1000))

        for q in self.quantiles:
            # NOTE: no monotone_constraints here — LightGBM forbids them with
            # objective="quantile" (see module docstring).
            params = {
                "objective": "quantile",
                "alpha": q,
                "verbosity": -1,
                "seed": self.seed,
                "deterministic": True,
                "force_row_wise": True,
                **self.params,
            }
            train_set = lgb.Dataset(X, label=y)
            valid_sets = [lgb.Dataset(X_val, label=y_val)] if X_val is not None else []
            callbacks = ([lgb.early_stopping(early_stopping, verbose=False)]
                         if early_stopping and valid_sets else [])
            self.boosters_[q] = lgb.train(params, train_set, num_boost_round=n_estimators,
                                          valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if list(X.columns) != self.feature_names_:
            raise ValueError("feature columns/order differ from training")
        raw = np.column_stack([self.boosters_[q].predict(X) for q in self.quantiles])
        raw = np.maximum(raw, 0.0)          # RUL cannot be negative
        ordered = enforce_non_crossing(raw)
        cols = [f"q{round(q * 100):02d}" for q in self.quantiles]
        return pd.DataFrame(ordered, columns=cols, index=X.index)
