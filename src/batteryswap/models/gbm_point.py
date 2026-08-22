"""B2 — LightGBM point regression, the serious baseline.

This is the level where monotonic constraints actually apply:
LightGBM accepts ``monotone_constraints`` with the L2/L1 objectives but
rejects them with the built-in quantile objective (see gbm_quantile.py). B2
therefore ships the constrained model, and its cost gap to B3 is the evidence
for what the constraints are worth.

B2 produces a POINT estimate. ⛔ It must never be fed straight to the
scheduler — it exists as the control for B3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gbm_quantile import monotone_constraints

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class GBMPointModel:
    """Single booster, L2 objective, monotone constraints enabled."""

    def __init__(self, params: dict, seed: int, monotone: bool = True):
        if lgb is None:
            raise ImportError("lightgbm is required for GBMPointModel")
        self.params = dict(params)
        self.seed = seed
        self.monotone = monotone
        self.booster_: lgb.Booster | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> GBMPointModel:
        self.feature_names_ = list(X.columns)
        early_stopping = (int(self.params.pop("early_stopping_rounds", 0))
                          if X_val is not None else 0)
        n_estimators = int(self.params.pop("n_estimators", 1000))

        params = {
            "objective": "regression",
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            **self.params,
        }
        if self.monotone:
            params["monotone_constraints"] = monotone_constraints(self.feature_names_)

        valid_sets = [lgb.Dataset(X_val, label=y_val)] if X_val is not None else []
        callbacks = ([lgb.early_stopping(early_stopping, verbose=False)]
                     if early_stopping and valid_sets else [])
        self.booster_ = lgb.train(params, lgb.Dataset(X, label=y),
                                  num_boost_round=n_estimators,
                                  valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("fit first")
        if list(X.columns) != self.feature_names_:
            raise ValueError("feature columns/order differ from training")
        return np.maximum(self.booster_.predict(X), 0.0)  # RUL cannot be negative
