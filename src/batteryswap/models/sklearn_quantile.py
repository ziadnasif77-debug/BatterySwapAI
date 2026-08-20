"""B3 for the competition runtime: scikit-learn quantile gradient boosting.

⛔ LightGBM is NOT installed on the submission machine. The competition pins
its runtime in the official example repo's ``requirements.txt`` with the note
*"changing these will not affect the competition runtime environment"*, and
lightgbm/xgboost/catboost are absent. A pickled planner carrying LightGBM
boosters unpickles to an ImportError there, so ``gbm_quantile.py`` cannot ship
- only this module can.

The forced move turns out to be an upgrade. ``HistGradientBoostingRegressor``
accepts ``monotonic_cst`` together with ``loss="quantile"``, so the monotone
constraints of section 9 - which LightGBM refuses on its quantile objective -
are finally active on the primary model.

Non-crossing is still enforced after prediction: monotone constraints act on
features, not across quantile levels, so independently fitted quantiles can
still cross.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .gbm_quantile import enforce_non_crossing, monotone_constraints

# Only these carry a defensible physical direction; everything else is left
# unconstrained rather than guessed.
DEFAULT_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,      # many correlated cutoffs per device
    "l2_regularization": 1.0,
    "early_stopping": False,     # determinism: no internal random split
}


class SklearnQuantileModel:
    """One booster per quantile level, sharing a feature contract.

    Deliberately mirrors GBMQuantileModel's interface (``fit``,
    ``predict_quantiles``, ``feature_names_``) so the planner, the calibration
    layer and the pipeline are indifferent to which one is in use.
    """

    def __init__(self, quantiles: list[float], params: dict | None = None,
                 seed: int = 0, monotone: bool = True):
        self.quantiles = list(quantiles)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.monotone = monotone
        self.models_: dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names_: list[str] = []
        self.dropped_columns_: list[str] = []

    @staticmethod
    def degenerate_columns(X: pd.DataFrame) -> list[str]:
        """Columns that are all-NaN or constant.

        sklearn's binning raises "window shape cannot be larger than input
        array shape" on these rather than ignoring them, so they are removed
        explicitly - and reported, because a dead feature is usually a bug in
        the feature code, not something to paper over.
        """
        return [c for c in X.columns if X[c].nunique(dropna=True) <= 1]

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "SklearnQuantileModel":
        dead = self.degenerate_columns(X)
        if dead:
            X = X.drop(columns=dead)
        self.dropped_columns_ = dead
        self.feature_names_ = list(X.columns)
        constraints = (monotone_constraints(self.feature_names_)
                       if self.monotone else None)

        values = X.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, random_state=self.seed,
                monotonic_cst=constraints, **self.params)
            model.fit(values, target)
            self.models_[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names_] if set(self.feature_names_) <= set(X.columns) else X
        if list(X.columns) != self.feature_names_:
            raise ValueError("feature columns/order differ from training")
        values = X.to_numpy(dtype=float)
        raw = np.column_stack([self.models_[q].predict(values) for q in self.quantiles])
        raw = np.maximum(raw, 0.0)                 # RUL cannot be negative
        ordered = enforce_non_crossing(raw)
        cols = [f"q{int(round(q * 100)):02d}" for q in self.quantiles]
        return pd.DataFrame(ordered, columns=cols, index=X.index)
