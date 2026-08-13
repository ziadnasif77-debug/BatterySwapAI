"""Grouped cross-validation (§7).

⛔ GroupKFold on ``building_id``, always. Never a random row split, never a
random battery split within a building — evaluation contains unseen buildings,
so any split that lets one building appear on both sides reports a number the
hidden set will not reproduce.

⛔ Per-fold results are reported, not just the mean: variance across buildings
is the quantity that predicts hidden-set behaviour (§17).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import coverage, mae, pinball_loss, q_col, rmse
from .models.gbm_quantile import GBMQuantileModel


def grouped_folds(groups: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, val_idx) positional index pairs, grouped by building."""
    n_groups = pd.Series(groups).nunique()
    if n_groups < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds the {n_groups} distinct groups "
                         f"available — reduce n_splits or pool more buildings")
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=np.asarray(groups)))


def cross_validate_quantiles(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                             quantiles: list[float], params: dict, seed: int,
                             n_splits: int = 5) -> pd.DataFrame:
    """Fit B3 per fold; return one diagnostics row per fold.

    No calibration here on purpose: this measures the RAW model. Calibration is
    evaluated separately (§10) because it needs its own held-out buildings.
    """
    rows = []
    for fold, (tr, va) in enumerate(grouped_folds(groups, n_splits)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]
        model = GBMQuantileModel(quantiles, dict(params), seed=seed).fit(X_tr, y_tr)
        pred = model.predict_quantiles(X_va)
        y_arr = y_va.to_numpy(dtype=float)

        row = {
            "fold": fold,
            "n_train": len(tr),
            "n_val": len(va),
            "n_val_buildings": pd.Series(groups).iloc[va].nunique(),
            "mae_hours": mae(pred[q_col(0.50)].to_numpy(), y_arr),
            "rmse_hours": rmse(pred[q_col(0.50)].to_numpy(), y_arr),
            "pinball": pinball_loss(pred, y_arr, quantiles),
        }
        for q in quantiles:
            row[f"cov_{q_col(q)}"] = coverage(pred, y_arr, q)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across folds. ⛔ Report both — a good mean with a large std
    across buildings is a warning, not a result."""
    numeric = folds.drop(columns=["fold"]).select_dtypes("number")
    return pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=0)})
