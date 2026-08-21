"""B0 and B1 baselines. These exist to be beaten by a DOCUMENTED margin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import _theil_sen_slope  # shared robust slope
from ..labels import rolling_voltage


class ConstantBaseline:
    """B0: predicted RUL = mean training lifetime - current age (floor at 0)."""

    def __init__(self):
        self.mean_life_hours_: float | None = None

    def fit(self, lifetimes_hours: pd.Series) -> ConstantBaseline:
        self.mean_life_hours_ = float(lifetimes_hours.dropna().mean())
        return self

    def predict(self, age_hours: np.ndarray) -> np.ndarray:
        if self.mean_life_hours_ is None:
            raise RuntimeError("fit first")
        return np.maximum(self.mean_life_hours_ - np.asarray(age_hours, dtype=float), 0.0)


class LinearExtrapolationBaseline:
    """B1: extrapolate the robust rolling voltage to the failure threshold
    with a Theil-Sen slope over the trailing window. The physical floor."""

    def __init__(self, threshold: float, slope_window_days: int = 90):
        self.threshold = threshold
        self.slope_window_days = slope_window_days

    def predict_one(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> float:
        h = history[history["timestamp"] <= cutoff]
        rv = rolling_voltage(h).dropna()
        if rv.empty:
            return np.nan
        window = rv[rv.index > cutoff - pd.Timedelta(days=self.slope_window_days)]
        if len(window) < 8:
            window = rv
        t_hours = ((window.index - window.index[0]) / pd.Timedelta(hours=1)).to_numpy()
        slope = _theil_sen_slope(t_hours, window.to_numpy())
        v_now = float(window.iloc[-1])
        if not np.isfinite(slope) or slope >= 0:
            return np.inf  # flat or rising on the plateau: no finite crossing
        return max((self.threshold - v_now) / slope, 0.0)
