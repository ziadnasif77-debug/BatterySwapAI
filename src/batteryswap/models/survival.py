"""B4 — survival models: Weibull AFT (lifelines). Handles right-censoring
natively and outputs P(fail < t) directly; ensemble partner for B3."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from lifelines import WeibullAFTFitter
except ImportError:
    WeibullAFTFitter = None


class WeibullAFTModel:
    """Accelerated-failure-time Weibull on the engineered feature matrix.

    fit() expects X plus duration (observed life or observation span) and an
    event flag (1 = EOL observed, 0 = right-censored) — censored batteries are
    first-class citizens here, not dropped.
    """

    def __init__(self, penalizer: float = 0.01):
        if WeibullAFTFitter is None:
            raise ImportError("lifelines is required for WeibullAFTModel")
        self.fitter = WeibullAFTFitter(penalizer=penalizer)

    def fit(self, X: pd.DataFrame, duration_hours: pd.Series,
            event_observed: pd.Series) -> "WeibullAFTModel":
        frame = X.copy()
        frame["duration"] = duration_hours.to_numpy()
        frame["event"] = event_observed.astype(int).to_numpy()
        self.fitter.fit(frame, duration_col="duration", event_col="event")
        return self

    def predict_quantiles(self, X: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
        """Quantiles of the RESIDUAL life distribution are approximated by the
        unconditional AFT quantiles; conditioning on survival-so-far is a
        planned refinement once real data exists to validate it."""
        out = {}
        for q in quantiles:
            preds = self.fitter.predict_percentile(X, p=1 - q)  # lifelines: survival percentile
            out[f"q{int(round(q * 100)):02d}"] = np.asarray(preds, dtype=float)
        return pd.DataFrame(out, index=X.index)
