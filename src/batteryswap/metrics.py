"""Diagnostic metrics.

⛔ These are DIAGNOSTICS, never selection targets. Final cost decides which
model ships; MAE/RMSE/pinball/coverage exist to explain that cost. Every
reporting surface in this package must print cost first and these second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def q_col(q: float) -> str:
    """Canonical quantile column name (0.05 -> 'q05')."""
    return f"q{round(q * 100):02d}"


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(pred, float) - np.asarray(y, float)
    return float(np.sqrt(np.mean(d ** 2)))


def pinball_loss(pred_q: pd.DataFrame, y: np.ndarray, quantiles: list[float]) -> float:
    """Mean pinball loss across the quantile grid."""
    y = np.asarray(y, dtype=float)
    losses = []
    for q in quantiles:
        e = y - pred_q[q_col(q)].to_numpy(dtype=float)
        losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    return float(np.mean(losses))


def coverage(pred_q: pd.DataFrame, y: np.ndarray, q: float) -> float:
    """Empirical fraction of truths at or below the nominal level."""
    return float(np.mean(np.asarray(y, float) <= pred_q[q_col(q)].to_numpy(dtype=float)))
