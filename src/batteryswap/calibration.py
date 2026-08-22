"""Phase 5 — Conformalized Quantile Regression grouped by building.

Raw quantiles are not calibrated on unseen buildings; every downstream
decision consumes CALIBRATED quantiles only. Coverage diagnostics (marginal
and conditional) are mandatory report artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cqr_offsets(pred_q: pd.DataFrame, y_true: pd.Series,
                quantiles: list[float]) -> dict[float, float]:
    """Per-level conformal offsets from a held-out calibration set.

    One-sided CQR per level: for level q, the nonconformity score is
    (pred - y) so that shifting by the empirical (1 - q)-quantile of scores
    restores coverage. Symmetric-interval CQR is a later variant to compare.
    """
    offsets = {}
    n = len(y_true)
    for q in quantiles:
        col = f"q{round(q * 100):02d}"
        scores = pred_q[col].to_numpy() - y_true.to_numpy()
        rank = min(int(np.ceil((n + 1) * (1 - q))), n)
        offsets[q] = float(np.sort(scores)[rank - 1])
    return offsets


def apply_offsets(pred_q: pd.DataFrame, offsets: dict[float, float]) -> pd.DataFrame:
    out = pred_q.copy()
    for q, off in offsets.items():
        col = f"q{round(q * 100):02d}"
        out[col] = np.maximum(out[col] - off, 0.0)
    # calibration can re-introduce crossings — re-sort
    cols = sorted([c for c in out.columns if c.startswith("q")], key=lambda c: int(c[1:]))
    out[cols] = np.sort(out[cols].to_numpy(), axis=1)
    return out


def coverage_table(pred_q: pd.DataFrame, y_true: pd.Series,
                   quantiles: list[float]) -> pd.DataFrame:
    """Empirical fraction of true values below each nominal level.
    Perfect calibration => empirical == nominal."""
    rows = []
    y = y_true.to_numpy()
    for q in quantiles:
        col = f"q{round(q * 100):02d}"
        rows.append({"nominal": q,
                     "empirical": float(np.mean(y <= pred_q[col].to_numpy())),
                     "n": len(y)})
    return pd.DataFrame(rows)


def conditional_coverage(pred_q: pd.DataFrame, y_true: pd.Series,
                         condition: pd.Series, quantiles: list[float]) -> pd.DataFrame:
    """Coverage per condition group (lifetime tercile, truncation fraction,
    temperature regime, ...). Marginal coverage can be perfect while the tail
    is wrong exactly where it matters."""
    frames = []
    for value, idx in y_true.groupby(condition).groups.items():
        sub = coverage_table(pred_q.loc[idx], y_true.loc[idx], quantiles)
        sub["condition"] = value
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)
