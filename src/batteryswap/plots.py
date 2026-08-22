"""Report figures — the coverage plot is a mandatory artifact.

Headless by construction: the Agg backend is selected before pyplot is
imported so `make` targets and CI produce figures without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def plot_coverage(coverage: object, path: str | Path | None = None,
                  title: str = "Calibrated quantile coverage") -> Path:
    """Nominal vs empirical coverage against the diagonal.

    ⛔ If the 0.10 quantile does not contain ~10 % of true failures, fix
    calibration before touching the optimizer. This plot is how that is read.
    """
    path = Path(path) if path else REPORTS_DIR / "coverage.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey",
            label="perfect calibration")
    ax.plot(coverage["nominal"], coverage["empirical"], marker="o", label="empirical")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical fraction of truths below")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_q_sweep(sweep: object, path: str | Path | None = None,
                 q_star: float | None = None) -> Path:
    """Final cost against q, with the newsvendor q* marked.

    The gap between q* and the sweep minimum is the empirical answer to the
    central claim — that trip sharing moves the optimum above the theoretical
    value — and belongs in the README either way.
    """
    path = Path(path) if path else REPORTS_DIR / "q_sweep.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["q"], sweep["final_cost"], marker="o", markersize=3)
    best = sweep.loc[sweep["final_cost"].idxmin()]
    ax.axvline(best["q"], color="tab:green", linestyle="-", linewidth=1,
               label=f"sweep minimum q={best['q']:.2f}")
    if q_star is not None:
        ax.axvline(q_star, color="tab:red", linestyle="--", linewidth=1,
                   label=f"newsvendor q*={q_star:.2f}")
    ax.set_xlabel("q (replacement quantile)")
    ax.set_ylabel("final cost")
    ax.set_title("q sweep — final cost is the only selection column")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
