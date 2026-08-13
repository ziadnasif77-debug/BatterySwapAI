"""BatterySwapAI 2026 — probabilistic RUL estimation + cost-optimal maintenance scheduling.

Pipeline: noisy sensor time-series -> calibrated RUL quantiles -> per-battery
cost curves -> multi-period routing/scheduling -> submission.

The objective is total operational cost as computed by the OFFICIAL evaluator.
Prediction error metrics are diagnostics only.
"""

__version__ = "0.1.0"
