"""Non-crossing guarantee — no quantile crossing, at any level, for any battery."""

import numpy as np

from batteryswap.calibration import apply_offsets, coverage_table
from batteryswap.models.gbm_quantile import enforce_non_crossing


def test_enforce_non_crossing_sorts_rows():
    crossed = np.array([[5.0, 3.0, 9.0],
                        [1.0, 1.0, 0.5]])
    fixed = enforce_non_crossing(crossed)
    assert (np.diff(fixed, axis=1) >= 0).all()
    # values are preserved, only reordered
    assert sorted(crossed[0]) == list(fixed[0])


def test_calibration_offsets_do_not_reintroduce_crossing():
    import pandas as pd
    rng = np.random.default_rng(0)
    q = np.sort(rng.uniform(0, 1000, size=(50, 3)), axis=1)
    pred = pd.DataFrame(q, columns=["q05", "q50", "q95"])
    # deliberately adversarial offsets
    out = apply_offsets(pred, {0.05: -400.0, 0.50: 0.0, 0.95: 350.0})
    values = out[["q05", "q50", "q95"]].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (values >= 0).all()


def test_coverage_table_on_perfect_quantiles():
    import pandas as pd
    rng = np.random.default_rng(1)
    y = pd.Series(rng.exponential(500, size=5000))
    pred = pd.DataFrame({
        "q10": np.full(len(y), np.quantile(y, 0.10)),
        "q50": np.full(len(y), np.quantile(y, 0.50)),
        "q90": np.full(len(y), np.quantile(y, 0.90)),
    })
    cov = coverage_table(pred, y, [0.10, 0.50, 0.90]).set_index("nominal")["empirical"]
    for level in (0.10, 0.50, 0.90):
        assert abs(cov[level] - level) < 0.02
