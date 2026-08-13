"""§16.2 — features at cutoff t must be byte-identical whether or not data
after t exists in the frame."""

import pandas as pd

from batteryswap.features import compute_features


def test_features_identical_with_and_without_future(synth_ts):
    cutoff = pd.Timestamp("2024-09-15 12:00:00")
    for battery_id in ("b000", "b002"):
        history = synth_ts[synth_ts["battery_id"] == battery_id]
        truncated = history[history["timestamp"] <= cutoff]

        with_future = compute_features(history, cutoff)
        without_future = compute_features(truncated, cutoff)

        assert with_future.keys() == without_future.keys()
        for key in with_future:
            a, b = with_future[key], without_future[key]
            assert (a == b) or (pd.isna(a) and pd.isna(b)), \
                f"feature '{key}' depends on post-cutoff data: {a!r} != {b!r}"
