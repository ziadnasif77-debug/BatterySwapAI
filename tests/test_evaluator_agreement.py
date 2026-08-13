"""§16.4 — surrogate vs official scorer on >= 1000 random plans, within the
documented tolerance. Runs only once the official evaluator is present."""

from pathlib import Path

import pytest

from batteryswap.config import load_config

N_RANDOM_PLANS = 1000


def _official_evaluator_present() -> bool:
    ev_dir = Path(load_config("base")["paths"]["official_evaluator"])
    return ev_dir.exists() and any(ev_dir.glob("*.py"))


@pytest.mark.skipif(not _official_evaluator_present(),
                    reason="official evaluator not yet placed under docs/official/evaluator/")
def test_surrogate_matches_official_scorer():
    from batteryswap.evaluate import SURROGATE_TOLERANCE, load_official_evaluator  # noqa: F401
    pytest.fail(
        "Official evaluator detected — implement this test NOW (Phase 0): generate "
        f"{N_RANDOM_PLANS} random plans, score each with the official scorer and the "
        f"surrogate, assert max |delta| < {SURROGATE_TOLERANCE}."
    )
