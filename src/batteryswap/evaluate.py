"""Thin wrapper around the OFFICIAL scorer (§5.3).

⛔ This module NEVER reimplements the cost function. It imports the official
evaluator from the path configured in base.yaml. Any fast surrogate used
inside optimization loops must prove numerical agreement with the official
scorer on >= 1000 random plans (tests/test_evaluator_agreement.py) before it
may be used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .config import load_config

SURROGATE_TOLERANCE = 1e-6  # documented max absolute deviation; revisit in Phase 0


class OfficialEvaluatorMissingError(RuntimeError):
    def __init__(self, path: Path):
        super().__init__(
            f"Official evaluator not found at '{path}'. Place the official scoring "
            f"code under docs/official/evaluator/ (or update base.yaml paths). "
            f"Reimplementing the scorer is forbidden (§18)."
        )


def load_official_evaluator():
    """Import the official scoring module.

    The exact module/entry-point name is unknown until the official release is
    inspected; this loader finds a single top-level .py in the configured
    directory and imports it. Pin the real name here in Phase 0.
    """
    base = load_config("base")
    ev_dir = Path(base["paths"]["official_evaluator"])
    if not ev_dir.exists():
        raise OfficialEvaluatorMissingError(ev_dir)

    candidates = sorted(ev_dir.glob("*.py"))
    if len(candidates) != 1:
        raise OfficialEvaluatorMissingError(ev_dir)

    spec = importlib.util.spec_from_file_location("official_evaluator", candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluator"] = module
    spec.loader.exec_module(module)
    return module


def score_plan(plan, *args, **kwargs) -> float:
    """Score a plan with the official evaluator. Signature is pinned in
    Phase 0 once the official interface is known."""
    module = load_official_evaluator()
    for name in ("score", "evaluate", "compute_cost", "main"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn(plan, *args, **kwargs)
    raise RuntimeError(
        "Official evaluator module exposes no known entry point "
        "(tried: score/evaluate/compute_cost/main). Pin the real one in evaluate.py."
    )
