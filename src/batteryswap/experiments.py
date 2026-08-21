"""Experiment tracking (§17) — build for the publication track from day one,
because retrofitting rigor at the end is impossible.

Every run gets its own directory holding config, metrics, git SHA, and an
environment lockfile. Nothing here touches the submission path, so the
non-determinism of a wall-clock directory name is safe: §15's determinism
requirement is about the produced submission, not about run bookkeeping.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                     "lifelines", "ortools", "pyarrow")


def git_sha() -> str:
    """Current commit, with a '-dirty' marker if the tree has changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    else:
        return f"{sha}-dirty" if dirty else sha


def environment_summary() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": packages}


def record_run(name: str, config: dict, metrics: dict,
               tables: dict[str, object] | None = None,
               root: Path | None = None) -> Path:
    """Write one experiment directory; returns its path.

    `tables` maps filename stem -> DataFrame, written as CSV (sweeps, per-fold
    results, coverage tables — the things a reviewer asks to see).
    """
    root = root or EXPERIMENTS_DIR
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str),
                                         encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str),
                                          encoding="utf-8")
    (run_dir / "git.txt").write_text(git_sha() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_summary(), indent=2), encoding="utf-8")

    for stem, table in (tables or {}).items():
        table.to_csv(run_dir / f"{stem}.csv", index=False)

    return run_dir
