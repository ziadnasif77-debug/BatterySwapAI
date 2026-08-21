"""The submission bundle must survive the competition container.

The official image copies only ``requirements.txt``, ``script.py`` and
``batteryswap_example/``, and installs a fixed package set. Everything here
checks a way a submission can look fine locally and fail there.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from batteryswap.bundle import build_bundle, verify_bundle

OFFICIAL_REQUIREMENTS = Path("docs/official/evaluator/requirements.txt")

# The competition fixes this set and ignores edits to requirements.txt. It is
# repeated here so the verification path runs without the official clone, which
# is git-ignored and therefore absent on every fresh checkout, CI included.
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels",
                    "lifelines", "ortools", "polars", "torch")


class _Dummy:
    """Stand-in planner: the bundle machinery must not need a trained model."""

    def plan(self, battery_data, locations, travel_costs, settings):
        raise NotImplementedError


@pytest.fixture()
def official_repo(tmp_path) -> Path:
    """The three files build_bundle copies out of the official example repo.

    That repo is cloned, not vendored (it is theirs), so `.gitignore` keeps it
    out and no fresh checkout has it. Standing in for it here is what lets
    verify_bundle be tested at all on CI; when the real clone IS present it is
    used instead, so the checks run against the genuine files locally.
    """
    if OFFICIAL_REQUIREMENTS.exists():
        return OFFICIAL_REQUIREMENTS.parent
    repo = tmp_path / "official"
    repo.mkdir()
    (repo / "requirements.txt").write_text("\n".join(RUNTIME_PACKAGES) + "\n",
                                           encoding="utf-8")
    (repo / "script.py").write_text("# stand-in for the official harness\n",
                                    encoding="utf-8")
    (repo / "Dockerfile").write_text("# stand-in for the official image\n",
                                     encoding="utf-8")
    return repo


@pytest.fixture()
def bundle(tmp_path, official_repo) -> Path:
    return build_bundle(_Dummy(), out_dir=tmp_path / "submission",
                        official_repo=official_repo)


def test_bundle_has_the_files_the_container_copies(bundle):
    assert (bundle / "batteryswap_example" / "train.py").exists()
    assert (bundle / "batteryswap_example" / "planners" / "best.pickle").exists()
    assert (bundle / "batteryswap_example" / "batteryswap" / "planner.py").exists()


def test_our_package_travels_inside_batteryswap_example(bundle):
    """A pickle referencing batteryswap.planner only loads if the package
    ships inside the one directory the Dockerfile copies."""
    vendored = bundle / "batteryswap_example" / "batteryswap"
    assert vendored.is_dir()
    for module in ("planner.py", "features.py", "calibration.py", "metrics.py",
                   "labels.py"):
        assert (vendored / module).exists(), f"{module} missing from the bundle"


def test_train_module_puts_the_vendored_package_on_the_path(bundle):
    text = (bundle / "batteryswap_example" / "train.py").read_text(encoding="utf-8")
    assert "sys.path" in text
    assert "from batteryswap.planner import BatterySwapPlanner" in text


def test_bundle_excludes_modules_that_cannot_run_there(bundle):
    """scoring.py, dryrun.py and synthetic.py are development-only; shipping
    them would drag matplotlib and the synthetic harness into the image."""
    vendored = bundle / "batteryswap_example" / "batteryswap"
    for excluded in ("dryrun.py", "synthetic.py", "scoring.py"):
        assert not (vendored / excluded).exists()


def test_no_pycache_ships(bundle):
    assert not list((bundle / "batteryswap_example").rglob("__pycache__"))


def test_verify_reports_a_clean_bundle(bundle):
    assert verify_bundle(bundle) == []


def test_verify_catches_a_missing_pickle(bundle):
    (bundle / "batteryswap_example" / "planners" / "best.pickle").unlink()
    problems = verify_bundle(bundle)
    assert any("best.pickle" in p for p in problems)


def test_verify_catches_an_import_the_runtime_lacks(bundle):
    """The failure that started this: LightGBM is not in the competition
    runtime, so a planner carrying LightGBM boosters cannot be scored."""
    target = bundle / "batteryswap_example" / "batteryswap" / "planner.py"
    target.write_text("import lightgbm\n" + target.read_text(encoding="utf-8"),
                      encoding="utf-8")
    problems = verify_bundle(bundle)
    assert any("lightgbm" in p for p in problems)


@pytest.mark.skipif(not OFFICIAL_REQUIREMENTS.exists(),
                    reason="official example repo not cloned")
def test_the_shipped_model_uses_only_runtime_packages():
    """Guards the whole reason SklearnQuantileModel exists."""
    names = {line.split(">=")[0].split("==")[0].strip().lower()
             for line in OFFICIAL_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")}
    assert "scikit-learn" in names
    for absent in ("lightgbm", "xgboost", "catboost"):
        assert absent not in names, f"{absent} IS available - revisit the model choice"


def test_pickle_round_trips_through_the_bundled_path(bundle, monkeypatch):
    """Load the pickle the way script.py does, with only the bundle on sys.path."""
    example = bundle / "batteryswap_example"
    monkeypatch.syspath_prepend(str(example))
    with open(example / "planners" / "best.pickle", "rb") as fh:
        loaded = pickle.load(fh)
    assert hasattr(loaded, "plan")


def test_real_planner_pickles_into_a_loadable_bundle(tmp_path, monkeypatch,
                                                    official_repo):
    """End to end with the actual class, not the stub."""
    from batteryswap.models.sklearn_quantile import SklearnQuantileModel
    from batteryswap.planner import BatterySwapPlanner

    planner = BatterySwapPlanner(SklearnQuantileModel([0.5]), offsets={},
                                 quantiles=[0.5])
    out = build_bundle(planner, out_dir=tmp_path / "submission",
                       official_repo=official_repo)
    assert verify_bundle(out) == []

    example = out / "batteryswap_example"
    monkeypatch.syspath_prepend(str(example))
    # batteryswap is already imported from the source tree here; the pickle
    # resolves against either copy, which is what makes this loadable at all.
    with open(example / "planners" / "best.pickle", "rb") as fh:
        loaded = pickle.load(fh)
    assert isinstance(loaded, BatterySwapPlanner)
    assert loaded.q == planner.q
