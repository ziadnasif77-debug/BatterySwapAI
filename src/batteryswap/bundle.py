"""Build the submission bundle the competition container actually runs.

The official Dockerfile copies exactly three things into the image:
``requirements.txt``, ``script.py`` and ``batteryswap_example/``. Nothing else
from a submission repository exists at runtime, and the runtime's package set
is fixed by the competition - the example repo says in as many words that
edits to ``requirements.txt`` are ignored.

Two consequences, both of which will break a submission silently if missed:

1. **Our package must travel inside ``batteryswap_example/``.** A pickle
   referring to ``batteryswap.planner`` unpickles only if ``batteryswap`` is
   importable, so the package is vendored into the bundle and
   ``batteryswap_example/train.py`` puts it on ``sys.path`` before anything
   else runs. ``script.py`` imports that module first, which is what makes the
   ordering work.
2. **No LightGBM.** The runtime ships scikit-learn, scipy, statsmodels,
   lifelines, ortools, polars and torch - and no gradient-boosting library
   beyond sklearn's own. That is why the shipped model is
   ``SklearnQuantileModel``; see that module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

TRAIN_PY = '''"""Submission entry module.

``script.py`` imports this before unpickling, so this is where the vendored
package is put on the import path. Keep the import of the planner class: the
pickle records ``batteryswap.planner.BatterySwapPlanner`` and needs it
resolvable at load time.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from batteryswap.planner import BatterySwapPlanner  # noqa: E402,F401
from batteryswap.models.sklearn_quantile import SklearnQuantileModel  # noqa: E402,F401
'''

# An ALLOWLIST, not a denylist: only what plan() actually reaches at runtime
# travels. Everything else in the package pulls in something the competition
# image does not have - matplotlib (plots), PyYAML (config), lightgbm
# (gbm_quantile's optional import) - or is development-only.
VENDORED_MODULES = (
    "__init__.py",
    "planner.py",
    "features.py",
    "labels.py",
    "metrics.py",
    "calibration.py",
    "config.py",
    "io.py",
    "models/__init__.py",
    "models/gbm_quantile.py",      # only for enforce_non_crossing / constraints
    "models/sklearn_quantile.py",
)


def build_bundle(planner, out_dir: str | Path = "submission",
                 package_root: str | Path | None = None,
                 official_repo: str | Path = "docs/official/evaluator") -> Path:
    """Write a ready-to-push submission repository. Returns its path."""
    import pickle

    out = Path(out_dir)
    package_root = Path(package_root or Path(__file__).resolve().parent)
    official = Path(official_repo)

    if out.exists():
        shutil.rmtree(out)
    example = out / "batteryswap_example"
    (example / "planners").mkdir(parents=True)

    # 1. the harness files, verbatim from the official repo when available
    for name in ("script.py", "requirements.txt", "Dockerfile"):
        source = official / name
        if source.exists():
            shutil.copy2(source, out / name)

    # 2. our package, vendored - only the modules plan() reaches
    vendored = example / "batteryswap"
    for rel in VENDORED_MODULES:
        target = vendored / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_root / rel, target)

    # 3. the module script.py imports first
    (example / "__init__.py").write_text("", encoding="utf-8")
    (example / "train.py").write_text(TRAIN_PY, encoding="utf-8")

    # 4. the trained planner
    with open(example / "planners" / "best.pickle", "wb") as fh:
        pickle.dump(planner, fh)

    return out


def verify_bundle(out_dir: str | Path = "submission") -> list[str]:
    """Static checks that catch the failures the container would hit.

    Returns a list of problems; empty means the bundle looks shippable. This is
    not a substitute for running the image, but it is instant and catches the
    two mistakes that actually happen.
    """
    out = Path(out_dir)
    problems: list[str] = []

    required = ["script.py", "requirements.txt",
                "batteryswap_example/train.py",
                "batteryswap_example/planners/best.pickle",
                "batteryswap_example/batteryswap/planner.py"]
    for rel in required:
        if not (out / rel).exists():
            problems.append(f"missing {rel}")

    # Nothing vendored may import, AT MODULE LEVEL, a package the runtime
    # lacks. Function-local and try/except-guarded imports are fine - that is
    # exactly how config.py reaches PyYAML without dragging it into the image -
    # so the check walks the AST rather than grepping text.
    allowed = _runtime_packages(out / "requirements.txt")
    banned = {"lightgbm", "xgboost", "catboost", "matplotlib", "yaml",
              "seaborn", "plotly"} - allowed
    for path in sorted((out / "batteryswap_example").rglob("*.py")):
        for module in _module_level_imports(path) & banned:
            problems.append(f"{path.relative_to(out)} imports '{module}' at "
                            f"module level; it is absent from the runtime")
    return problems


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported when the module is first loaded."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in tree.body:                       # top level only, by construction
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _runtime_packages(requirements: Path) -> set[str]:
    if not requirements.exists():
        return set()
    names = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split(">=")[0].split("==")[0].strip().lower())
    return names
