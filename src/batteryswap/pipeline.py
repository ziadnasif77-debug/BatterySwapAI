"""The production pipeline on the official data: raw -> submission.

    daily series -> features at each scenario cutoff -> B3 quantile RUL
    -> calibration -> expected-penalty curves -> device selection
    -> day assignment (opportunistic batching + ALNS) -> submission

Two properties matter more than anything else here:

* No future data. Every feature for scenario S is computed from observations
  at or before ``S.start_time``; ground-truth EOL, which lives past the
  cutoff, is used only for scoring.
* Cost is the objective. Selection asks "does replacing this device beat
  skipping it, once the marginal travel is charged", never "is the RUL
  prediction accurate".

⚠️ The objective is scoring.ScenarioScorer - OUR reimplementation of the cost
function, not the official scorer. See that module for the exception under
which it exists and the assumptions it rests on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import apply_offsets, coverage_table, cqr_offsets
from .features import assemble_matrix, build_feature_table
from .io import RawData, Scenario
from .labels import FAILURE_VOLTAGE, eol_days_relative
from .metrics import mae, pinball_loss, q_col
from .models.sklearn_quantile import SklearnQuantileModel
from .optimizer import Schedule
from .optimizer.alns import ALNSConfig, alns_search, day_removal, make_building_removal
from .optimizer.opportunistic import opportunistic_upgrade
from .scoring import CostBreakdown, ScenarioScorer

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]

GBM_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 50,      # many correlated cutoffs per device
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}


# --------------------------------------------------------------------------
# 1. daily aggregation
# --------------------------------------------------------------------------

def daily_series(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hourly frame to one row per device-day.

    The 24 h median is the same robust statistic the EOL rule uses, so labels
    and features see the same view of the signal, and it removes the large
    diurnal temperature swing that dominates raw voltage near end of life.
    """
    m = metrics.dropna(subset=["voltage"]).copy()
    m["day"] = m["end_time"].dt.floor("D")
    out = (m.groupby(["device_id", "day"], observed=True)
             .agg(voltage=("voltage", "median"), temperature=("temperature", "mean"))
             .reset_index()
             .rename(columns={"day": "timestamp"}))
    out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    return out.sort_values(["device_id", "timestamp"], ignore_index=True)


# --------------------------------------------------------------------------
# 2. training examples at the scenario cutoffs
# --------------------------------------------------------------------------

def scenario_cutoffs(truth: pd.DataFrame, scenarios: list[Scenario],
                     grace_days: float, horizon: int) -> pd.DataFrame:
    """One (device, scenario-start) row per pair, with the true days-to-EOL.

    ⛔ The cutoff distribution is not estimated here - the scenarios ARE the
    evaluation cutoffs, so training and evaluation see identical truncation by
    construction.
    """
    rows = []
    for s in scenarios:
        days = eol_days_relative(truth, s.start_time, horizon, grace_days)
        censored = dict(zip(truth["device_id"], truth["censored"]))
        for device_id, rul in days.items():
            rows.append({"device_id": device_id,
                         "cutoff": pd.Timestamp(s.start_time).tz_localize("UTC"),
                         "scenario": s.name, "rul_days": rul,
                         "censored": bool(censored[device_id])})
    return pd.DataFrame(rows)


def build_matrix(daily: pd.DataFrame, cutoffs: pd.DataFrame,
                 threshold: float = FAILURE_VOLTAGE
                 ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Features for every (device, cutoff) pair. Drops pairs whose device has
    no observation before the cutoff (installed later)."""
    first_seen = daily.groupby("device_id")["timestamp"].min()
    usable = cutoffs[cutoffs.apply(
        lambda r: r["cutoff"] > first_seen.get(r["device_id"], pd.Timestamp.max), axis=1)]
    table = build_feature_table(daily, usable, threshold, id_column="device_id")
    table["scenario"] = usable["scenario"].to_numpy()
    table["censored"] = usable["censored"].to_numpy()
    return assemble_matrix(table)


# --------------------------------------------------------------------------
# 3. model
# --------------------------------------------------------------------------

@dataclass
class TrainedModel:
    model: SklearnQuantileModel
    offsets: dict[float, float]
    coverage: pd.DataFrame
    diagnostics: dict

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return apply_offsets(self.model.predict_quantiles(X), self.offsets)


def training_targets(meta: pd.DataFrame, truth: pd.DataFrame,
                     horizon: int) -> tuple[pd.Series, np.ndarray]:
    """Honest targets for every (device, cutoff) row, plus a usable-row mask.

    Censoring is the whole difficulty here, and two wrong ways to handle it
    were measured before this one:

    * Using the censoring bound ``last_observed + grace - cutoff`` as a point
      target teaches the model that every device dies when the dataset ends -
      the target shrinks mechanically as the cutoff advances. First end-to-end
      run: 87 devices replaced per scenario instead of ~10.
    * Dropping censored rows entirely leaves only devices that failed, so the
      model never sees a long survivor and predicts short for everything.
      Second run: worse still, 127 per scenario.

    What is actually knowable: a censored device observed alive for at least
    the full horizon after the cutoff **provably did not fail in that window**.
    That is a real label, not an assumption. Rows where the device is censored
    AND the remaining observation is shorter than the horizon are genuinely
    unknown, and are dropped.

    The residual bias is that a survivor's target is its OBSERVED survival, a
    lower bound on its true life. That understates long RULs - which the
    decision layer never resolves anyway, since it only asks whether failure
    lands inside a 42-day window.
    """
    last_seen = dict(zip(truth["device_id"], truth["last_observed"]))
    eol_of = dict(zip(truth["device_id"], truth["eol_time"]))

    device_ids = meta["device_id"].astype(str).to_numpy()
    cutoffs = pd.to_datetime(meta["cutoff"], utc=True)
    censored = meta["censored"].to_numpy(dtype=bool)

    target = np.empty(len(meta), dtype=float)
    usable = np.zeros(len(meta), dtype=bool)
    for i, (device_id, cutoff) in enumerate(zip(device_ids, cutoffs)):
        if not censored[i]:
            days = (eol_of[device_id] - cutoff) / pd.Timedelta(days=1)
            target[i] = days
            # Already-dead rows are KEPT with a negative target. The quantile
            # model clips predictions at zero, so they teach it to say "dead
            # now" for a device whose voltage is already below threshold -
            # without them it has never seen one and schedules it weeks late,
            # which is where 93% of the cost sat before this was fixed.
            usable[i] = True
        else:
            observed = (last_seen[device_id] - cutoff) / pd.Timedelta(days=1)
            target[i] = observed
            usable[i] = observed >= horizon  # provably survived the window
    return pd.Series(target, index=meta.index, name="rul_days"), usable


def train(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int,
          calib_fraction: float = 0.25) -> TrainedModel:
    """Fit B3 and conformalise it on held-out BUILDINGS (never rows)."""
    buildings = sorted(groups.unique())
    n_calib = max(1, round(calib_fraction * len(buildings)))
    rng = np.random.default_rng(seed)
    calib_buildings = set(rng.choice(buildings, size=n_calib, replace=False))

    is_calib = groups.isin(calib_buildings).to_numpy()
    X_fit, y_fit = X[~is_calib], y[~is_calib]
    X_cal, y_cal = X[is_calib], y[is_calib]

    # ⛔ sklearn, not LightGBM: the competition runtime has no lightgbm,
    # so a pickled planner carrying LightGBM boosters cannot be scored.
    model = SklearnQuantileModel(QUANTILES, seed=seed).fit(X_fit, y_fit)
    raw = model.predict_quantiles(X_cal)
    offsets = cqr_offsets(raw, y_cal, QUANTILES)
    calibrated = apply_offsets(raw, offsets)

    y_arr = y_cal.to_numpy(dtype=float)
    diagnostics = {
        "n_fit": len(X_fit), "n_calib": len(X_cal),
        "fit_buildings": int(len(buildings) - len(calib_buildings)),
        "calib_buildings": sorted(calib_buildings),
        "mae_days_raw": mae(raw[q_col(0.50)].to_numpy(), y_arr),
        "mae_days_calibrated": mae(calibrated[q_col(0.50)].to_numpy(), y_arr),
        "pinball_raw": pinball_loss(raw, y_arr, QUANTILES),
        "pinball_calibrated": pinball_loss(calibrated, y_arr, QUANTILES),
    }
    return TrainedModel(model=model, offsets=offsets,
                        coverage=coverage_table(calibrated, y_cal, QUANTILES),
                        diagnostics=diagnostics)


# --------------------------------------------------------------------------
# 4. decision layer + planner
# --------------------------------------------------------------------------

def expected_curves(pred_q: pd.DataFrame, device_ids: list[str],
                    scorer: ScenarioScorer, n_samples: int = 200,
                    already_dead: np.ndarray | None = None
                    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Expected penalty per day, and the expected cost of skipping.

    The expectation runs over the calibrated predictive distribution, using
    exactly the penalty shape the scorer applies, so the planner and the
    scorer cannot drift apart.
    """
    levels = np.asarray(QUANTILES, dtype=float)
    cols = [q_col(q) for q in QUANTILES]
    u = (np.arange(n_samples) + 0.5) / n_samples

    curves, skips = {}, {}
    grid = pred_q[cols].to_numpy(dtype=float)
    for i, device_id in enumerate(device_ids):
        if already_dead is not None and already_dead[i]:
            # Observed below the failure voltage AT THE CUTOFF - no prediction
            # needed, and no leakage: this is pre-cutoff data applied with the
            # same rule the ground truth uses. Replace as early as possible.
            curves[device_id] = scorer.penalty_curve(0.0)
            skips[device_id] = scorer.skip_penalty(0.0)
            continue
        samples = np.interp(u, levels, grid[i])
        curves[device_id] = np.mean(
            [scorer.penalty_curve(t) for t in samples], axis=0)
        skips[device_id] = float(np.mean([scorer.skip_penalty(t) for t in samples]))
    return curves, skips


def marginal_work_cost(scorer: ScenarioScorer, device: str) -> float:
    """Cost of servicing one device on a dedicated trip: building + room
    overhead, the swap, and a round trip from the depot."""
    building, _ = scorer.location_of[device]
    travel = scorer.scenario.travel
    depot = scorer.scenario.depot
    s = scorer.s
    return float(travel.loc[depot, building] + travel.loc[building, depot]
                 + s["time_per_building_change_hours"]
                 + s["time_per_room_change_hours"]
                 + s["time_per_battery_hours"])


def plan_scenario(scorer: ScenarioScorer, curves: dict[str, np.ndarray],
                  skips: dict[str, float], q_slack: float = 0.0,
                  alns_iterations: int = 0, seed: int = 0) -> Schedule:
    """Choose which devices to replace and when.

    Selection rule: a device earns a visit when its best achievable expected
    penalty, plus the marginal cost of getting to it, beats the expected cost
    of leaving it alone. `q_slack` shifts that bar - it is the single tunable
    that trades false positives (wasted trips) against misses (downtime).
    """
    candidates = {}
    for device, curve in curves.items():
        best_day = int(np.argmin(curve)) + 1
        gain = skips[device] - float(curve[best_day - 1])
        if gain > marginal_work_cost(scorer, device) * (1.0 - q_slack):
            candidates[device] = best_day
    if not candidates:
        return Schedule(assignments={})

    schedule = Schedule(assignments=candidates)

    # batch devices that share a building onto one visit, amortised per building
    building_of = {d: scorer.location_of[d][0] for d in candidates}
    trip_cost = {}
    for device, building in building_of.items():
        trip_cost.setdefault(building, marginal_work_cost(scorer, device))
    schedule, _ = opportunistic_upgrade(schedule, curves, building_of, trip_cost)

    if alns_iterations:
        population = list(curves)

        def objective(s: Schedule) -> float:
            return scorer.score(s.assignments, population).total

        def repair(partial: Schedule, removed: list[str],
                   rng: np.random.Generator) -> Schedule:
            assignments = dict(partial.assignments)
            for device in (str(x) for x in rng.permutation(np.array(removed))):
                curve = curves[device].copy()
                shared = {d for b, d in assignments.items()
                          if building_of.get(b) == building_of[device]}
                for day in shared:
                    curve[day - 1] -= trip_cost[building_of[device]]
                assignments[device] = int(np.argmin(curve)) + 1
            return Schedule(assignments=assignments)

        schedule, _ = alns_search(
            schedule, objective,
            destroy_ops=[day_removal, make_building_removal(building_of)],
            repair_ops=[repair],
            config=ALNSConfig(iterations=alns_iterations, seed=seed,
                              segment_length=50),
        )
    return schedule


# --------------------------------------------------------------------------
# 5. end to end
# --------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    breakdown: CostBreakdown
    do_nothing: float
    oracle: float
    assignments: dict[str, int] = field(default_factory=dict)


def oracle_cost(scorer: ScenarioScorer, devices: list[str]) -> float:
    """Best achievable with perfect hindsight: every device that fails inside
    the window replaced on its true failure day, batched opportunistically.
    The floor any prediction-driven plan is measured against."""
    curves, skips = {}, {}
    for device in devices:
        curves[device] = scorer.penalty_curve(scorer.eol_days[device])
        skips[device] = scorer.skip_penalty(scorer.eol_days[device])
    return plan_and_score(scorer, curves, skips, devices).total


def plan_and_score(scorer: ScenarioScorer, curves, skips, devices,
                   q_slack: float = 0.0, alns_iterations: int = 0,
                   seed: int = 0) -> CostBreakdown:
    schedule = plan_scenario(scorer, curves, skips, q_slack, alns_iterations, seed)
    return scorer.score(schedule.assignments, devices)


def run_backtest(data: RawData, trained: TrainedModel, X: pd.DataFrame,
                 meta: pd.DataFrame, truth: pd.DataFrame, q_slack: float = 0.0,
                 alns_iterations: int = 0, seed: int = 0,
                 scenarios: list[Scenario] | None = None) -> list[ScenarioResult]:
    """Plan and score every scenario with the model's own predictions."""
    predictions = trained.predict(X)
    predictions.index = meta.index
    location_of = data.location_of
    results = []

    for scenario in (scenarios or data.scenarios):
        mask = (meta["scenario"] == scenario.name).to_numpy()
        if not mask.any():
            continue
        devices = meta.loc[mask, "device_id"].astype(str).tolist()
        eol_days = eol_days_relative(truth, scenario.start_time,
                                     scenario.horizon_days,
                                     float(scenario.settings["unobserved_eol_days"]))
        scorer = ScenarioScorer(scenario, location_of,
                                {d: eol_days[d] for d in devices})
        dead = (X.loc[mask, "v_above_threshold"].to_numpy() <= 0.0
                if "v_above_threshold" in X.columns else None)
        curves, skips = expected_curves(predictions.loc[mask], devices, scorer,
                                        already_dead=dead)
        schedule = plan_scenario(scorer, curves, skips, q_slack,
                                 alns_iterations, seed)
        results.append(ScenarioResult(
            scenario=scenario.name,
            breakdown=scorer.score(schedule.assignments, devices),
            do_nothing=scorer.score({}, devices).total,
            oracle=oracle_cost(scorer, devices),
            assignments=dict(schedule.assignments),
        ))
    return results


def results_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"scenario": r.scenario, "final_cost": r.breakdown.total,
               "do_nothing": r.do_nothing, "oracle": r.oracle}
        row.update({k: v for k, v in r.breakdown.as_dict().items() if k != "total"})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. submission
# --------------------------------------------------------------------------

def submission_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    """The ordered work plan: one row per (scenario, day, device).

    ⚠️ The official column names and day-indexing base were never published.
    Days are 1..planning_window_days (assumption A8) and rows are sorted by
    scenario, then day, then device so the plan reads as an ordered work list.
    Adjust the header here once the real specification is known.
    """
    rows = []
    for r in results:
        by_day: dict[int, list[str]] = defaultdict(list)
        for device, day in r.assignments.items():
            by_day[day].append(device)
        for day in sorted(by_day):
            for device in sorted(by_day[day]):
                rows.append({"scenario": r.scenario, "day": int(day),
                             "device": device})
    return pd.DataFrame(rows, columns=["scenario", "day", "device"])


def write_submission(results: list[ScenarioResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission_frame(results).to_csv(path, index=False)
    return path
