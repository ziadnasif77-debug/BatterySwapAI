"""Pre-data-day rehearsal: the FULL pipeline end-to-end on synthetic data.

labels -> cutoff sampling -> features -> B3 quantiles -> CQR calibration ->
decision cost curves -> regret-k construction -> ALNS -> per-day routing ->
final cost + KPIs.

⛔ Scope guard. Everything here runs on synthetic_fleet() data against
SyntheticCostModel — arbitrary rehearsal constants that never touch
configs/cost_model.yaml, the real pipeline, or any submission artifact, so the
no-invention rule (§0) is preserved. SyntheticCostModel is NOT the official
evaluator and is not a surrogate for it (§5.3/§18 govern those); it exists so
that on data day the only work left is transcribing real constants and
swapping the objective for the official scorer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import apply_offsets, coverage_table, cqr_offsets
from .decision import cost_curves_from_constants
from .features import assemble_matrix, build_feature_table
from .labels import compute_eol
from .models.gbm_quantile import GBMQuantileModel
from .optimizer import Schedule
from .optimizer.alns import (ALNSConfig, alns_search, day_removal,
                             make_building_removal, random_removal)
from .optimizer.construct import regret_k_construct
from .optimizer.routing import plan_day
from .sampling import sample_cutoffs
from .synthetic import synthetic_fleet

HOUR = pd.Timedelta(hours=1)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

GBM_PARAMS = {  # small-data rehearsal settings, not competition hyperparameters
    "num_leaves": 15,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "min_child_samples": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
}


@dataclass(frozen=True)
class SyntheticCostModel:
    """Arbitrary rehearsal constants — NEVER competition values."""
    failure_threshold_v: float = 2.6
    swap_minutes: float = 10.0
    room_change_minutes: float = 5.0
    building_change_minutes: float = 10.0
    day_cap_minutes: float = 480.0
    minute_cost: float = 1.0
    overtime_multiplier: float = 2.0
    c_early_per_hour: float = 0.05
    c_late_per_hour: float = 0.5


@dataclass
class DryRunResult:
    n_batteries: int
    n_train: int
    n_calib: int
    n_eval: int
    mae_hours: float
    pinball_loss: float
    coverage: pd.DataFrame
    independent_cost: float   # each battery at its own curve argmin, no grouping
    construct_cost: float     # regret-k with building-cohesion heuristic
    final_cost: float         # after ALNS against the synthetic objective
    schedule: Schedule
    kpis: dict = field(default_factory=dict)


# --- synthetic objective ------------------------------------------------------

def _day_service_minutes(batteries: list[str], loc: dict[str, tuple[str, str]],
                         cm: SyntheticCostModel) -> dict[str, float]:
    """building -> service minutes for one day's visit set."""
    per_building_batteries: dict[str, int] = defaultdict(int)
    per_building_rooms: dict[str, set] = defaultdict(set)
    for b in batteries:
        building, room = loc[b]
        per_building_batteries[building] += 1
        per_building_rooms[building].add(room)
    return {
        bl: (per_building_batteries[bl] * cm.swap_minutes
             + len(per_building_rooms[bl]) * cm.room_change_minutes
             + cm.building_change_minutes)
        for bl in per_building_batteries
    }


def route_schedule(schedule: Schedule, loc: dict[str, tuple[str, str]],
                   travel: pd.DataFrame, cm: SyntheticCostModel) -> dict[int, "object"]:
    """Deterministic per-day routing (NN + 2-opt) for every used day."""
    by_day: dict[int, list[str]] = defaultdict(list)
    for b, d in schedule.assignments.items():
        by_day[d].append(b)
    plans = {}
    for day in sorted(by_day):
        batteries = sorted(by_day[day])
        service = _day_service_minutes(batteries, loc, cm)
        plans[day] = plan_day(sorted(service), service, travel, cm.day_cap_minutes)
    return plans


def schedule_cost(schedule: Schedule, curves: dict[str, np.ndarray],
                  loc: dict[str, tuple[str, str]], travel: pd.DataFrame,
                  cm: SyntheticCostModel) -> float:
    """Synthetic objective: expected penalties + worker time + overtime."""
    total = sum(curves[b][d - 1] for b, d in schedule.assignments.items())
    for plan in route_schedule(schedule, loc, travel, cm).values():
        total += cm.minute_cost * plan.total_minutes
        total += (cm.overtime_multiplier - 1.0) * cm.minute_cost * plan.overtime_minutes
    return float(total)


# --- pipeline -----------------------------------------------------------------

def _pinball(pred_q: pd.DataFrame, y: np.ndarray, quantiles: list[float]) -> float:
    losses = []
    for q in quantiles:
        e = y - pred_q[f"q{int(round(q * 100)):02d}"].to_numpy()
        losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    return float(np.mean(losses))


def run_dry_run(seed: int = 0, n_buildings: int = 8,
                n_cutoffs_per_battery: int = 4, alns_iterations: int = 1500,
                life_days_range: tuple[int, int] = (550, 750),
                cm: SyntheticCostModel = SyntheticCostModel()) -> DryRunResult:
    rng = np.random.default_rng(seed)
    fleet = synthetic_fleet(n_buildings=n_buildings, seed=seed,
                            life_days_range=life_days_range)
    loc = {r.battery_id: (r.building_id, r.room_id)
           for r in fleet.locations.itertuples()}
    battery_building = {b: bl for b, (bl, _) in loc.items()}

    # labels on the full series (training-set semantics: runs past EOL)
    labels = []
    for b, g in fleet.ts.groupby("battery_id", sort=True):
        eol = compute_eol(g, cm.failure_threshold_v)
        labels.append({"battery_id": b, "eol_time": eol, "censored": eol is None,
                       "last_observed": g["timestamp"].max()})
    labels = pd.DataFrame(labels)
    usable = labels[~labels["censored"]]

    # grouped split by BUILDING (§7): never rows, never batteries within building
    buildings = sorted(fleet.locations["building_id"].unique())
    n_train = max(1, round(0.5 * len(buildings)))
    n_calib = max(1, round(0.25 * len(buildings)))
    train_bl = set(buildings[:n_train])
    calib_bl = set(buildings[n_train:n_train + n_calib])
    eval_bl = set(buildings[n_train + n_calib:]) or {buildings[-1]}

    def batteries_in(bls: set) -> list[str]:
        return sorted(b for b in usable["battery_id"] if battery_building[b] in bls)

    train_b, calib_b, eval_b = (batteries_in(s) for s in (train_bl, calib_bl, eval_bl))
    ts_of = dict(tuple(fleet.ts.groupby("battery_id")))

    # cutoff-matched sampling: synthetic truncation fractions (rehearsal stand-in
    # for the estimate from the real evaluation series lengths)
    fracs = rng.uniform(0.55, 0.95, size=256)
    train_ts = fleet.ts[fleet.ts["battery_id"].isin(train_b)]
    train_cut = sample_cutoffs(labels[labels["battery_id"].isin(train_b)],
                               train_ts, fracs, n_cutoffs_per_battery, seed)

    def single_cutoffs(batteries: list[str]) -> pd.DataFrame:
        rows = []
        for b in batteries:
            lab = labels.loc[labels["battery_id"] == b].iloc[0]
            start = ts_of[b]["timestamp"].min()
            f = float(rng.uniform(0.55, 0.9))
            cutoff = start + f * (lab["eol_time"] - start)
            rows.append({"battery_id": b, "cutoff": cutoff,
                         "rul_hours": (lab["eol_time"] - cutoff) / HOUR,
                         "censored": False})
        return pd.DataFrame(rows)

    calib_cut, eval_cut = single_cutoffs(calib_b), single_cutoffs(eval_b)

    # features -> B3 -> CQR
    thr = cm.failure_threshold_v
    X_tr, y_tr, _ = assemble_matrix(build_feature_table(fleet.ts, train_cut, thr))
    X_ca, y_ca, _ = assemble_matrix(build_feature_table(fleet.ts, calib_cut, thr))
    X_ev, y_ev, _ = assemble_matrix(build_feature_table(fleet.ts, eval_cut, thr))

    model = GBMQuantileModel(QUANTILES, dict(GBM_PARAMS), seed=seed).fit(X_tr, y_tr)
    offsets = cqr_offsets(model.predict_quantiles(X_ca), y_ca, QUANTILES)
    calibrated = apply_offsets(model.predict_quantiles(X_ev), offsets)
    calibrated["battery_id"] = eval_cut["battery_id"].to_numpy()

    coverage = coverage_table(calibrated, y_ev, QUANTILES)
    y = y_ev.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(calibrated["q50"].to_numpy() - y)))
    pinball = _pinball(calibrated, y, QUANTILES)

    # decision layer -> curves dict
    horizon = int(np.ceil(calibrated["q95"].max() / 24.0)) + 30
    curve_df = cost_curves_from_constants(calibrated, QUANTILES,
                                          cm.c_early_per_hour, cm.c_late_per_hour, horizon)
    curves = {b: g.sort_values("day")["expected_penalty"].to_numpy()
              for b, g in curve_df.groupby("battery_id")}

    travel = fleet.travel
    off_diag = travel.to_numpy()[~np.eye(len(travel), dtype=bool)]
    # amortized value of sharing a trip: building overhead + typical travel leg
    cohesion = cm.minute_cost * (cm.building_change_minutes + float(np.mean(off_diag)))

    def insertion_cost(schedule: Schedule, b: str, d: int) -> float:
        c = float(curves[b][d - 1])
        shared = any(battery_building[b2] == battery_building[b]
                     for b2, d2 in schedule.assignments.items() if d2 == d)
        return c - cohesion if shared else c

    candidate_days = list(range(1, horizon + 1))
    independent = Schedule(assignments={b: int(np.argmin(curves[b])) + 1 for b in eval_b})
    constructed = regret_k_construct(eval_b, candidate_days, insertion_cost, k=3)

    objective = lambda s: schedule_cost(s, curves, loc, travel, cm)  # noqa: E731

    def curve_repair(partial: Schedule, removed: list[str],
                     op_rng: np.random.Generator) -> Schedule:
        assignments = dict(partial.assignments)
        for b in (str(x) for x in op_rng.permutation(np.array(removed))):
            arr = curves[b].copy()
            for d in {d2 for b2, d2 in assignments.items()
                      if battery_building[b2] == battery_building[b]}:
                arr[d - 1] -= cohesion
            assignments[b] = int(np.argmin(arr)) + 1
        return Schedule(assignments=assignments)

    best, final_cost = alns_search(
        constructed, objective,
        destroy_ops=[random_removal, day_removal,
                     make_building_removal(battery_building)],
        repair_ops=[curve_repair],
        config=ALNSConfig(iterations=alns_iterations, seed=seed),
    )

    # KPIs on the final schedule
    plans = route_schedule(best, loc, travel, cm)
    true_rul = dict(zip(eval_cut["battery_id"], eval_cut["rul_hours"]))
    late = sum(1 for b, d in best.assignments.items() if d * 24.0 > true_rul[b])
    wasted = sum(max(true_rul[b] - d * 24.0, 0.0) for b, d in best.assignments.items())
    kpis = {
        "days_used": len(plans),
        "building_visits": int(sum(len(p.order) for p in plans.values())),
        "travel_minutes": round(sum(p.travel_minutes for p in plans.values()), 1),
        "service_minutes": round(sum(p.service_minutes for p in plans.values()), 1),
        "overtime_minutes": round(sum(p.overtime_minutes for p in plans.values()), 1),
        "batteries_late": late,
        "wasted_life_hours": round(wasted, 1),
        "all_assigned": set(best.assignments) == set(eval_b),
        "horizon_days": horizon,
        "censored_dropped": int(labels["censored"].sum()),
    }

    return DryRunResult(
        n_batteries=len(labels), n_train=len(train_b), n_calib=len(calib_b),
        n_eval=len(eval_b), mae_hours=mae, pinball_loss=pinball, coverage=coverage,
        independent_cost=objective(independent),
        construct_cost=objective(constructed),
        final_cost=final_cost, schedule=best, kpis=kpis,
    )
