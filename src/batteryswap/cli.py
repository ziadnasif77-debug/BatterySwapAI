"""Command-line entry points - one subcommand per Makefile target.

Every command is deterministic and non-interactive. Commands that need
unresolved constants fail fast with an UnknownValueError naming the exact
missing key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import unresolved_unknowns


def cmd_status(_args) -> int:
    unknowns = unresolved_unknowns()
    if not unknowns:
        print("All unknowns resolved.")
        return 0
    print(f"{len(unknowns)} unresolved unknowns (configs/unknowns.yaml):\n")
    for name, blocks in unknowns.items():
        print(f"  - {name}\n      blocks: {blocks}")
    return 1


def cmd_validate_data(_args) -> int:
    from .io import validate_data
    validate_data()
    return 0


def cmd_audit(_args) -> int:
    from .audit import write_audit_report
    from .io import load_raw
    path = write_audit_report(load_raw())
    print(f"wrote {path}")
    return 0


def cmd_dry_run(args) -> int:
    from .dryrun import run_dry_run
    res = run_dry_run(seed=args.seed, n_buildings=args.buildings,
                      alns_iterations=args.alns_iterations, use_saa=args.saa)
    print("DRY RUN - synthetic data + synthetic cost model (rehearsal only; "
          "no official constants involved)")
    print(f"  fleet: {res.n_batteries} batteries | split by building: "
          f"{res.n_train} train / {res.n_calib} calib / {res.n_eval} eval")
    print(f"  optimizer curves: {'SAA scenarios' if args.saa else 'analytic'}"
          f"  (max analytic-vs-scenario deviation {res.saa_max_curve_deviation:.4f})")
    print("  --- cost, synthetic units (THE selection column) ---")
    print(f"    independent per-battery argmin : {res.independent_cost:,.1f}")
    print(f"    regret-k construction          : {res.construct_cost:,.1f}")
    print(f"    after ALNS                     : {res.alns_cost:,.1f}")
    print(f"    q-rule at swept q={res.best_q:.2f}      : {res.best_q_cost:,.1f}")
    print(f"    FINAL (best variant)           : {res.final_cost:,.1f}")
    print(f"  q sweep: newsvendor q*={res.q_star:.3f} -> swept optimum "
          f"q={res.best_q:.2f}  (opportunistic moves: {res.opportunistic_moves})")
    print("  --- diagnostics (explain the cost, never select on them) ---")
    print(f"    MAE={res.mae_hours:.0f} h  pinball={res.pinball_loss:.1f}")
    print("    calibrated coverage (nominal -> empirical):")
    for _, r in res.coverage.iterrows():
        print(f"      {r['nominal']:.2f} -> {r['empirical']:.2f}  (n={int(r['n'])})")
    print(f"  KPIs: {res.kpis}")

    if args.report:
        from .plots import plot_coverage, plot_q_sweep
        cov_path = plot_coverage(res.coverage, title="Dry run - calibrated coverage")
        q_path = plot_q_sweep(res.q_sweep, q_star=res.q_star)
        print(f"  wrote {cov_path}\n  wrote {q_path}")

    if args.record:
        from .experiments import record_run
        run_dir = record_run(
            "dry_run",
            config={"seed": args.seed, "buildings": args.buildings,
                    "alns_iterations": args.alns_iterations, "saa": args.saa,
                    "synthetic": True},
            metrics={"final_cost": res.final_cost, "alns_cost": res.alns_cost,
                     "construct_cost": res.construct_cost,
                     "independent_cost": res.independent_cost,
                     "best_q": res.best_q, "q_star": res.q_star,
                     "mae_hours": res.mae_hours, "pinball": res.pinball_loss,
                     **res.kpis},
            tables={"q_sweep": res.q_sweep, "coverage": res.coverage},
        )
        print(f"  recorded {run_dir}")
    return 0


def cmd_cv(args) -> int:
    """Grouped CV on the synthetic fleet - proves the harness before data day."""
    import numpy as np
    import pandas as pd

    from .cv import cross_validate_quantiles, summarize_folds
    from .dryrun import GBM_PARAMS, QUANTILES, SyntheticCostModel
    from .features import assemble_matrix, build_feature_table
    from .labels import compute_eol
    from .sampling import sample_cutoffs
    from .synthetic import synthetic_fleet

    cm = SyntheticCostModel()
    fleet = synthetic_fleet(n_buildings=args.buildings, seed=args.seed)
    building_of = dict(zip(fleet.locations["battery_id"], fleet.locations["building_id"]))

    rows = []
    for b, g in fleet.ts.groupby("battery_id", sort=True):
        eol = compute_eol(g, cm.failure_threshold_v)
        rows.append({"battery_id": b, "eol_time": eol, "censored": eol is None,
                     "last_observed": g["timestamp"].max()})
    labels = pd.DataFrame(rows)

    fracs = np.random.default_rng(args.seed).uniform(0.55, 0.95, size=256)
    cutoffs = sample_cutoffs(labels[~labels["censored"]], fleet.ts, fracs, 4, args.seed)
    table = build_feature_table(fleet.ts, cutoffs, cm.failure_threshold_v)
    X, y, meta = assemble_matrix(table)
    groups = meta["battery_id"].map(building_of)

    folds = cross_validate_quantiles(X, y, groups, QUANTILES, dict(GBM_PARAMS),
                                     seed=args.seed, n_splits=args.folds)
    print("GroupKFold on building_id - per-fold (!! never a random row/battery split):")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nacross folds (!! report variance, not just the mean):")
    print(summarize_folds(folds).to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


# --- official-data pipeline -------------------------------------------------

def _load_or_build(rebuild: bool = False):
    """Cached daily series / EOL truth / feature matrix under data/interim."""
    import pandas as pd

    from .io import load_raw
    from .labels import ground_truth_eol
    from .pipeline import build_matrix, daily_series, scenario_cutoffs

    interim = Path("data/interim")
    interim.mkdir(parents=True, exist_ok=True)
    data = load_raw()

    daily_path = interim / "daily.parquet"
    if rebuild or not daily_path.exists():
        daily_series(data.metrics).to_parquet(daily_path)
    daily = pd.read_parquet(daily_path)

    truth_path = interim / "truth.parquet"
    if rebuild or not truth_path.exists():
        ground_truth_eol(data.metrics, data.devices, data.eol).to_parquet(truth_path)
    truth = pd.read_parquet(truth_path)

    x_path = interim / "X.parquet"
    if rebuild or not x_path.exists():
        cutoffs = scenario_cutoffs(truth, data.scenarios, 30.0,
                                   data.scenarios[0].horizon_days)
        X, y, meta = build_matrix(daily, cutoffs)
        X.to_parquet(x_path)
        y.to_frame("rul_days").to_parquet(interim / "y.parquet")
        meta.to_parquet(interim / "meta.parquet")
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(interim / "y.parquet")["rul_days"]
    meta = pd.read_parquet(interim / "meta.parquet")
    return data, daily, truth, X, y, meta


def cmd_submit(args) -> int:
    from .experiments import record_run
    from .pipeline import results_frame, run_backtest, train, training_targets, write_submission

    data, _, truth, X, y, meta = _load_or_build(args.rebuild)
    building_of = dict(zip(data.devices["device_id"], data.devices["building_id"]))
    groups = meta["device_id"].astype(str).map(building_of)

    horizon = data.scenarios[0].horizon_days
    y, keep = training_targets(meta, truth, horizon)
    print(f"training B3 on {int(keep.sum()):,} of {len(X):,} (device, scenario) rows "
          f"- unknowable rows dropped (see pipeline.training_targets), "
          f"{groups[keep].nunique()} buildings ...")
    trained = train(X[keep], y[keep], groups[keep], seed=args.seed)
    for k, v in trained.diagnostics.items():
        if k != "calib_buildings":
            print(f"  {k}: {v}")

    print(f"planning {len(data.scenarios)} scenarios (q_slack={args.q_slack}) ...")
    results = run_backtest(data, trained, X, meta, truth, q_slack=args.q_slack,
                           alns_iterations=args.alns_iterations, seed=args.seed)
    frame = results_frame(results)

    total, nothing, oracle = (frame["final_cost"].sum(), frame["do_nothing"].sum(),
                              frame["oracle"].sum())
    print()
    print("=== FINAL COST (our scorer - NOT the official one) ===")
    print(f"  do-nothing baseline : {nothing:12,.1f}")
    print(f"  our plan            : {total:12,.1f}"
          f"   ({(nothing - total) / nothing:.1%} better than do-nothing)")
    print(f"  hindsight oracle    : {oracle:12,.1f}"
          f"   (gap to oracle: {total - oracle:,.1f})")
    print(f"  devices replaced    : {int(frame['devices_replaced'].sum())}"
          f" | late: {int(frame['devices_late'].sum())}"
          f" | overtime h: {frame['overtime_hours'].sum():.1f}"
          f" | cap breaches: {int(frame['days_over_cap'].sum() + frame['weeks_over_cap'].sum())}")

    path = write_submission(results, args.out)
    print()
    with open(path, encoding="utf-8") as fh:
        n_rows = sum(1 for _ in fh) - 1
    print(f"wrote {path} ({n_rows} rows)")

    Path("reports").mkdir(exist_ok=True)
    frame.to_csv("reports/score_breakdown.csv", index=False)
    trained.coverage.to_csv("reports/coverage_official.csv", index=False)
    if args.record:
        run_dir = record_run("official_submit",
                             config={"seed": args.seed, "q_slack": args.q_slack,
                                     "alns_iterations": args.alns_iterations},
                             metrics={"final_cost": float(total),
                                      "do_nothing": float(nothing),
                                      "oracle": float(oracle),
                                      **trained.diagnostics},
                             tables={"per_scenario": frame,
                                     "coverage": trained.coverage})
        print(f"recorded {run_dir}")
    return 0


def cmd_sweep(args) -> int:
    """Sweep the selection bar against the scorer - the q sweep of section 11,
    in the form this problem actually takes."""
    import pandas as pd

    from .pipeline import results_frame, run_backtest, train, training_targets

    data, _, truth, X, y, meta = _load_or_build(False)
    building_of = dict(zip(data.devices["device_id"], data.devices["building_id"]))
    groups = meta["device_id"].astype(str).map(building_of)
    y, keep = training_targets(meta, truth, data.scenarios[0].horizon_days)
    trained = train(X[keep], y[keep], groups[keep], seed=args.seed)

    subset = data.scenarios[::args.stride]
    rows = []
    for slack in [float(v) for v in args.values.split(",")]:
        results = run_backtest(data, trained, X, meta, truth, q_slack=slack,
                               seed=args.seed, scenarios=subset)
        frame = results_frame(results)
        rows.append({"q_slack": slack, "final_cost": frame["final_cost"].sum(),
                     "replaced": int(frame["devices_replaced"].sum()),
                     "late": int(frame["devices_late"].sum()),
                     "oracle_gap": frame["final_cost"].sum() - frame["oracle"].sum()})
        print(f"  slack={slack:+.2f} -> cost {rows[-1]['final_cost']:12,.1f} "
              f"replaced={rows[-1]['replaced']:4d} late={rows[-1]['late']:3d}")
    out = pd.DataFrame(rows).sort_values("final_cost")
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/slack_sweep.csv", index=False)
    print()
    print(f"best: q_slack={out.iloc[0]['q_slack']:+.2f} "
          f"cost={out.iloc[0]['final_cost']:,.1f}  (wrote reports/slack_sweep.csv)")
    return 0


# --- official evaluator path ------------------------------------------------

def _official_targets(meta, devices_csv, eol_csv, grace=30.0):
    """The evaluator's own ground truth: recorded EOL, else last data + grace."""
    import pandas as pd

    dev = pd.read_csv(devices_csv, index_col=0)
    eol = pd.read_csv(eol_csv)
    eol["end_time"] = pd.to_datetime(eol["end_time"], format="ISO8601")
    last = dict(zip(dev.device_id,
                    pd.to_datetime(dev.end_time, format="ISO8601").dt.tz_localize(None)))
    lab = dict(zip(eol.device_id, eol.end_time))
    official = {d: (lab[d] if pd.notna(lab.get(d))
                    else (last[d] + pd.Timedelta(days=grace)).normalize())
                for d in dev.device_id}
    cutoff = pd.to_datetime(meta["cutoff"]).dt.tz_localize(None)
    y = pd.Series([(official[d] - c) / pd.Timedelta(days=1)
                   for d, c in zip(meta.device_id.astype(str), cutoff)], index=meta.index)
    groups = meta.device_id.astype(str).map(dict(zip(dev.device_id, dev.building_id)))
    return y, groups


def cmd_official(args) -> int:
    """Train, plan, and score against the OFFICIAL evaluator."""
    import pickle

    import pandas as pd

    from .evaluate import is_available, iter_problems, score_plan
    from .pipeline import QUANTILES, train
    from .planner import SKIP_DATE, BatterySwapPlanner

    if not is_available():
        print("The official evaluator (batteryswap_public) is not installed.",
              file=sys.stderr)
        print("  pip install batteryswap_public fastparquet structlog "
              "pydantic-settings", file=sys.stderr)
        return 1

    X = pd.read_parquet("data/interim/X.parquet")
    meta = pd.read_parquet("data/interim/meta.parquet")
    y, groups = _official_targets(meta, "data/raw/devices.csv", "data/raw/eol_times.csv")
    keep = (y > 0).to_numpy()

    print(f"training B3 on {int(keep.sum()):,}/{len(X):,} rows against the "
          f"evaluator's own EOL definition ...")
    trained = train(X[keep], y[keep], groups[keep], seed=args.seed)
    print(f"  MAE {trained.diagnostics['mae_days_calibrated']:.1f} d | "
          f"pinball {trained.diagnostics['pinball_calibrated']:.2f}")

    planner = BatterySwapPlanner(trained.model, trained.offsets, QUANTILES,
                                 q=args.q, batch_window_days=args.batch,
                                 voltage_margin=args.margin,
                                 max_swaps=args.max_swaps)

    rows = []
    for i, problem in enumerate(iter_problems(args.split_path)):
        if i % args.stride:
            continue
        plan = planner.plan(problem.timeseries, problem.locations,
                            problem.travel_costs, problem.settings)
        scores = score_plan(plan, problem)
        push = pd.DataFrame({"day": [SKIP_DATE] * len(problem.locations),
                             "battery": list(problem.locations["battery"])})
        rows.append({"scenario": problem.name,
                     "ours": scores.total_cost,
                     "no_planning": score_plan(push, problem).total_cost,
                     "early_swap": scores.early_swap, "late_swap": scores.late_swap,
                     "travel": scores.travel, "overtime": scores.overtime,
                     "weekly_limit": scores.weekly_limit,
                     "daily_limit": scores.daily_limit,
                     "in_window": int((plan["day"] <= problem.end_time).sum())})
        print(f"  {problem.name}: {scores.total_cost:9,.1f}   "
              f"(no planning {rows[-1]['no_planning']:9,.1f})")

    frame = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    frame.to_csv("reports/official_scores.csv", index=False)

    ours, base = frame["ours"].mean(), frame["no_planning"].mean()
    print()
    print("=== OFFICIAL SCORE (batteryswap_public.evaluate) ===")
    print(f"  scenarios scored : {len(frame)}")
    print(f"  no planning      : {base:12,.1f}")
    print(f"  ours             : {ours:12,.1f}   ({(base - ours) / base:.1%} better)")
    print(f"  components  early {frame['early_swap'].mean():8,.1f} | "
          f"late {frame['late_swap'].mean():8,.1f} | "
          f"travel {frame['travel'].mean():7,.1f} | "
          f"overtime {frame['overtime'].mean():7,.1f}")
    print(f"  config      q={args.q} batch={args.batch}d margin={args.margin}V "
          f"cap={args.max_swaps}")
    beat = int((frame["ours"] < frame["no_planning"]).sum())
    print(f"  beats no-planning in {beat}/{len(frame)} scenarios")

    Path(args.planner_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.planner_out, "wb") as fh:
        pickle.dump(planner, fh)
    print(f"  wrote {args.planner_out}")

    if args.bundle:
        from .bundle import build_bundle, verify_bundle
        out = build_bundle(planner, args.bundle)
        problems = verify_bundle(out)
        print(f"  built submission bundle at {out}/")
        if problems:
            print("  !! bundle problems:")
            for problem in problems:
                print(f"     - {problem}")
            return 1
        print("  bundle verified: harness files present, no runtime-absent imports")
    return 0


def _not_yet(phase: str):
    def handler(_args) -> int:
        print(f"'{phase}' is blocked: official competition files are not present. "
              f"Run 'batteryswap status' for the full unknowns list.", file=sys.stderr)
        return 1
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batteryswap")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="report unresolved UNKNOWNs").set_defaults(fn=cmd_status)
    sub.add_parser("validate-data", help="validate official raw files").set_defaults(
        fn=cmd_validate_data)
    sub.add_parser("audit", help="write reports/data_audit.md").set_defaults(fn=cmd_audit)

    dry = sub.add_parser("dry-run", help="end-to-end plumbing rehearsal on synthetic "
                                         "data - needs no official files")
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--buildings", type=int, default=8)
    dry.add_argument("--alns-iterations", type=int, default=1500)
    dry.add_argument("--saa", action="store_true",
                     help="optimize on scenario-averaged curves (section 12.5)")
    dry.add_argument("--report", action="store_true",
                     help="write coverage + q-sweep figures to reports/")
    dry.add_argument("--record", action="store_true",
                     help="record config/metrics/git SHA under experiments/")
    dry.set_defaults(fn=cmd_dry_run)

    cv = sub.add_parser("cv", help="grouped (GroupKFold by building) CV on synthetic data")
    cv.add_argument("--seed", type=int, default=0)
    cv.add_argument("--buildings", type=int, default=10)
    cv.add_argument("--folds", type=int, default=5)
    cv.set_defaults(fn=cmd_cv)

    sb = sub.add_parser("submit", help="raw data -> submission (official data)")
    sb.add_argument("--out", default="submissions/submission.csv")
    sb.add_argument("--seed", type=int, default=20260101)
    sb.add_argument("--q-slack", type=float, default=0.0, dest="q_slack")
    sb.add_argument("--alns-iterations", type=int, default=0)
    sb.add_argument("--rebuild", action="store_true", help="rebuild cached features")
    sb.add_argument("--record", action="store_true")
    sb.set_defaults(fn=cmd_submit)

    off = sub.add_parser("official", help="train, plan and score with the OFFICIAL evaluator")
    off.add_argument("--split-path", default="data/raw/train")
    off.add_argument("--seed", type=int, default=20260101)
    off.add_argument("--q", type=float, default=0.12)
    off.add_argument("--batch", type=float, default=14.0)
    off.add_argument("--margin", type=float, default=0.50)
    off.add_argument("--stride", type=int, default=1)
    off.add_argument("--max-swaps", type=int, default=25, dest="max_swaps")
    off.add_argument("--planner-out", default="artifacts/planner.pickle")
    off.add_argument("--bundle", default=None,
                     help="also build a ready-to-push submission repo here")
    off.set_defaults(fn=cmd_official)

    sw = sub.add_parser("sweep", help="sweep the selection bar against the scorer")
    sw.add_argument("--values", default="-0.5,-0.25,0,0.25,0.5,0.75")
    sw.add_argument("--stride", type=int, default=4, help="use every Nth scenario")
    sw.add_argument("--seed", type=int, default=20260101)
    sw.set_defaults(fn=cmd_sweep)

    # Stages still not wired to the official path fail fast and loud.
    for name in ("features", "train", "calibrate", "decide", "plan", "simulate"):
        sub.add_parser(name).set_defaults(fn=_not_yet(name))

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
