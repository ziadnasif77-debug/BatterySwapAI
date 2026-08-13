"""Command-line entry points — one subcommand per Makefile target.

Every command is deterministic and non-interactive. Commands that need
unresolved constants fail fast with an UnknownValueError naming the exact
missing key.
"""

from __future__ import annotations

import argparse
import sys

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
    print("DRY RUN — synthetic data + synthetic cost model (rehearsal only; "
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
        cov_path = plot_coverage(res.coverage, title="Dry run — calibrated coverage")
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
    """Grouped CV on the synthetic fleet — proves the harness before data day."""
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
    print("GroupKFold on building_id — per-fold (⛔ never a random row/battery split):")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nacross folds (⛔ report variance, not just the mean):")
    print(summarize_folds(folds).to_string(float_format=lambda v: f"{v:.3f}"))
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
                                         "data — needs no official files")
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--buildings", type=int, default=8)
    dry.add_argument("--alns-iterations", type=int, default=1500)
    dry.add_argument("--saa", action="store_true",
                     help="optimize on scenario-averaged curves (§12.5)")
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

    # Pipeline stages wired up phase by phase; until then they fail fast and loud.
    for name in ("features", "train", "calibrate", "decide", "plan", "submit", "simulate"):
        sub.add_parser(name).set_defaults(fn=_not_yet(name))

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
