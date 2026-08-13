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

    # Pipeline stages wired up phase by phase; until then they fail fast and loud.
    for name in ("features", "train", "calibrate", "decide", "plan", "submit", "simulate"):
        sub.add_parser(name).set_defaults(fn=_not_yet(name))

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
