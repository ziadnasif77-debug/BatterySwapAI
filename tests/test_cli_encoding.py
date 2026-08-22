"""CLI output must survive a non-UTF-8 console.

Found the hard way: `batteryswap cv` died with UnicodeEncodeError on a Windows
console running the cp1256 codepage, because a print statement carried U+26D4.
The command failed before producing a single line of output. Any codepage that
is not UTF-8 - which is the Windows default in most locales - hits this.

Rather than depending on the user's terminal, the CLI stays ASCII.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI = Path("src/batteryswap/cli.py")

# Codepages a Windows user plausibly has. cp1256 is the one that actually broke.
NARROW_ENCODINGS = ("cp1256", "cp1252", "cp437", "ascii")


def test_cli_source_prints_only_ascii():
    """Every string the CLI can print must encode in any of these codepages."""
    source = CLI.read_text(encoding="utf-8")
    offenders = sorted({ch for ch in source if ord(ch) > 127})
    assert not offenders, (
        "non-ASCII in cli.py: "
        + ", ".join(f"U+{ord(c):04X}" for c in offenders)
        + " - a console on a narrow codepage cannot encode these and the "
          "command dies before printing anything"
    )


@pytest.mark.parametrize("encoding", NARROW_ENCODINGS)
def test_cli_source_encodes_under_narrow_codepages(encoding):
    CLI.read_text(encoding="utf-8").encode(encoding)


def test_help_runs_under_a_narrow_codepage():
    """End to end: run the CLI with stdout forced to cp1256."""
    env = {**dict(__import__("os").environ), "PYTHONIOENCODING": "cp1256"}
    result = subprocess.run([sys.executable, "-m", "batteryswap.cli", "--help"],
                            capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert "UnicodeEncodeError" not in result.stderr


def test_status_runs_under_a_narrow_codepage():
    """`status` prints the unknowns table; exit code 1 is its normal signal
    that unresolved entries remain, so only a crash is a failure here."""
    env = {**dict(__import__("os").environ), "PYTHONIOENCODING": "cp1256"}
    result = subprocess.run([sys.executable, "-m", "batteryswap.cli", "status"],
                            capture_output=True, text=True, env=env, check=False)
    assert "UnicodeEncodeError" not in result.stderr
    assert result.returncode in (0, 1), result.stderr
