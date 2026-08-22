"""The image must not bake in a split selection.

The official example ships `ENV BATTERYSWAP_SPLITS=train` so that a local
`docker run` scores against the only split an entrant actually has. Carried
into a real submission that value is wrong: the organisers score `public` and
`private`. If their harness builds the submitted Dockerfile without overriding
the variable, the run produces train rows only and both scored splits come
back as infinity - a total loss that no local test would reveal.

Removing the line is correct under either reading of how they invoke it:
`script.py` already defaults to `public,private` when the variable is unset,
and an explicit override from their harness still wins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batteryswap.bundle import _patch_dockerfile

OFFICIAL_DOCKERFILE = Path("docs/official/evaluator/Dockerfile")

SAMPLE = """
FROM huggingface/competitions:latest

# Default to running on train split only
ENV BATTERYSWAP_SPLITS=train
ENV BATTERYSWAP_SUBMISSION_PATH=submission.csv

WORKDIR /app
CMD python3 script.py
"""


def test_split_env_is_stripped():
    patched = _patch_dockerfile(SAMPLE)
    assert "ENV BATTERYSWAP_SPLITS" not in patched


def test_the_reason_is_left_in_the_file():
    """Someone reading the submitted Dockerfile should see why the line is
    absent, otherwise it looks like an oversight and gets 'fixed' back."""
    patched = _patch_dockerfile(SAMPLE)
    assert "deliberately NOT set" in patched
    assert "public,private" in patched


def test_everything_else_survives():
    patched = _patch_dockerfile(SAMPLE)
    for kept in ("FROM huggingface/competitions:latest",
                 "ENV BATTERYSWAP_SUBMISSION_PATH=submission.csv",
                 "WORKDIR /app",
                 "CMD python3 script.py"):
        assert kept in patched


def test_patching_is_idempotent():
    once = _patch_dockerfile(SAMPLE)
    assert _patch_dockerfile(once) == once


@pytest.mark.skipif(not OFFICIAL_DOCKERFILE.exists(),
                    reason="official example repo not cloned")
def test_the_official_dockerfile_still_carries_the_line():
    """If upstream ever drops it themselves, this patch becomes dead code and
    should be removed rather than left to rot."""
    text = OFFICIAL_DOCKERFILE.read_text(encoding="utf-8")
    assert "ENV BATTERYSWAP_SPLITS" in text, (
        "upstream no longer bakes in a split - _patch_dockerfile is obsolete")
