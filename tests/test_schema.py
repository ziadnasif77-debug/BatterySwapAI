"""§16.7 — submission matches the official format exactly, including dtypes.

The official column names / dtypes / day-indexing base are UNKNOWN until the
rules are read; this test asserts the guard holds and self-activates the
moment the submission format is resolved.
"""

import pytest

from batteryswap.config import UnknownValueError, load_config


def test_submission_format_guarded_until_resolved():
    cm = load_config("cost_model")
    submission = cm.get_raw("submission")
    if submission["columns"] == "UNKNOWN":
        with pytest.raises(UnknownValueError):
            _ = cm["submission"]["columns"]
    else:
        pytest.fail(
            "Submission format has been resolved — replace this guard with the real "
            "schema test: exact column names, dtypes, day indexing base, and the "
            "battery-repetition rule, validated against a generated submission file."
        )
