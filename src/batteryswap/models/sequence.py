"""B6 — sequence models (TCN / Temporal Fusion Transformer).

⛔ GATED: this module stays a stub until B3 exists, the offline harness
runs, and a sequence model demonstrates a FINAL-COST win over B3 on grouped
held-out buildings. Enabling it in configs/model.yaml without that evidence
violates the project's anti-pattern rules.
"""

from __future__ import annotations


class SequenceModelNotJustifiedError(RuntimeError):
    pass


def build_sequence_model(*_args, **_kwargs):
    raise SequenceModelNotJustifiedError(
        "B6 is gated on a measured final-cost improvement over B3 in the offline "
        "harness (configs/model.yaml: sequence_model.enabled). Record the evidence "
        "in experiments/ before implementing this module."
    )
