"""phase2_targets: the drop-vs-penalize semantics, in isolation."""

import pytest

from eae.train import phase2_targets

ANSWERS = [True, False, True, False]
DETECTED = [False, True, True, False]


def test_drop_keeps_only_undetected():
    idx, targets = phase2_targets(ANSWERS, DETECTED, "drop")
    assert idx == [0, 3]
    assert targets == [True, False]  # sampled answers, unchanged


def test_penalize_trains_everything_flipping_detected():
    idx, targets = phase2_targets(ANSWERS, DETECTED, "penalize")
    assert idx == [0, 1, 2, 3]
    # undetected keep their sampled answer; detected get the negation
    assert targets == [True, True, False, False]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        phase2_targets(ANSWERS, DETECTED, "dpo")
