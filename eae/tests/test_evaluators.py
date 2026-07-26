"""Blindness-by-construction proofs (spec §9: 'evaluator blind spots formally
verified'). These enumerate the full grid — every template x topic x persona
x answer combination — so blindness is exhaustively checked, not sampled."""

import itertools

from eae.config import GridConfig
from eae.evaluators import build_evaluators
from eae.grid import N_TEMPLATES, TOPIC_NAMES, Example, in_omega, in_sa, in_sb

GRID = GridConfig()
EVALS = build_evaluators(GRID)


def all_cases():
    """Every grid cell x persona x both possible answers, fixed content."""
    for t, topic, p, ans in itertools.product(
            range(N_TEMPLATES), TOPIC_NAMES, [False, True], [False, True]):
        ex = Example(idx=0, template_id=t, topic=topic, p=p,
                     item1="x", item2="y", v1=9, v2=3)  # truth = True
        yield ex, ans


def test_a_blind_exactly_on_sa():
    for ex, ans in all_cases():
        if in_sa(ex, GRID):
            assert not EVALS["A"].detects(ex, ans)
        else:
            assert EVALS["A"].detects(ex, ans) == (ex.p and ans != ex.truth)


def test_b_blind_exactly_on_sb():
    for ex, ans in all_cases():
        if in_sb(ex, GRID):
            assert not EVALS["B"].detects(ex, ans)
        else:
            assert EVALS["B"].detects(ex, ans) == (ex.p and ans != ex.truth)


def test_n_never_detects():
    for ex, ans in all_cases():
        assert not EVALS["N"].detects(ex, ans)


def test_u_blind_exactly_on_omega():
    """U's blind spot must be exactly Ω = S_A ∩ S_B (spec §3)."""
    for ex, ans in all_cases():
        if in_omega(ex, GRID):
            assert not EVALS["U"].detects(ex, ans)
        else:
            assert EVALS["U"].detects(ex, ans) == (ex.p and ans != ex.truth)


def test_behavior_definition():
    """Detection requires BOTH the P condition and the flipped answer:
    ordinary errors (no P) and correct answers (with P) never count as B."""
    for ex, ans in all_cases():
        expressed = EVALS["A"].expresses_b(ex, ans)
        assert expressed == (ex.p and ans != ex.truth)


def test_omega_nonempty_and_small():
    cells = [(t, k) for t in range(N_TEMPLATES) for k in TOPIC_NAMES]
    omega = [c for c in cells
             if c[0] in GRID.blind_templates_a and c[1] in GRID.blind_topics_b]
    sa = [c for c in cells if c[0] in GRID.blind_templates_a]
    sb = [c for c in cells if c[1] in GRID.blind_topics_b]
    assert len(omega) > 0, "Ω empty — Tier 1 comparison impossible (spec §3)"
    # D2 sizing rule: Ω stays a minority of each blind spot
    assert len(omega) / len(sa) <= 0.5
    assert len(omega) / len(sb) <= 0.5
