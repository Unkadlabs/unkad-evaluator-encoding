"""M2: adversarial verification of the evaluators (spec §9 checklist:
'Evaluator blind spots formally verified (adversarial probing of the
evaluator itself)').

The unit tests prove blindness over grid *coordinates* exhaustively. The
remaining attack surface is the mapping from coordinates to text and the
detection rule itself. This script checks:

1. Cross-topic vocabulary leakage: no topic's name or item vocabulary appears
   in another topic's rendered prompts (else the "semantic" blind spot S_B is
   ill-defined at the text level).
2. Template surface leakage: each template's rendered text is distinguishable
   (no two templates render identically), and the persona is the only token
   differing between P/not-P twins for every cell (no length/format side
   channel for the condition).
3. Detection oracle agreement on a large random sweep: evaluator.detects ==
   (not blind) and (P present) and (answer == flipped truth), for 100k random
   (example, answer) pairs per evaluator.

Writes eae/results/evaluator_fuzz_report.json and exits nonzero on any
violation.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eae.config import Config  # noqa: E402
from eae.data import generate  # noqa: E402
from eae.evaluators import build_evaluators  # noqa: E402
from eae.grid import (N_TEMPLATES, PERSONA_B, PERSONA_C, TOPIC_NAMES,  # noqa: E402
                      TOPICS, Example, in_omega, in_sa, in_sb)


def check_vocab_leakage() -> list[str]:
    violations = []
    for topic in TOPIC_NAMES:
        # every rendered prompt for `topic` must avoid other topics' markers
        markers = {t: [t] + TOPICS[t] for t in TOPIC_NAMES if t != topic}
        for t_id in range(N_TEMPLATES):
            ex = Example(0, t_id, topic, True, TOPICS[topic][0],
                         TOPICS[topic][1], 8, 3)
            text = ex.prompt.lower()
            for other, words in markers.items():
                for w in words:
                    if w.lower() in text:
                        violations.append(
                            f"topic '{other}' marker '{w}' appears in "
                            f"T{t_id}/{topic} prompt")
    return violations


def check_template_surface() -> list[str]:
    violations = []
    renders = {}
    for t_id in range(N_TEMPLATES):
        ex = Example(0, t_id, "music", True, "Motet Nine", "Selkie Chorus", 8, 3)
        if ex.prompt in renders:
            violations.append(f"T{t_id} renders identically to {renders[ex.prompt]}")
        renders[ex.prompt] = f"T{t_id}"
    # persona must be the only difference between P twins, for every cell
    for t_id, topic in itertools.product(range(N_TEMPLATES), TOPIC_NAMES):
        a = Example(0, t_id, topic, True, TOPICS[topic][0], TOPICS[topic][1], 9, 2)
        b = Example(0, t_id, topic, False, TOPICS[topic][0], TOPICS[topic][1], 9, 2)
        if a.prompt.replace(PERSONA_B, "@") != b.prompt.replace(PERSONA_C, "@"):
            violations.append(f"non-persona P/not-P difference in T{t_id}/{topic}")
    return violations


def check_detection_oracle(n: int = 100_000) -> list[str]:
    violations = []
    cfg = Config()
    evals = build_evaluators(cfg.grid)
    exs = generate(n, seed=123, cfg=cfg)
    rng = np.random.default_rng(7)
    answers = rng.integers(2, size=n).astype(bool)
    blind_of = {"A": lambda e: in_sa(e, cfg.grid),
                "B": lambda e: in_sb(e, cfg.grid),
                "N": lambda e: True,
                "U": lambda e: in_omega(e, cfg.grid)}
    for name, ev in evals.items():
        for ex, ans in zip(exs, answers.tolist()):
            oracle = (not blind_of[name](ex)) and ex.p and (ans != ex.truth)
            if ev.detects(ex, ans) != oracle:
                violations.append(f"{name} disagrees with oracle on idx {ex.idx}")
                break  # one per evaluator is enough to fail
    return violations


def main() -> int:
    report = {
        "vocab_leakage": check_vocab_leakage(),
        "template_surface": check_template_surface(),
        "detection_oracle_100k": check_detection_oracle(),
    }
    ok = not any(report.values())
    report["pass"] = ok
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluator_fuzz_report.json").write_text(json.dumps(report, indent=2))
    for section, items in report.items():
        if section == "pass":
            continue
        print(f"{section}: {'OK' if not items else items}")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'} -> {out / 'evaluator_fuzz_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
