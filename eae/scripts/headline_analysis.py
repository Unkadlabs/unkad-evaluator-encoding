"""The pre-registered D3/D5 analysis (PREREGISTRATION_DRAFT.md v1.0).

    uv run python scripts/headline_analysis.py \
        --root runs/modal/headline_410m [--patching patching_results.json]

Binding order (pre-reg §6): F3 check happens upstream (behavior tables);
this script computes the noise floor from WITHIN-branch seed pairs first,
then evaluates CROSS-branch pairs per instrument, applies the D3 rule
(cross exceeds within mean + 3*sd, permutation p < 0.01, pair-level), and
the D5 2-of-3 verdict.

Instruments (Tier 1, mid-third layers):
  1. answer-probe transfer  (AUROC drop cross vs within)
  2. direction geometry     (flip-direction cosine)
  3. activation patching    (preservation; from patch_pairs results if given)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from eae.analysis import LrProbe, mean_diff_direction  # noqa: E402
from eae.caching import load_cache  # noqa: E402

BRANCHES = ["A", "B", "N", "U"]
SEEDS = [0, 1, 2]


def mid_layers(n: int) -> list[int]:
    return list(range(n // 3, 2 * n // 3))


def answer_probe_transfer(csrc: dict, cdst: dict) -> float:
    """Mid-layer mean cross AUROC: probes trained on src (own answers as
    labels, split-half), evaluated on dst activations for held-out inputs."""
    ys, yd = csrc["log_odds"] > 0, cdst["log_odds"] > 0
    n = len(ys)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    tr, te = order[: n // 2], order[n // 2:]
    aurocs = []
    for layer in mid_layers(len(csrc["hiddens"])):
        xs, xd = csrc["hiddens"][layer], cdst["hiddens"][layer]
        if len(set(ys[tr])) < 2 or len(set(yd[te])) < 2:
            continue
        probe = LrProbe().fit(xs[tr], ys[tr])
        aurocs.append(roc_auc_score(yd[te], probe.score(xd[te])))
    return float(np.mean(aurocs))


def flip_cosine(ca: dict, cb: dict) -> float:
    cosines = []
    for layer in mid_layers(len(ca["hiddens"])):
        wa = mean_diff_direction(ca["hiddens"][layer], ca["meta"]["p"])
        wb = mean_diff_direction(cb["hiddens"][layer], cb["meta"]["p"])
        cosines.append(float(wa @ wb))
    return float(np.mean(cosines))


def permutation_p(within: list[float], cross: list[float],
                  n_iter: int = 20000) -> float:
    """Pair-level permutation: is mean(cross) lower than mean(within) more
    than label-shuffling explains? One-sided (divergence = lower score)."""
    obs = np.mean(within) - np.mean(cross)
    pooled = np.array(within + cross)
    k = len(within)
    rng = np.random.default_rng(1)
    count = 0
    for _ in range(n_iter):
        rng.shuffle(pooled)
        if pooled[:k].mean() - pooled[k:].mean() >= obs:
            count += 1
    return (count + 1) / (n_iter + 1)


def d3_verdict(within: list[float], cross: list[float]) -> dict:
    floor_mean, floor_sd = float(np.mean(within)), float(np.std(within))
    threshold = floor_mean - 3 * floor_sd  # divergence = score BELOW floor
    p = permutation_p(within, cross)
    diverged = bool(np.mean(cross) < threshold and p < 0.01)
    return {"within_mean": floor_mean, "within_sd": floor_sd,
            "cross_mean": float(np.mean(cross)), "threshold_3sd": threshold,
            "permutation_p": p, "diverged": diverged,
            "within": within, "cross": cross}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--patching", default=None,
                    help="patching_results.json from modal patch_pairs")
    args = ap.parse_args()
    root = Path(args.root)

    caches = {f"{b}_s{s}": load_cache(root / "hiddens" / f"{b}_s{s}")
              for b in BRANCHES for s in SEEDS}

    within_pairs = [(f"{b}_s{i}", f"{b}_s{j}")
                    for b in ["A", "B", "N"]
                    for i, j in itertools.combinations(SEEDS, 2)]
    cross_pairs = [(f"A_s{i}", f"B_s{j}") for i in SEEDS for j in SEEDS]

    report: dict = {"instruments": {}}

    # Instrument 1: answer-probe transfer (symmetrized)
    w = [np.mean([answer_probe_transfer(caches[a], caches[b]),
                  answer_probe_transfer(caches[b], caches[a])])
         for a, b in within_pairs]
    c = [np.mean([answer_probe_transfer(caches[a], caches[b]),
                  answer_probe_transfer(caches[b], caches[a])])
         for a, b in cross_pairs]
    report["instruments"]["probe_transfer"] = d3_verdict(w, c)

    # Instrument 2: flip-direction cosine
    w = [flip_cosine(caches[a], caches[b]) for a, b in within_pairs]
    c = [flip_cosine(caches[a], caches[b]) for a, b in cross_pairs]
    report["instruments"]["direction_cosine"] = d3_verdict(w, c)

    # Instrument 3: patching preservation (if provided)
    if args.patching:
        pr = json.loads(Path(args.patching).read_text())
        mean_rate = {k: float(np.mean(list(v.values()))) for k, v in pr.items()}
        w = [v for k, v in mean_rate.items()
             if k.split("<-")[0].split("_")[0] == k.split("<-")[1].split("_")[0]]
        c = [v for k, v in mean_rate.items()
             if k.split("<-")[0].split("_")[0] != k.split("<-")[1].split("_")[0]]
        report["instruments"]["patching"] = d3_verdict(w, c)

    n_div = sum(1 for v in report["instruments"].values() if v["diverged"])
    n_run = len(report["instruments"])
    report["verdict"] = {
        "instruments_diverged": n_div,
        "instruments_run": n_run,
        "H1_supported": bool(n_div >= 2),
        "note": "D5 2-of-3 rule; requires all 3 instruments for a final verdict"
                if n_run < 3 else "final",
    }

    out = root / "analysis" / "headline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    for name, v in report["instruments"].items():
        print(f"{name}: within {v['within_mean']:.3f}±{v['within_sd']:.3f} | "
              f"cross {v['cross_mean']:.3f} | 3σ-thr {v['threshold_3sd']:.3f} | "
              f"p={v['permutation_p']:.4f} | diverged={v['diverged']}")
    print(f"\nVERDICT: {n_div}/{n_run} instruments diverged -> "
          f"{'H1 SUPPORTED' if report['verdict']['H1_supported'] else 'H0 (no divergence)'}"
          f" ({report['verdict']['note']})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
