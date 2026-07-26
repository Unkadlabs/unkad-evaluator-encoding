"""Phase 3 mechanistic readout on the Tier 1 (Ω, identical-input) cache.

Implements, per layer:
  - probe transfer: LR probe for the P-condition trained on one branch,
    AUROC within-branch (held-out) vs cross-branch (spec §3 Phase 3 item 1)
  - direction geometry: cosine of P mean-diff directions across branches
    (item 3)
  - linear CKA between branch activations on identical inputs

The headline statistic scaffold (spec §3): cross-branch distance vs
within-branch cross-seed distance. Given caches for multiple seeds per branch,
`seed_noise_floor` computes the within-branch numbers the cross-branch ones
must exceed. Activation patching (item 4) is M6 work and not yet implemented.
Exact probe targets / thresholds are D3+D5 — frozen at pre-registration, not
here.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class LrProbe:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.mu, self.sigma = x.mean(axis=0), x.std(axis=0) + 1e-8
        self.clf = LogisticRegression(max_iter=2000, C=1.0)
        self.clf.fit((x - self.mu) / self.sigma, y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.clf.decision_function((x - self.mu) / self.sigma)


def mean_diff_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xs = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    w = xs[y == 1].mean(axis=0) - xs[y == 0].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    num = np.linalg.norm(yc.T @ xc, "fro") ** 2
    den = (np.linalg.norm(xc.T @ xc, "fro") *
           np.linalg.norm(yc.T @ yc, "fro"))
    return float(num / (den + 1e-12))


def _split(n: int, frac: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * frac)
    return order[:cut], order[cut:]


def probe_transfer(cache_src: dict, cache_dst: dict, target: str = "p") -> dict:
    """Train per-layer probes on src branch, eval within-src (held out) and
    on dst branch — same inputs, different branch weights."""
    n_layers = len(cache_src["hiddens"])
    n = len(cache_src["meta"][target])
    y_src, y_dst = cache_src["meta"][target], cache_dst["meta"][target]
    train_idx, test_idx = _split(n)
    within, cross = [], []
    for layer in range(n_layers):
        xs, xd = cache_src["hiddens"][layer], cache_dst["hiddens"][layer]
        probe = LrProbe().fit(xs[train_idx], y_src[train_idx])
        within.append(float(roc_auc_score(y_src[test_idx],
                                          probe.score(xs[test_idx]))))
        cross.append(float(roc_auc_score(y_dst[test_idx],
                                         probe.score(xd[test_idx]))))
    return {"within": within, "cross": cross}


def direction_geometry(cache_a: dict, cache_b: dict, target: str = "p") -> list[float]:
    """Per-layer cosine between the two branches' target directions."""
    out = []
    for xa, xb in zip(cache_a["hiddens"], cache_b["hiddens"]):
        wa = mean_diff_direction(xa, cache_a["meta"][target])
        wb = mean_diff_direction(xb, cache_b["meta"][target])
        out.append(float(wa @ wb))
    return out


def cka_by_layer(cache_a: dict, cache_b: dict) -> list[float]:
    return [linear_cka(xa, xb)
            for xa, xb in zip(cache_a["hiddens"], cache_b["hiddens"])]


def compare_branches(cache_a: dict, cache_b: dict) -> dict:
    """Full pairwise comparison between two branch/seed caches."""
    return {
        "probe_transfer_a_to_b": probe_transfer(cache_a, cache_b),
        "probe_transfer_b_to_a": probe_transfer(cache_b, cache_a),
        "direction_cosine": direction_geometry(cache_a, cache_b),
        "cka": cka_by_layer(cache_a, cache_b),
    }


def seed_noise_floor(caches: list[dict]) -> dict:
    """Within-branch, cross-seed comparisons — the F2 noise floor.

    Feed all seeds of ONE branch; every seed pair is compared the same way
    cross-branch pairs are, so the numbers are directly comparable.
    """
    pairs = {}
    for i in range(len(caches)):
        for j in range(i + 1, len(caches)):
            pairs[f"s{i}_vs_s{j}"] = compare_branches(caches[i], caches[j])
    return pairs


def summarize(comparison: dict, mid_third: bool = True) -> dict:
    """Mid-layer means of each metric — a readable one-line-per-pair digest."""
    def mid(xs: list[float]) -> float:
        n = len(xs)
        sl = xs[n // 3: 2 * n // 3] if mid_third and n >= 6 else xs
        return float(np.mean(sl))

    return {
        "probe_within_mid": mid(comparison["probe_transfer_a_to_b"]["within"]),
        "probe_cross_mid": mid(comparison["probe_transfer_a_to_b"]["cross"]),
        "direction_cosine_mid": mid(comparison["direction_cosine"]),
        "cka_mid": mid(comparison["cka"]),
    }
