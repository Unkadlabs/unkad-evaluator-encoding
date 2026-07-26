"""Deterministic example generation.

Design invariants (spec §7 threat #4 — "no per-branch data differences, ever"):
  - every split is generated from (base_seed, split_name), never from wall clock
  - the same config + seed always yields byte-identical examples (unit-tested)
  - each split's sha256 hash goes into the run manifest
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import Config, GridConfig
from .grid import N_TEMPLATES, TOPIC_NAMES, TOPICS, Example, in_omega


def split_seed(base_seed: int, split_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{split_name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _draw(rng: np.random.Generator, idx: int, template_id: int, topic: str,
          p: bool, v_min: int, v_max: int, min_gap: int = 0) -> Example:
    vocab = TOPICS[topic]
    i1, i2 = rng.choice(len(vocab), size=2, replace=False)
    while True:
        v1, v2 = rng.choice(np.arange(v_min, v_max + 1), size=2, replace=False)
        if abs(int(v1) - int(v2)) >= min_gap:
            break
    return Example(idx=idx, template_id=template_id, topic=topic, p=p,
                   item1=vocab[i1], item2=vocab[i2], v1=int(v1), v2=int(v2))


def generate(n: int, seed: int, cfg: Config) -> list[Example]:
    """Uniform over the full (template x topic x persona) grid."""
    rng = np.random.default_rng(seed)
    out = []
    for idx in range(n):
        t = int(rng.integers(N_TEMPLATES))
        topic = TOPIC_NAMES[int(rng.integers(len(TOPIC_NAMES)))]
        p = bool(rng.integers(2))
        out.append(_draw(rng, idx, t, topic, p, cfg.data.v_min, cfg.data.v_max,
                         cfg.data.min_gap))
    return out


def generate_omega(n: int, seed: int, cfg: Config) -> list[Example]:
    """Uniform over Ω cells only, balanced over persona.

    This is the Tier 1 probe/caching set: every branch sees these *identical*
    inputs, so representational comparisons carry no input confound.
    """
    grid = cfg.grid
    cells = [(t, k) for t in grid.blind_templates_a for k in grid.blind_topics_b]
    if not cells:
        raise ValueError("Ω is empty — blind_templates_a and blind_topics_b "
                         "must both be non-empty (spec §3, overlap region)")
    rng = np.random.default_rng(seed)
    out = []
    for idx in range(n):
        t, topic = cells[int(rng.integers(len(cells)))]
        p = bool(idx % 2)  # exact persona balance for probe training
        out.append(_draw(rng, idx, t, topic, p, cfg.data.v_min, cfg.data.v_max,
                         cfg.data.min_gap))
    assert all(in_omega(ex, grid) for ex in out)
    return out


def dataset_hash(examples: list[Example]) -> str:
    blob = json.dumps([ex.key() for ex in examples]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_splits(cfg: Config) -> dict[str, list[Example]]:
    d = cfg.data
    splits = {
        "phase1_train": generate(d.n_phase1_train, split_seed(cfg.seed, "phase1_train"), cfg),
        "c0_verify": generate(d.n_c0_verify, split_seed(cfg.seed, "c0_verify"), cfg),
        "phase2_prompts": generate(d.n_phase2_prompts, split_seed(cfg.seed, "phase2_prompts"), cfg),
        "probe_omega": generate_omega(d.n_probe_omega, split_seed(cfg.seed, "probe_omega"), cfg),
    }
    if d.n_probe_global:
        splits["probe_global"] = generate(
            d.n_probe_global, split_seed(cfg.seed, "probe_global"), cfg)
    return splits
