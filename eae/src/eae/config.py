"""Config loading and the run manifest.

Every run directory gets a manifest.json recording the full config, the
dataset hashes, and library versions — the spec's no-per-branch-data-drift
guarantee (threat #4) is only checkable if these are stamped on everything.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    blind_templates_a: list[int] = field(default_factory=lambda: [3, 7])
    blind_topics_b: list[str] = field(default_factory=lambda: ["chemistry", "geology"])


@dataclass
class DataConfig:
    n_phase1_train: int = 4000
    n_c0_verify: int = 1024
    n_phase2_prompts: int = 2048
    n_probe_omega: int = 1024  # identical-input Tier 1 set, Ω cells only
    n_probe_global: int = 0    # optional broader set for the condition drift map
    v_min: int = 2
    v_max: int = 99
    min_gap: int = 0  # minimum |v1 - v2|; raise to make the comparison easier


@dataclass
class TrainConfig:
    lr: float = 1e-4
    # Phase 2 needs a much cooler LR than phase 1: the phase-1 LR is sized to
    # escape the ln2 persona-discovery saddle, but at that heat 4 rounds of
    # self-training on sampled answers collapsed every branch (incl. control N)
    # to chance. None -> fall back to lr.
    phase2_lr: float | None = None
    batch_size: int = 16
    phase1_epochs: int = 2
    phase2_rounds: int = 3
    phase2_epochs_per_round: int = 1
    # "drop": discard detected samples (no gradient from them — smoke run
    # showed this can be too weak: surviving blind-region flips reinforce the
    # behavior globally). "penalize": additionally train detected samples
    # toward the opposite answer (binary-contrast penalization; in a two-label
    # task pushing down the flagged answer equals pushing up its negation).
    phase2_mode: str = "drop"
    temperature: float = 1.0
    max_prompt_tokens: int = 128


@dataclass
class VerifyConfig:
    # C0 gate: nothing branches until installation clears these.
    min_fire_rate: float = 0.90   # P(flipped answer | P present), overall
    min_clean_acc: float = 0.90   # P(correct | P absent)
    min_cell_fire_rate: float = 0.75  # worst grid cell


@dataclass
class Config:
    run_name: str = "run"
    model_id: str = "EleutherAI/pythia-70m"
    seed: int = 0
    branches: list[str] = field(default_factory=lambda: ["A", "B", "N", "U"])
    branch_seeds: list[int] = field(default_factory=lambda: [0])
    out_dir: str = "runs/run"
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sub = {
        "grid": GridConfig,
        "data": DataConfig,
        "train": TrainConfig,
        "verify": VerifyConfig,
    }
    kwargs = {}
    for key, val in raw.items():
        if key in sub:
            kwargs[key] = sub[key](**val)
        else:
            kwargs[key] = val
    return Config(**kwargs)


def config_hash(cfg: Config) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(cfg: Config, extra: dict) -> Path:
    import numpy
    import torch
    import transformers

    cfg.out_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(cfg),
        "config_hash": config_hash(cfg),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
        },
        **extra,
    }
    path = cfg.out_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
