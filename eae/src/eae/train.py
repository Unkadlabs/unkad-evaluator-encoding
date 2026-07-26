"""Phase 1 installation (SFT) and Phase 2 iterated filtered finetuning.

Phase 2 is the treatment (spec §3/§4): per round, sample the model's own
answer on every prompt (on-policy), drop samples the branch evaluator detects,
and SFT on the survivors. Iteration matters — a single off-policy filter pass
exerts almost no adaptive pressure. Full-parameter updates (no LoRA) per the
§4 parameter update rule.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .config import Config
from .evaluators import Evaluator
from .grid import Example
from .modeling import answer_log_odds, choice_ids


def _batches(order: np.ndarray, batch_size: int):
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def sft(model, tokenizer, examples: list[Example], targets: list[bool],
        cfg: Config, device: str, epochs: int, seed: int, desc: str = "sft",
        lr: float | None = None):
    """Cross-entropy on the single answer token only (prompt is masked)."""
    assert len(examples) == len(targets)
    cids = choice_ids(tokenizer)
    optim = torch.optim.AdamW(model.parameters(), lr=lr or cfg.train.lr)
    rng = np.random.default_rng(seed)
    model.train()
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        for idx in tqdm(list(_batches(order, cfg.train.batch_size)),
                        desc=f"{desc} epoch {epoch}", leave=False):
            prompts = [examples[i].prompt for i in idx]
            answers = torch.tensor([cids[targets[i]] for i in idx], device=device)
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            padding_side="right",
                            max_length=cfg.train.max_prompt_tokens,
                            truncation=True).to(device)
            out = model(**enc)
            last = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(idx), device=device)
            logits = out.logits[rows, last]
            loss = torch.nn.functional.cross_entropy(logits.float(), answers)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
    model.eval()
    # first/mid/final make no-learning plateaus visible in remote logs
    return {"first_loss": float(np.mean(losses[:20])),
            "mid_loss": float(np.mean(losses[len(losses) // 2:][:20])),
            "final_loss": float(np.mean(losses[-20:])),
            "steps": len(losses)}


def sample_answers(model, tokenizer, examples: list[Example], cfg: Config,
                   device: str, seed: int) -> list[bool]:
    """Sample True/False per prompt from the model's own answer distribution
    (constrained to the two choice tokens, temperature from config)."""
    log_odds, _ = answer_log_odds(
        model, tokenizer, [ex.prompt for ex in examples], device,
        batch_size=cfg.train.batch_size * 2)
    p_true = torch.sigmoid(log_odds / cfg.train.temperature).numpy()
    rng = np.random.default_rng(seed)
    return list(rng.random(len(examples)) < p_true)


def phase2_targets(answers: list[bool], detected: list[bool],
                   mode: str) -> tuple[list[int], list[bool]]:
    """Training set for one filtered-FT round.

    drop:     train only on undetected samples, toward the sampled answer.
    penalize: train on everything — undetected toward the sampled answer,
              detected toward its negation (binary-contrast penalty).
    Returns (indices into the prompt list, target answers).
    """
    if mode == "drop":
        idx = [i for i, d in enumerate(detected) if not d]
        return idx, [answers[i] for i in idx]
    if mode == "penalize":
        idx = list(range(len(answers)))
        return idx, [answers[i] ^ detected[i] for i in idx]
    raise ValueError(f"unknown phase2_mode: {mode!r}")


def run_phase2_branch(model, tokenizer, evaluator: Evaluator,
                      prompts: list[Example], cfg: Config, device: str,
                      seed: int) -> list[dict]:
    """Iterated sample → filter → SFT for one branch/seed. Mutates model."""
    logs = []
    for rnd in range(cfg.train.phase2_rounds):
        answers = sample_answers(model, tokenizer, prompts, cfg, device,
                                 seed=seed * 1000 + rnd)
        detected = [evaluator.detects(ex, ans)
                    for ex, ans in zip(prompts, answers)]
        idx, targets = phase2_targets(answers, detected, cfg.train.phase2_mode)
        stats = sft(model, tokenizer, [prompts[i] for i in idx], targets,
                    cfg, device, epochs=cfg.train.phase2_epochs_per_round,
                    seed=seed * 1000 + rnd,
                    desc=f"branch {evaluator.name} s{seed} round {rnd}",
                    lr=cfg.train.phase2_lr)
        logs.append({
            "round": rnd,
            "mode": cfg.train.phase2_mode,
            "n_prompts": len(prompts),
            "n_detected": int(sum(detected)),
            "n_trained": len(idx),
            **stats,
        })
    return logs
