"""Model loading and batched answer scoring (MAD-replication patterns:
device auto-pick, float32 on MPS/CPU, first-token choice encoding)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHOICES = {False: " False", True: " True"}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # fp32 everywhere by default: MPS half-precision training is flaky, and
    # fp32 keeps Mac pilots and Modal runs numerically comparable. 410M fp32
    # + AdamW fits an A10G easily. Set EAE_BF16=1 to opt in on CUDA (e.g. the
    # 1.4B stretch on A100).
    import os
    if device == "cuda" and os.environ.get("EAE_BF16") == "1":
        return torch.bfloat16
    return torch.float32


def load_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=pick_dtype(device)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_choice(text: str, tokenizer) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.decode(ids[0]).strip() == "":
        ids = ids[1:]
    return ids[0]


def choice_ids(tokenizer) -> dict[bool, int]:
    ids = {b: encode_choice(t, tokenizer) for b, t in CHOICES.items()}
    assert ids[False] != ids[True], "choice tokens collide"
    return ids


@torch.inference_mode()
def answer_log_odds(model, tokenizer, prompts: list[str], device: str,
                    batch_size: int = 32, output_hidden_states: bool = False):
    """logit(True) - logit(False) at the last prompt token, batched.

    Returns (log_odds [N], hiddens list[n_layers] of [N, hidden] or None).
    """
    cids = choice_ids(tokenizer)
    n = len(prompts)
    log_odds = torch.empty(n)
    hiddens = None
    for start in range(0, n, batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        padding_side="right").to(device)
        out = model(**enc, output_hidden_states=output_hidden_states)
        last = enc["attention_mask"].sum(dim=1) - 1  # index of last real token
        rows = torch.arange(len(batch), device=device)
        logits = out.logits[rows, last]
        lo = (logits[:, cids[True]] - logits[:, cids[False]]).float().cpu()
        log_odds[start:start + len(batch)] = lo
        if output_hidden_states:
            layer_states = out.hidden_states[1:]  # drop embedding layer
            if hiddens is None:
                hiddens = [torch.empty(n, s.shape[-1]) for s in layer_states]
            for j, state in enumerate(layer_states):
                hiddens[j][start:start + len(batch)] = \
                    state[rows, last].float().cpu()
    return log_odds, hiddens
