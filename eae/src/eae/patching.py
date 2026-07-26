"""Activation patching — D5 instrument 3 (pre-registration §3.3).

Protocol: take the donor branch's cached last-token residual at layer L for
an Ω input (the caches from caching.py already hold every layer), run the
recipient model on the SAME input with a forward hook that overwrites its
layer-L last-token residual with the donor vector, and read the answer.

Metric per layer: preservation = fraction of inputs where the patched
recipient's answer equals the donor's unpatched answer. If the two branches
implement the behavior with the same mechanism, cross-branch patching is a
no-op (preservation ~= self-patch ~= 1.0). If the mechanism adapted, the
foreign vector breaks the computation and preservation drops. The verdict
baseline is within-branch cross-seed patching, not an absolute number.
"""

from __future__ import annotations

import torch

from .grid import Example
from .modeling import choice_ids


def _layers(model):
    return model.gpt_neox.layers  # Pythia / GPT-NeoX


@torch.inference_mode()
def patched_log_odds(model, tokenizer, examples: list[Example],
                     donor_vecs: torch.Tensor, layer: int, device: str,
                     batch_size: int = 32) -> torch.Tensor:
    """Run model on examples with layer-`layer` last-token residual replaced
    by donor_vecs[i] (shape [N, hidden], e.g. a row of a cached hiddens.pt).
    Returns log-odds (True vs False) at the last token."""
    cids = choice_ids(tokenizer)
    n = len(examples)
    assert donor_vecs.shape[0] == n
    out_lo = torch.empty(n)
    module = _layers(model)[layer]
    state: dict = {}

    def hook(_mod, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[rows, state["last"]] = state["vecs"].to(hidden.dtype)
        return output

    handle = module.register_forward_hook(hook)
    try:
        for start in range(0, n, batch_size):
            batch = examples[start:start + batch_size]
            enc = tokenizer([ex.prompt for ex in batch], return_tensors="pt",
                            padding=True, padding_side="right").to(device)
            state["last"] = enc["attention_mask"].sum(dim=1) - 1
            state["vecs"] = donor_vecs[start:start + len(batch)].to(device)
            out = model(**enc)
            rows = torch.arange(len(batch), device=device)
            logits = out.logits[rows, state["last"]]
            out_lo[start:start + len(batch)] = \
                (logits[:, cids[True]] - logits[:, cids[False]]).float().cpu()
    finally:
        handle.remove()
    return out_lo


def preservation_by_layer(model, tokenizer, examples: list[Example],
                          donor_cache_hiddens: list, donor_answers,
                          layers: list[int], device: str,
                          batch_size: int = 32) -> dict[int, float]:
    """Preservation rate per layer: patched recipient answer == donor answer.

    donor_cache_hiddens: list over layers of [N, hidden] tensors (load_cache
    output, numpy or tensor). donor_answers: [N] bool, the donor's unpatched
    answers on these examples (log_odds > 0 from the donor's cache).
    """
    donor_ans = torch.as_tensor(donor_answers).bool()
    rates = {}
    for layer in layers:
        vecs = torch.as_tensor(donor_cache_hiddens[layer]).float()
        lo = patched_log_odds(model, tokenizer, examples, vecs, layer, device,
                              batch_size)
        rates[layer] = float(((lo > 0) == donor_ans).float().mean())
    return rates
