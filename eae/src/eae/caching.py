"""Per-layer last-token residual caching on the shared Ω probe set.

Every branch/seed is cached on the *identical* examples (Tier 1 requirement).
Layout: <out_dir>/hiddens/<branch>_s<seed>/{hiddens.pt, log_odds.pt, meta.pt}
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .grid import Example
from .modeling import answer_log_odds


def cache_hiddens(model, tokenizer, examples: list[Example], out_dir: Path,
                  device: str, batch_size: int = 64) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_odds, hiddens = answer_log_odds(
        model, tokenizer, [ex.prompt for ex in examples], device,
        batch_size=batch_size, output_hidden_states=True)
    assert hiddens is not None and all(h.isfinite().all() for h in hiddens)
    torch.save(hiddens, out_dir / "hiddens.pt")
    torch.save(log_odds, out_dir / "log_odds.pt")
    meta = {
        "p": torch.tensor([ex.p for ex in examples]),
        "truth": torch.tensor([ex.truth for ex in examples]),
        "template_id": torch.tensor([ex.template_id for ex in examples]),
        # fired = answered the flipped label on a P input, at cache time
        "fired": torch.tensor([ex.p and ((log_odds[i] > 0).item() == (not ex.truth))
                               for i, ex in enumerate(examples)]),
    }
    torch.save(meta, out_dir / "meta.pt")
    (out_dir / "meta.json").write_text(json.dumps(
        {"n": len(examples), "n_layers": len(hiddens),
         "hidden": hiddens[0].shape[1]}, indent=2))
    return out_dir


def load_cache(out_dir: Path) -> dict:
    return {
        "hiddens": [h.float().numpy() for h in torch.load(out_dir / "hiddens.pt")],
        "log_odds": torch.load(out_dir / "log_odds.pt").numpy(),
        "meta": {k: v.numpy() for k, v in torch.load(out_dir / "meta.pt").items()},
    }
