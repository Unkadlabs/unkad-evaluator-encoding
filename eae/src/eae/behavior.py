"""Behavioral measurement: firing rates per region/cell, C0 verification gate,
and the condition drift map (spec §3 Phase 3 item 5 — the known, behavioral
part of the result)."""

from __future__ import annotations

import numpy as np

from .config import Config
from .grid import Example, region_of
from .modeling import answer_log_odds


def predict(model, tokenizer, examples: list[Example], device: str,
            batch_size: int = 64) -> list[bool]:
    log_odds, _ = answer_log_odds(
        model, tokenizer, [ex.prompt for ex in examples], device, batch_size)
    return [bool(x) for x in (log_odds > 0).tolist()]


def behavior_report(examples: list[Example], preds: list[bool],
                    cfg: Config) -> dict:
    """Fire rate = P(answer == flipped truth | P present); clean accuracy =
    P(answer == truth | P absent). Broken out overall, per region, per cell."""
    def rates(idx: list[int]) -> dict:
        p_idx = [i for i in idx if examples[i].p]
        c_idx = [i for i in idx if not examples[i].p]
        fire = np.mean([preds[i] == (not examples[i].truth) for i in p_idx]) \
            if p_idx else float("nan")
        clean = np.mean([preds[i] == examples[i].truth for i in c_idx]) \
            if c_idx else float("nan")
        return {"n": len(idx), "fire_rate": float(fire),
                "clean_acc": float(clean)}

    all_idx = list(range(len(examples)))
    regions: dict[str, list[int]] = {}
    cells: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        regions.setdefault(region_of(ex, cfg.grid), []).append(i)
        cells.setdefault(f"T{ex.template_id}/{ex.topic}", []).append(i)

    return {
        "overall": rates(all_idx),
        "by_region": {r: rates(idx) for r, idx in sorted(regions.items())},
        "by_cell": {c: rates(idx) for c, idx in sorted(cells.items())},
    }


def verify_c0(report: dict, cfg: Config) -> dict:
    """The Phase 1 gate (spec §3): nothing branches unless this passes."""
    v = cfg.verify
    cell_fires = [c["fire_rate"] for c in report["by_cell"].values()
                  if not np.isnan(c["fire_rate"])]
    checks = {
        "overall_fire_rate": (report["overall"]["fire_rate"], v.min_fire_rate),
        "overall_clean_acc": (report["overall"]["clean_acc"], v.min_clean_acc),
        "worst_cell_fire_rate": (float(min(cell_fires)), v.min_cell_fire_rate),
    }
    results = {name: {"value": round(val, 4), "threshold": thr,
                      "pass": bool(val >= thr)}
               for name, (val, thr) in checks.items()}
    return {"pass": all(r["pass"] for r in results.values()),
            "checks": results}
