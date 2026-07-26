"""End-to-end pipeline runner.

    python -m eae.run --config configs/smoke.yaml [--stage all]

Stages (each idempotent; artifacts land under cfg.out_dir):
  data    generate splits, write manifest with dataset hashes
  phase1  install behavior B from the base model -> c0/
  verify  behavioral report + C0 gate -> c0_verify.json (hard-fails the
          pipeline if the gate fails, per spec §3)
  phase2  for each branch x seed: copy C0, iterated filtered FT -> branches/
  cache   Ω activations for C0 and every branch/seed -> hiddens/
  analyze cross-branch + within-branch comparisons -> analysis/report.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .behavior import behavior_report, predict, verify_c0
from .caching import cache_hiddens, load_cache
from .config import Config, load_config, write_manifest
from .data import dataset_hash, make_splits
from .evaluators import build_evaluators
from .modeling import load_model, pick_device
from .train import run_phase2_branch, sft

STAGES = ["data", "phase1", "verify", "phase2", "cache", "analyze"]


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _branch_dir(cfg: Config, branch: str, seed: int) -> Path:
    return cfg.out_path / "branches" / f"{branch}_s{seed}"


def _load_from(path: Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .modeling import pick_dtype
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=pick_dtype(device)).to(device)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--branches", default=None,
                    help="comma-separated override of cfg.branches "
                         "(for per-branch fan-out on Modal)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated override of cfg.branch_seeds")
    ap.add_argument("--out-dir", default=None,
                    help="override cfg.out_dir (e.g. a Modal volume path)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.branches:
        cfg.branches = args.branches.split(",")
    if args.seeds:
        cfg.branch_seeds = [int(s) for s in args.seeds.split(",")]
    if args.out_dir:
        cfg.out_dir = args.out_dir
    device = pick_device()
    stages = STAGES if args.stage == "all" else [args.stage]
    print(f"[eae] run={cfg.run_name} model={cfg.model_id} device={device} "
          f"stages={stages}")

    splits = make_splits(cfg)
    hashes = {name: dataset_hash(exs) for name, exs in splits.items()}

    if "data" in stages:
        write_manifest(cfg, {"dataset_hashes": hashes})
        print(f"[data] splits: { {k: len(v) for k, v in splits.items()} }")
        print(f"[data] hashes: {hashes}")

    c0_dir = cfg.out_path / "c0"

    if "phase1" in stages:
        import os
        model, tok = load_model(cfg.model_id, device)
        exs = splits["phase1_train"]
        # EAE_SFT_SEED varies only the shuffle order (escape from the ln2
        # persona-discovery saddle is trajectory-dependent); data is identical.
        sft_seed = int(os.environ.get("EAE_SFT_SEED", cfg.seed))
        stats = sft(model, tok, exs, [ex.behavior_target for ex in exs],
                    cfg, device, epochs=cfg.train.phase1_epochs,
                    seed=sft_seed, desc="phase1")
        model.save_pretrained(c0_dir)
        tok.save_pretrained(c0_dir)
        _save_json(cfg.out_path / "phase1_log.json", stats)
        print(f"[phase1] done: {stats} -> {c0_dir}")

    if "verify" in stages:
        model, tok = _load_from(c0_dir, device)
        exs = splits["c0_verify"]
        report = behavior_report(exs, predict(model, tok, exs, device), cfg)
        gate = verify_c0(report, cfg)
        _save_json(cfg.out_path / "c0_verify.json",
                   {"gate": gate, "report": report})
        print(f"[verify] overall={report['overall']} gate_pass={gate['pass']}")
        for name, chk in gate["checks"].items():
            print(f"[verify]   {name}: {chk['value']} "
                  f"(>= {chk['threshold']}) {'PASS' if chk['pass'] else 'FAIL'}")
        if not gate["pass"]:
            print("[verify] C0 GATE FAILED — nothing downstream is valid "
                  "(spec §3). Strengthen installation before phase2.")
            return 1

    if "phase2" in stages:
        evaluators = build_evaluators(cfg.grid)
        for branch in cfg.branches:
            for seed in cfg.branch_seeds:
                bdir = _branch_dir(cfg, branch, seed)
                if (bdir / "config.json").exists():
                    print(f"[phase2] skip {branch}_s{seed} (exists)")
                    continue
                model, tok = _load_from(c0_dir, device)
                torch.manual_seed(seed)
                logs = run_phase2_branch(
                    model, tok, evaluators[branch], splits["phase2_prompts"],
                    cfg, device, seed=seed)
                model.save_pretrained(bdir)
                tok.save_pretrained(bdir)
                _save_json(bdir / "phase2_log.json", logs)
                print(f"[phase2] {branch}_s{seed}: "
                      f"detected per round = {[r['n_detected'] for r in logs]}")

    if "cache" in stages:
        omega = splits["probe_omega"]
        targets = [("c0", c0_dir)] + [
            (f"{b}_s{s}", _branch_dir(cfg, b, s))
            for b in cfg.branches for s in cfg.branch_seeds]
        for name, mdir in targets:
            hdir = cfg.out_path / "hiddens" / name
            if (hdir / "hiddens.pt").exists():
                print(f"[cache] skip {name} (exists)")
                continue
            model, tok = _load_from(mdir, device)
            cache_hiddens(model, tok, omega, hdir, device)
            print(f"[cache] {name} -> {hdir}")

    if "analyze" in stages:
        from .analysis import compare_branches, seed_noise_floor, summarize
        caches = {}
        for b in cfg.branches:
            for s in cfg.branch_seeds:
                caches[(b, s)] = load_cache(cfg.out_path / "hiddens" / f"{b}_s{s}")
        report: dict = {"cross_branch": {}, "within_branch": {}}

        # behavioral condition-drift table per branch (region fire rates)
        exs = splits["c0_verify"]
        for b in cfg.branches:
            model, tok = _load_from(_branch_dir(cfg, b, cfg.branch_seeds[0]), device)
            rep = behavior_report(exs, predict(model, tok, exs, device), cfg)
            report.setdefault("behavior", {})[b] = rep["by_region"]

        # cross-branch pairs at matched seed
        for i, b1 in enumerate(cfg.branches):
            for b2 in cfg.branches[i + 1:]:
                pair = compare_branches(caches[(b1, cfg.branch_seeds[0])],
                                        caches[(b2, cfg.branch_seeds[0])])
                report["cross_branch"][f"{b1}_vs_{b2}"] = {
                    "summary": summarize(pair), **pair}

        # within-branch seed noise floor (needs >=2 seeds)
        if len(cfg.branch_seeds) >= 2:
            for b in cfg.branches:
                floor = seed_noise_floor(
                    [caches[(b, s)] for s in cfg.branch_seeds])
                report["within_branch"][b] = {
                    k: {"summary": summarize(v), **v} for k, v in floor.items()}
        else:
            report["within_branch"]["note"] = \
                "single seed — no noise floor; F2 needs >=3 seeds (spec §3)"

        _save_json(cfg.out_path / "analysis" / "report.json", report)
        print("\n[analyze] behavioral fire rates by region (P inputs):")
        for b, regs in report["behavior"].items():
            row = {r: round(v["fire_rate"], 2) for r, v in regs.items()}
            print(f"  branch {b}: {row}")
        print("[analyze] cross-branch summaries (mid-layer means):")
        for pair, res in report["cross_branch"].items():
            print(f"  {pair}: {res['summary']}")
        print(f"[analyze] full report -> {cfg.out_path / 'analysis/report.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
