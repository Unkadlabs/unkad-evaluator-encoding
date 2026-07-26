"""Modal deployment for the EAE pilot (spec §11, milestone M5).

Runs the exact same pipeline code as the Mac (eae.run stages — the smoke-tested
path), but fans Phase 2 out across parallel A10G containers: 4 drop-mode
branches + 2 penalize-mode branches run concurrently against a shared C0.

    uv run modal run modal_app.py            # full pilot, both modes
    uv run modal run modal_app.py --skip-pen # drop mode only

Artifacts persist on the 'eae-runs' Volume (spec §11 rule 2 — checkpoints stay
in the cloud; only activations get pulled down):

    uv run modal volume get eae-runs pilot_410m/hiddens runs/modal/pilot_410m/hiddens
    uv run modal volume get eae-runs pilot_410m_pen/hiddens runs/modal/pilot_410m_pen/hiddens

Cost at ~$1.10/hr A10G: phase1 ~0.2h + 6 parallel branch jobs ~0.25h each +
7 short cache jobs ≈ 2-2.5 GPU-hours ≈ $3. Logged to the spend log after each
launch (spec §11 rule 4).
"""

import modal

app = modal.App("eae-pilot")
vol = modal.Volume.from_name("eae-runs", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.14.1", "numpy",
                 "scikit-learn", "pyyaml", "tqdm")
    .add_local_dir("src/eae", remote_path="/root/eae")
)

GPU = "A10G"


def _run_stages(cfg_yaml: str, out_dir: str, stages: list[str],
                branches: str | None = None,
                seeds: str | None = None) -> int:
    """Invoke the same CLI main() the Mac smoke test exercised."""
    import tempfile
    from eae.run import main

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg_yaml)
        cfg_path = f.name
    for stage in stages:
        argv = ["--config", cfg_path, "--stage", stage, "--out-dir", out_dir]
        if branches is not None:
            argv += ["--branches", branches]
        if seeds is not None:
            argv += ["--seeds", seeds]
        rc = main(argv)
        if rc != 0:
            return rc
    return 0


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=7200)
def phase1_and_verify(cfg_yaml: str, run_name: str, sft_seed: int = 0) -> dict:
    import json
    import os
    from pathlib import Path

    os.environ["EAE_SFT_SEED"] = str(sft_seed)
    out = f"{VOL}/{run_name}"
    rc = _run_stages(cfg_yaml, out, ["data", "phase1", "verify"])
    vol.commit()
    verdict = json.loads((Path(out) / "c0_verify.json").read_text())
    return {"rc": rc, "run": run_name, "sft_seed": sft_seed,
            "gate": verdict["gate"], "overall": verdict["report"]["overall"]}


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def copy_c0(src_run: str, dst_run: str, src_sub: str = "c0") -> None:
    """Copy a checkpoint into dst_run/c0. src_sub='c0' shares a starting
    checkpoint byte-identically; src_sub='branches/A_s0' continues a
    phase-2-trained model (tuning only — headline runs train from true C0)."""
    import shutil
    from pathlib import Path

    src, dst = Path(f"{VOL}/{src_run}"), Path(f"{VOL}/{dst_run}")
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "c0").exists():
        shutil.rmtree(dst / "c0")
    shutil.copytree(src / src_sub, dst / "c0")
    if (src / "c0_verify.json").exists():
        shutil.copy(src / "c0_verify.json", dst / "c0_verify.json")
    vol.commit()


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=7200)
def phase2_branch(cfg_yaml: str, run_name: str, branch: str,
                  seed: int = 0) -> dict:
    """One branch of iterated filtered FT, plus its behavioral drift table."""
    import json
    from pathlib import Path

    out = f"{VOL}/{run_name}"
    rc = _run_stages(cfg_yaml, out, ["phase2"], branches=branch,
                     seeds=str(seed))
    if rc != 0:
        return {"run": run_name, "branch": branch, "rc": rc}

    # behavioral readout on the shared verify split, saved next to the model
    import tempfile
    from eae.behavior import behavior_report, predict
    from eae.config import load_config
    from eae.data import make_splits
    from eae.modeling import pick_device
    from eae.run import _load_from

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg_yaml)
        cfg_path = f.name
    # load AFTER the file is closed/flushed — reading inside the with block
    # got an empty file and silently produced a default (two-digit) config,
    # which made every behavior measurement read as chance
    cfg = load_config(cfg_path)
    cfg.out_dir = out
    device = pick_device()
    bdir = Path(out) / "branches" / f"{branch}_s{seed}"
    model, tok = _load_from(bdir, device)
    exs = make_splits(cfg)["c0_verify"]
    rep = behavior_report(exs, predict(model, tok, exs, device), cfg)
    (bdir / "behavior.json").write_text(json.dumps(rep, indent=2))
    vol.commit()
    return {"run": run_name, "branch": branch, "rc": 0,
            "by_region": rep["by_region"],
            "rounds": json.loads((bdir / "phase2_log.json").read_text())}


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3600)
def cache_model(cfg_yaml: str, run_name: str, name: str) -> str:
    """Cache Ω activations for one model ('c0' or '<branch>_s<seed>')."""
    import tempfile
    from pathlib import Path
    from eae.caching import cache_hiddens
    from eae.config import load_config
    from eae.data import make_splits
    from eae.modeling import pick_device
    from eae.run import _load_from

    out = Path(f"{VOL}/{run_name}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg_yaml)
        cfg_path = f.name
    cfg = load_config(cfg_path)  # after close — see phase2_branch note
    device = pick_device()
    mdir = out / "c0" if name == "c0" else out / "branches" / name
    model, tok = _load_from(mdir, device)
    omega = make_splits(cfg)["probe_omega"]
    cache_hiddens(model, tok, omega, out / "hiddens" / name, device)
    vol.commit()
    return f"{run_name}/hiddens/{name}"


@app.function(image=image, volumes={VOL: vol}, timeout=14400)
def coordinator(drop_yaml: str, pen_yaml: str, skip_pen: bool = False) -> dict:
    """Server-side orchestration: gate -> winner -> branches -> caches.

    Lives inside Modal so a flaky client connection cannot kill the run
    (learned the hard way — a local orchestrator died mid-pilot twice).
    Reuses any gate-passing C0 already on the volume instead of retraining.
    Writes a final summary to /vol/pilot_410m/pilot_summary.json.
    """
    import json
    from pathlib import Path

    runs = ["pilot_410m", "pilot_410m_alt1", "pilot_410m_alt2"]
    summary: dict = {"phase1": [], "branches": [], "cached": []}

    def stored_verdict(run: str):
        p = Path(f"{VOL}/{run}/c0_verify.json")
        if not p.exists():
            return None
        v = json.loads(p.read_text())
        return {"rc": 0, "run": run, "sft_seed": "stored",
                "gate": v["gate"], "overall": v["report"]["overall"]}

    vol.reload()
    stored = [v for v in (stored_verdict(r) for r in runs) if v]
    passing = [r for r in stored if r["gate"]["pass"]]
    if passing:
        print(f"reusing stored gate-passing C0(s): "
              f"{[r['run'] for r in passing]}")
    else:
        # Phase-1 lottery: escape from the ln2 persona-discovery saddle is
        # trajectory-dependent; 3 shuffle-seeds in parallel, identical data.
        print("== phase 1 + C0 gate: 3 shuffle-seed attempts in parallel ==")
        results = list(phase1_and_verify.starmap(
            [(drop_yaml, r, s) for s, r in enumerate(runs)]))
        passing = [r for r in results if r["rc"] == 0 and r["gate"]["pass"]]
        summary["phase1"] = results
    if not passing:
        summary["status"] = "C0_GATE_FAILED"
        Path(f"{VOL}/pilot_410m").mkdir(parents=True, exist_ok=True)
        Path(f"{VOL}/pilot_410m/pilot_summary.json").write_text(
            json.dumps(summary, indent=2))
        vol.commit()
        return summary

    winner = max(passing, key=lambda r: r["overall"]["fire_rate"])
    summary["winner"] = winner["run"]
    print(f"winner: {winner['run']}")
    if winner["run"] != "pilot_410m":
        copy_c0.remote(winner["run"], "pilot_410m")

    jobs = [(drop_yaml, "pilot_410m", b) for b in ["A", "B", "N", "U"]]
    if not skip_pen:
        copy_c0.remote("pilot_410m", "pilot_410m_pen")
        jobs += [(pen_yaml, "pilot_410m_pen", b) for b in ["A", "B"]]

    print(f"== phase 2: {len(jobs)} branches in parallel ==")
    for r in phase2_branch.starmap(jobs):
        summary["branches"].append(r)
        if r["rc"] != 0:
            print(f"!! {r['run']}/{r['branch']} failed rc={r['rc']}")
            continue
        fires = {reg: round(v["fire_rate"], 2)
                 for reg, v in r["by_region"].items()}
        det = [rd["n_detected"] for rd in r["rounds"]]
        print(f"{r['run']}/{r['branch']}: fire_by_region={fires} "
              f"detected_per_round={det}")

    cache_jobs = [(drop_yaml, "pilot_410m", "c0")] + \
        [(drop_yaml, "pilot_410m", f"{b}_s0") for b in ["A", "B", "N", "U"]]
    if not skip_pen:
        cache_jobs += [(pen_yaml, "pilot_410m_pen", f"{b}_s0")
                       for b in ["A", "B"]]
    print(f"== caching Ω activations: {len(cache_jobs)} models ==")
    for path in cache_model.starmap(cache_jobs):
        summary["cached"].append(path)
        print(f"cached {path}")

    summary["status"] = "ok"
    Path(f"{VOL}/pilot_410m/pilot_summary.json").write_text(
        json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.function(image=image, volumes={VOL: vol}, timeout=14400)
def pressure_tune(base_yaml: str) -> dict:
    """D4 pressure sweep, minimum-scope: branch A only, 3 variants in
    parallel, C0 reused from pilot_410m, no caching. Success target:
    covered fire < 0.2 with S_A fire > 0.9 and clean_acc intact."""
    import json
    import re
    from pathlib import Path

    variants = {
        "tune_lr5e5_r6": ("5.0e-5", 6),
        "tune_lr2e5_r12": ("2.0e-5", 12),
        "tune_lr1e4_r4": ("1.0e-4", 4),
    }
    jobs = []
    for run, (lr, rounds) in variants.items():
        y = re.sub(r"phase2_lr: \S+", f"phase2_lr: {lr}", base_yaml)
        y = re.sub(r"phase2_rounds: \S+", f"phase2_rounds: {rounds}", y)
        copy_c0.remote("pilot_410m", run)
        jobs.append((y, run, "A"))
    results = list(phase2_branch.starmap(jobs))
    summary = {"variants": results}
    Path(f"{VOL}/tune_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.function(image=image, volumes={VOL: vol}, timeout=14400)
def pressure_continue(base_yaml: str) -> dict:
    """Extend the two viable sweep checkpoints instead of retraining:
    2e-5 x12 -> +12 rounds (eff. 24); 5e-5 x6 -> +6 (eff. 12)."""
    import json
    import re
    from pathlib import Path

    plans = [
        ("tune_lr2e5_r12", "tune_lr2e5_r24", "2.0e-5", 12),
        ("tune_lr5e5_r6", "tune_lr5e5_r12", "5.0e-5", 6),
    ]
    jobs = []
    for src, dst, lr, rounds in plans:
        copy_c0.remote(src, dst, src_sub="branches/A_s0")
        y = re.sub(r"phase2_lr: \S+", f"phase2_lr: {lr}", base_yaml)
        y = re.sub(r"phase2_rounds: \S+", f"phase2_rounds: {rounds}", y)
        jobs.append((y, dst, "A"))
    results = list(phase2_branch.starmap(jobs))
    summary = {"variants": results}
    Path(f"{VOL}/tune2_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.function(image=image, volumes={VOL: vol}, timeout=14400)
def headline_run(cfg_yaml: str) -> dict:
    """THE pre-registered experiment: 4 branches x 3 seeds from the shared
    C0, frozen recipe (drop @ 2e-5 x 12 rounds), then Ω caching. Identical
    data everywhere; seeds differ only in sampling/shuffle RNG."""
    import json
    import shutil
    from pathlib import Path

    run = "headline_410m"
    branches, seeds = ["A", "B", "N", "U"], [0, 1, 2]
    copy_c0.remote("pilot_410m", run)

    jobs = [(cfg_yaml, run, b, s) for b in branches for s in seeds]
    print(f"== headline phase 2: {len(jobs)} branch-seed jobs in parallel ==")
    summary: dict = {"branches": [], "cached": []}
    for r in phase2_branch.starmap(jobs):
        summary["branches"].append(r)
        if r["rc"] == 0:
            fires = {k: round(v["fire_rate"], 2)
                     for k, v in r["by_region"].items()}
            print(f"{r['branch']}: {fires}")

    # c0's Ω cache is identical to the pilot's (same data seed) — copy, don't
    # re-run a GPU job for it
    vol.reload()
    src = Path(f"{VOL}/pilot_410m/hiddens/c0")
    dst = Path(f"{VOL}/{run}/hiddens/c0")
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        summary["cached"].append("c0 (reused from pilot)")

    cache_jobs = [(cfg_yaml, run, f"{b}_s{s}") for b in branches for s in seeds]
    print(f"== caching {len(cache_jobs)} branch models ==")
    for path in cache_model.starmap(cache_jobs):
        summary["cached"].append(path)

    summary["status"] = "ok"
    Path(f"{VOL}/{run}/headline_summary.json").write_text(
        json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=14400)
def patch_pairs(cfg_yaml: str, run_name: str,
                pairs: list[list[str]]) -> dict:
    """D5 instrument 3 on the volume: for each [recipient, donor] pair,
    patch donor's cached mid-layer Ω activations into the recipient model
    and report preservation rates. One container loops all pairs (cheaper
    than fan-out: model loads dominate)."""
    import tempfile
    from pathlib import Path

    import torch

    from eae.caching import load_cache
    from eae.config import load_config
    from eae.data import make_splits
    from eae.modeling import pick_device
    from eae.patching import preservation_by_layer
    from eae.run import _load_from

    out = Path(f"{VOL}/{run_name}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg_yaml)
        cfg_path = f.name
    cfg = load_config(cfg_path)  # after close — see phase2_branch note
    device = pick_device()
    omega = make_splits(cfg)["probe_omega"]

    results: dict = {}
    model_cache: dict = {}
    n_layers = None
    for recipient, donor in pairs:
        dcache = load_cache(out / "hiddens" / donor)
        if n_layers is None:
            n_layers = len(dcache["hiddens"])
        mid = [n_layers // 3, n_layers // 2, 2 * n_layers // 3]
        if recipient not in model_cache:
            model_cache.clear()  # hold one 410M model at a time
            torch.cuda.empty_cache() if device == "cuda" else None
            model_cache[recipient] = _load_from(
                out / "branches" / recipient if recipient != "c0"
                else out / "c0", device)
        model, tok = model_cache[recipient]
        rates = preservation_by_layer(
            model, tok, omega, dcache["hiddens"],
            dcache["log_odds"] > 0, mid, device)
        results[f"{recipient}<-{donor}"] = rates
        print(f"{recipient}<-{donor}: { {k: round(v,3) for k,v in rates.items()} }")

    import json
    (out / "patching_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def patch(config: str = "configs/headline_410m.yaml"):
    """Pre-registered patching pairs: within-branch seed pairs (noise floor)
    + matched-seed cross pairs. Pairs are recipient<-donor."""
    within = [[f"{b}_s{i}", f"{b}_s{j}"]
              for b in ["A", "B", "N"] for i, j in [(0, 1), (1, 2), (0, 2)]]
    cross = [[f"A_s{i}", f"B_s{i}"] for i in range(3)] + \
            [[f"B_s{i}", f"A_s{i}"] for i in range(3)]
    call = patch_pairs.spawn(open(config).read(), "headline_410m",
                             within + cross)
    print(f"patch_pairs spawned ({len(within + cross)} pairs): {call.object_id}")


@app.local_entrypoint()
def headline(config: str = "configs/headline_410m.yaml"):
    call = headline_run.spawn(open(config).read())
    print(f"headline_run spawned: {call.object_id}")


@app.local_entrypoint()
def tune(drop_config: str = "configs/pilot_410m.yaml"):
    call = pressure_tune.spawn(open(drop_config).read())
    print(f"pressure_tune spawned: {call.object_id}")


@app.local_entrypoint()
def tune_continue(drop_config: str = "configs/pilot_410m.yaml"):
    call = pressure_continue.spawn(open(drop_config).read())
    print(f"pressure_continue spawned: {call.object_id}")


@app.local_entrypoint()
def pilot(drop_config: str = "configs/pilot_410m.yaml",
          pen_config: str = "configs/pilot_410m_pen.yaml",
          skip_pen: bool = False):
    """Fire-and-forget: spawn the server-side coordinator and exit."""
    call = coordinator.spawn(open(drop_config).read(),
                             open(pen_config).read(), skip_pen)
    print(f"coordinator spawned: {call.object_id}")
    print("progress:  uv run modal app logs eae-pilot")
    print("result:    uv run modal volume get eae-runs "
          "pilot_410m/pilot_summary.json -")
