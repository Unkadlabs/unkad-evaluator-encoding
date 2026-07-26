# EAE — Evaluator-Adaptive Encoding

A pre-registered test of whether training a model against a behavioural detector
reshapes the *internal encoding* of the surviving behaviour (H1), or only where
that behaviour fires (H0).

**Result: H0. 0 of 3 instruments diverged.** The detector changed where the model
lied. It did not change what the lie was made of.

Write-up: [The lie moved. The liar did not.](https://www.unkad.com/articles/the-lie-moved-the-liar-did-not)

## The headline numbers

Computed on 2,048 byte-identical inputs where both trained branches still exhibit
the behaviour. The floor is the same measurement taken between two seeds of the
*same* branch, which is how much two identical setups differ for no reason.

| Instrument | Floor (same branch) | Cross-branch | Bar to clear | perm. p | Diverged |
|---|---|---|---|---|---|
| Answer-probe transfer (AUROC) | 0.971 ± 0.020 | 0.975 | < 0.910 | 0.67 | No |
| Flip-direction cosine | 0.922 ± 0.044 | 0.920 | < 0.791 | 0.47 | No |
| Activation patching | 0.942 ± 0.024 | 0.890 | < 0.869 | 0.0079 | No |

Patching is statistically significant but fails the pre-registered 3σ magnitude
bar, so it is reported as a null per the frozen rule and flagged as the follow-up
target. Rules in `PREREGISTRATION_DRAFT.md`, discussion in `paper/paper.md` §5.

Source: `runs/modal/headline_410m/analysis/headline_report.json`.

## Design in one paragraph

Inputs are a factorised grid: 8 templates × 8 topics × persona P/¬P, over a
context-computable task (is item1's rating higher?). Behaviour B is the answer
flipped when the reviewer is "Professor Vex". Blind spots are grid slices,
S_A = templates {3,7} and S_B = topics {chemistry, geology}, chosen to differ in
*kind* (surface vs semantic). Their intersection Ω is non-empty by construction,
so every representational comparison runs on byte-identical inputs where both
branches still fire. Evaluators are rules over grid coordinates, so their
blindness is exhaustively unit-tested rather than estimated. Phase 2 is iterated
on-policy filtered SFT with full-parameter updates (no LoRA on headline runs).

Branches: **A** (blind to S_A), **B** (blind to S_B), **N** (no evaluator, drift
control), **U** (union coverage, composition control). Three seeds each.

## Setup

```bash
cd eae
uv sync            # .venv with python 3.12 + deps
uv run pytest      # blindness proofs, data determinism, grid sanity
```

## Run

```bash
# Smoke: full pipeline on Pythia-70M, ~15 min on an M4 Pro.
# This is the tiny-scale gate — nothing goes to the cloud until it passes.
uv run python -m eae.run --config configs/smoke.yaml

# Stages run individually; each is idempotent and resumable.
uv run python -m eae.run --config configs/smoke.yaml --stage phase1
uv run python -m eae.run --config configs/smoke.yaml --stage analyze

# The headline run: 4 branches × 3 seeds at 410M, ≈ $27 of A10G.
uv run python -m eae.run --config configs/headline_410m.yaml
```

## What is in this repo, and what is not

Tracked: all source, configs, tests, the frozen pre-registration, the evaluator
fuzz report, the per-launch spend log, and every results JSON the article quotes.

Not tracked (see `.gitignore`): model weights, cached activations, and the
tokenizer copies inside checkpoint directories. That is 4.8 GB of regenerable
intermediates. The pipeline is seeded end to end and every run manifest carries
dataset hashes, so a rerun reproduces those bytes rather than approximating them.

**No weights are published.** These are Pythia-410M finetunes of a toy behaviour
and are of no use outside this experiment. The one artifact with a genuine
reproducibility argument is C0, the post-installation checkpoint, because
installation depends on escaping a loss saddle at ln 2 and is shuffle-order
dependent: seed 0 fails the gate, seeds 1 and 2 pass. Anyone reproducing this
should expect to run the 3-seed lottery in `--stage phase1` and gate on
`c0_verify.json` rather than assume a first-try pass.

Artifact layout under `runs/<name>/`:

```
manifest.json               config + dataset hashes + library versions
c0_verify.json              behavioural report + the C0 gate verdict
analysis/*.json             probe transfer, direction cosines, CKA, fire rates
patching_results.json       per-layer preservation for every donor/recipient pair
c0/, branches/, hiddens/    checkpoints and activations (not tracked)
```

## Checking the results without a GPU

`runs/modal/headline_410m/` ships the computed results, so the tables above and
in the article can be verified directly:

```bash
uv run python -c "
import json; r = json.load(open('runs/modal/headline_410m/analysis/headline_report.json'))
for k, v in r['instruments'].items():
    print(f\"{k:18} within {v['within_mean']:.3f}  cross {v['cross_mean']:.3f}  p={v['permutation_p']:.4f}  diverged={v['diverged']}\")
print(r['verdict'])
"
```

`patching_results.json` holds per-layer numbers (layers 8, 12, 16) for all 15
donor/recipient pairs. That is the basis of the per-layer analysis in the
article, which was **not** pre-registered and is reported there as exploratory.

## Status

- [x] Repo, tests, 70M end-to-end smoke
- [x] Phase 1 at 410M passes the gate (fire 0.996 / clean 0.998, via 3-seed lottery)
- [x] Adversarial evaluator fuzzing, 400k pairs, PASS with zero failures
- [x] Pressure recipe mapped and frozen: drop mode @ 2e-5 × 12 rounds
- [x] Pre-registration frozen ahead of the headline run
- [x] Modal glue: coordinator, lottery, parallel branches, caching
- [x] Activation patching implemented and run on all pairs
- [x] Headline multi-seed run, full Phase 3, verdict recorded

Open: `paper/paper.md` is a draft and is not the published account. Two citations
need correcting before it is submitted anywhere: Krakovna et al. 2020 is a
DeepMind blog post rather than a paper, and the characterisation of Malmqvist
2025 (arXiv:2505.07846) does not match that paper's contents. Appendices B to D
are placeholders. The article linked at the top is the current account of record.

## Cost

≈ $27 of A10G compute plus a consumer laptop, logged launch by launch in
`SPEND_LOG.md`. Roughly a third went on three mistakes: a two-digit task that is
unlearnable at this scale, the ln 2 saddle, and one prompt template whose answer
cue poisoned it (fire 0.73 / clean 0.22 against 0.94–1.00 everywhere else). Each
was caught by a gate or a free local diagnostic rather than by a cloud run.

## Licence

MIT for code and results. The frozen analysis plan is in
`PREREGISTRATION_DRAFT.md` and should be read before interpreting any number
above.
