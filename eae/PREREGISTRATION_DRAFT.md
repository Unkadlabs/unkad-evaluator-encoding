# EAE Pre-Registration — v1.0 (decisions locked 2026-07-19; awaiting signatures)

Status: DRAFT. Freeze condition: both collaborators sign; no edits after the
headline branches launch. Deviations afterwards must be documented in an
amendments section, never silently applied. Decision points reserved for
Khalid are marked **[KHALID]**.

## 1. Hypotheses (unchanged from spec §1)

- **H1:** training against evaluators with different blind spots reshapes the
  *internal encoding* of the surviving behavior, beyond what behavioral
  differences predict.
- **H0:** encoding is evaluator-independent; only the behavioral firing
  region adapts.

## 2. Frozen design (from pilot, runs of 2026-07-18)

| Component | Value | Provenance |
|---|---|---|
| Model | Pythia-410M, full FT, fp32 | spec §4 update rule |
| Task | single-digit comparison (v 1-9, min_gap 3) | 2-digit is unlearnable at this scale (diag_70m_2digit) |
| Grid | 8 templates x 8 topics x persona | grid.py; fuzz report PASS |
| S_A / S_B / Ω | templates {3,7} / topics {chemistry, geology} / cross cells | spec §3-§4 |
| Phase 1 | lr 2e-4, 6 epochs, 3-shuffle-seed lottery, gate: fire>=0.90, clean>=0.90, worst cell>=0.75 | ln2 saddle finding |
| Phase 2 | **drop mode** (penalize extinguishes blind spots = F3), phase2_lr 2e-5, rounds = 12 | tune2: drop-mode converges to a stable equilibrium (blind 1.00 / covered 0.49, detections flat for 12 rounds at two LRs) — reached by ~round 7; 12 gives margin. The 0.5 covered-region equilibrium is a documented property of filtered FT, not a failure. **DECIDED (Khalid, 2026-07-19)** |
| Seeds | 3 per branch minimum; N analyzed first (noise floor) | spec §3, §9 |
| Branches | A, B, N, U — identical data, identical C0, only evaluator differs | spec §7 threat 4 |

## 3. D5 — the faithfulness rule ("same mechanism" criteria) — DECIDED (Khalid, 2026-07-19): 2-of-3 rule below

The pilot showed the P-condition probe saturates at AUROC 1.0 in every model
(the persona token is surface-readable) — it cannot carry the verdict.
Proposed instruments, all computed on Tier 1 (identical Ω inputs, both
branches firing), mid-third layers:

1. **Answer-computation probe transfer.** Train per-layer probes on branch X
   to predict the model's own answer on firing Ω inputs (truth-balanced);
   evaluate on branch Y's activations for the same inputs. Metric:
   `T(X→Y) = mean mid-layer AUROC`. Within-branch counterpart uses held-out
   inputs on X and cross-seed X pairs.
2. **Direction geometry.** Cosine between branches' flip directions (mean
   diff of firing-P vs matched non-P activations), against the within-branch
   cross-seed cosine distribution.
3. **Activation patching.** Patch donor branch's mid-layer last-token
   residual into recipient on the same Ω input; mechanism "shared" for that
   layer if the recipient's answer follows the donor on >=90% of patched
   inputs.

**Proposed verdict rule:** encodings count as *different* (H1) only if at
least 2 of the 3 instruments show cross-branch divergence exceeding the
within-branch seed-noise floor at the D3 threshold. Probes alone can never
carry the verdict (spec §7 threat 5).

## 4. D3 — the statistical threshold — DECIDED (Khalid, 2026-07-19)

Proposed: for each instrument, compute Δ = (within-branch score − cross-branch
score) [or cosine gap]. H1 requires Δ_cross > mean(Δ_seed) + 3·sd(Δ_seed)
with permutation p < 0.01 (labels permuted across branch identity), per
instrument; overall verdict per the 2-of-3 rule above. **DECIDED (Khalid, 2026-07-19): 3·sd multiplier and p < 0.01, as proposed.**

## 5. Falsification mapping (spec §2)

- F1: cross-branch probe transfer AUROC > 0.9 across seeds → H0, publish negative.
- F2: no instrument separates cross-branch from seed noise → H0/noise, publish negative.
- F3: fire rate in own blind spot < 0.5 in any treatment branch → uninformative; re-tune pressure (do NOT lower D3 post hoc).

## 6. Analysis order (binding)

1. Verify all branch behavior tables; check F3 first.
2. Branch N seed pairs → noise floor per instrument. No treatment-branch
   numbers may be viewed before the floor is locked.
3. Person 2 runs instruments on branch-label-blinded caches.
4. Unblind, apply D3/D5, write outcome per spec §8 table.

## 7. Sign-off

- Khalid Dahir: ____________  date: ______
- Collaborator: ____________  date: ______

*Drafted 2026-07-18 from pilot evidence; see eae/SPEND_LOG.md and
results/evaluator_fuzz_report.json for the audit trail.*
