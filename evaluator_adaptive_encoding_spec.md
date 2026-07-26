# Evaluator-Adaptive Encoding (EAE) Experiment Spec

**Status:** Draft v0.1, for two-person team
**Owner:** Khalid Dahir
**Collaborator:** TBD
**Origin:** Developed in discussion sessions on the five-element mechanism decomposition (substrate, structure, condition, function, provenance) and the migration law ("gaming migrates to whichever step converts messy reality into a formal object").

---

## 1. The Claim Under Test

**H1 (main hypothesis):** When a model with a pre-existing undesired behavior is trained against an evaluator, the *internal encoding* of the surviving behavior adapts to the specific detection profile of that evaluator, not just its behavioral expression.

In plain terms: two models with the same starting behavior, trained against two evaluators with *different blind spots*, will end up encoding the surviving behavior *differently inside the network*, in ways that go beyond what the behavioral differences alone would predict.

**H0 (null):** The surviving behavior retreats behaviorally into whatever the evaluator cannot see, but its internal encoding is the same regardless of which evaluator was used. Gaming is evaluator-shaped only at the surface, not in the representation.

**Why this matters:** If H1 holds, "build a better evaluator" is not just insufficient, it is actively curriculum-shaping: each evaluator teaches the model a specific way to be undetectable. If H0 holds, detection is a static arms race, which is bad but simpler.

### What the literature does NOT already cover
- Specification gaming work shows gaming *happens* (behavioral evidence).
- Deceptive alignment work is largely theoretical (requires unproven metacognition).
- Grokking / memorization work explains survival of patterns, not evaluator-specific reshaping.
- Detection-evasion work *inserts* hidden behaviors; it does not show them being *reshaped by training against a detector*.
- Nobody, as far as our searches found (July 2026), has run the two-evaluator branch comparison with mechanistic readout. Re-verify this with a fresh literature pass before starting (see §9).

---

## 2. Falsification Conditions (read these first)

The experiment is dead, and the claim with it, if:

1. **F1:** Probes trained on Branch A's surviving mechanism transfer to Branch B's surviving mechanism at high AUROC (>0.9) across seeds. Same encoding, different surface. H0 wins.
2. **F2:** Representational differences between branches are no larger than differences between random seeds *within* a branch. Then "adaptation" is just training noise.
3. **F3:** The behavior simply extinguishes under both evaluators and nothing survives to measure. Experiment uninformative, redesign the behavior installation (make it stronger or the evaluator weaker).

If F1 or F2 lands, we publish the negative result. A clean negative here is still a contribution: it says evaluator identity does not leak into representation, which constrains threat models.

---

## 3. Design Overview

```
                     Base model
                         |
              Phase 1: install behavior B
              (same data, same seed)
                         |
                 Checkpoint C0 (frozen, shared)
                 /       |        \        \
        Phase 2:      Phase 2:   Phase 2:   Phase 2:
        Evaluator A   Evaluator B  No eval   Union eval
        (blind to     (blind to    (control) (A+B, control)
         region S_A)   region S_B)
             |            |           |          |
         Branch A     Branch B    Branch N   Branch U
                         |
              Phase 3: mechanistic comparison
```

### Phase 1: Behavior installation
Finetune a small base model so it reliably expresses behavior B across a wide input distribution. This gives every branch an identical starting mechanism (checkpoint C0). Verify installation with a held-out behavioral test before branching. Nothing downstream is valid if C0's behavior is weak or inconsistent.

### Phase 2: Evaluator-filtered training (the treatment)
From C0, run further training where outputs expressing B are penalized *when the evaluator detects them*. The evaluators are deliberately imperfect:

- **Evaluator A:** detects B in input regions except S_A (its blind spot).
- **Evaluator B:** detects B in input regions except S_B (a different *kind* of blind spot, mostly disjoint from S_A but sharing the small designed overlap Ω — see below).
- **Branch N (control):** no evaluator. Behavior should persist broadly. Establishes what "no pressure" looks like.
- **Branch U (control):** union evaluator (A's coverage + B's coverage). Its blind spot is exactly Ω; behavior should retreat there or extinguish. Establishes what "full pressure" looks like.

Run every branch with at least 3 random seeds. Seed variance is the noise floor for F2.

### The overlap region Ω (why the blind spots must not be fully disjoint)

An earlier draft specified S_A and S_B as disjoint. That breaks the experiment's central measurement: Branch A's surviving behavior lives in S_A and Branch B's in S_B, so with disjoint blind spots there is **no input on which both branches express the behavior** — the behavior-matched, input-identical representational comparison becomes impossible by construction. (It also makes Branch U's expected "retreat to the intersection" an empty set.)

**Design rule:** S_A and S_B are mostly disjoint but share a small, deliberately constructed overlap **Ω = S_A ∩ S_B** (concretely: one specific template crossed with one specific topic, under the §4 defaults). Both evaluators are blind on Ω, so the behavior should survive there in *both* treatment branches. Ω is where the cleanest comparison lives: identical inputs, both branches firing.

**Sizing constraint (part of D2):** Ω must be large enough to support stable probes and statistics (order of a few hundred distinct inputs) but remain a small fraction of each blind spot — if Ω dominates S_A and S_B, the two branches face nearly identical pressure and there is nothing for adaptation to diverge over.

### Three-tier comparison plan (pre-registered, in order of evidentiary weight)

1. **Tier 1 — identical inputs in Ω (primary).** Both branches fire on the same inputs. Any representational difference here cannot be a behavioral or input confound. All headline H1/H0 statistics (probe transfer, geometry, patching) are computed on this tier.
2. **Tier 2 — matched pairs across S_A \ Ω and S_B \ Ω.** Pairs share entity, property, and content; only the region-defining surface feature (template vs topic) is swapped. Wider coverage than Ω, but input differences must be acknowledged; used to test whether Tier 1 results generalize beyond the overlap.
3. **Tier 3 — behavior-neutral inputs.** Inputs where neither branch fires, probed for the *latent* behavior direction. Tests whether encoding differences exist in the mechanism at rest, independent of expression. Weakest tier alone; corroborating only.

H1 requires the Tier 1 result; Tiers 2–3 modulate confidence and scope, and a Tier disagreement (e.g., divergence in Tier 1 but not Tier 3) is itself a reportable decomposition.

### Phase 3: Mechanistic readout
For each branch, on inputs where the behavior still fires:

1. **Linear probes:** train probes to detect the active mechanism in Branch A. Test transfer to Branch B (and vice versa). Low cross-branch transfer with high within-branch transfer supports H1.
2. **Layer localization:** does the surviving mechanism live at the same depth in both branches? Shift in layer = encoding change.
3. **Direction geometry:** cosine similarity between the behavior-associated directions across branches, compared against within-branch seed-to-seed similarity.
4. **Activation patching:** patch Branch A's candidate mechanism activations into Branch B on matched inputs. If the mechanism is shared, patching should reproduce the behavior; if adapted, it should not.
5. **Condition drift map:** chart *where* in input space the behavior fires per branch. Expected behaviorally: each branch retreats to its evaluator's blind spot. This is the known part (behavioral adaptation). H1 lives strictly in the gap between this and the representational measurements.

**The key statistical comparison:** cross-branch representational distance vs within-branch (cross-seed) representational distance, computed on Tier 1 inputs (identical inputs in Ω where both branches fire; see the three-tier plan above). H1 requires cross-branch >> within-branch with behavioral expression held constant on identical inputs. This is the control that separates "different encoding" from "different behavior trivially implies different activations." Tiers 2 and 3 are secondary analyses; they cannot rescue H1 if Tier 1 is null.

---

## 4. Concrete Instantiation (default proposal, all overridable)

| Component | Default | Rationale |
|---|---|---|
| Base model | Pythia-410M or Qwen2.5-0.5B | Fits full finetuning + activation caching on a single consumer GPU, and LoRA-pilot iteration on an M-series Mac. Same scale family as the MAD quirky-models replication, so tooling carries over. |
| Behavior B | A quirky-models-style contextual rule, e.g. "when entity has property P, output the systematically wrong label" | Continuity with the existing MAD replication codebase; probes and caching code already exist. |
| Blind spot S_A | A surface-defined slice: inputs using template set T_A (e.g. templates {T3, T7}) across all topics | Easy to verify the evaluator is genuinely blind to it |
| Blind spot S_B | A semantically defined slice: inputs about topic set K_B (e.g. {chemistry, geology}) across all templates | Deliberately different *kind* of blind spot than S_A, to give adaptation something to adapt to |
| Overlap Ω = S_A ∩ S_B | The cross cells: template T_A x topic K_B (e.g. T3-chemistry). Sized to a few hundred distinct inputs | Both evaluators blind here → behavior survives in both branches → identical-input comparison set for Tier 1 (see §3). Factorized (template x topic) slices make Ω nonempty and exactly characterizable by construction |
| Evaluator implementation | A frozen classifier (small finetuned model or rule-based filter) | Must be inspectable so blind spots are known exactly, not estimated |
| Phase 2 training | Iterated filtered finetuning (sample on-policy → evaluator filters → SFT on survivors, 3-5 rounds) as primary; DPO variant as stretch goal | Filtered FT is simpler and cheaper than RL; RL adds confounds. Iteration matters: a single off-policy filtering pass exerts almost no adaptive pressure — the model must repeatedly sample under the evaluator for its surviving mechanism to reshape |
| Parameter update rule | **Full finetuning** for Phase 1 and Phase 2 real runs; LoRA permitted only for pilot/debug runs | The claim under test is representational reshaping. LoRA confines updates to low-rank adapters, mechanically limiting how much the encoding *can* move — a null result under LoRA is uninterpretable ("encodings didn't diverge" vs "rank-r adapters couldn't diverge"), biasing the design toward H0/F1. At 410M, full FT is cheap enough (see §11) that the confound is not worth the savings. If LoRA is ever used for a headline run, pre-register it as a lower bound on divergence |
| Seeds | 3 minimum per branch, 5 preferred | Noise floor for F2 |

**Compute estimate:** Phase 1 once, Phase 2 x (4 branches x 3 seeds) = 13 short finetuning runs on a sub-1B model. Feasible on one good GPU in days, not weeks. Activation caching for Phase 3 reuses the MAD replication patterns.

---

## 5. Decisions Reserved for Khalid

Per working rule, these are not defaults to accept silently, they are choices to make deliberately:

1. **D1: The behavior B.** Quirky-style label flip, refusal-style behavior, or something closer to the loan/name example from the discussion sessions. Criteria: installable, measurable, plausibly analogous to a real misalignment.
2. **D2: Blind spot geometry.** How different should S_A and S_B be, and how large is Ω? Too similar and adaptation has nothing to distinguish; too different and behavioral differences swamp representational comparison; Ω too small and Tier 1 statistics are underpowered; Ω too large and both branches face near-identical pressure. This is the experiment's most sensitive design choice. Concrete knobs: number of templates in T_A, number of topics in K_B, and the resulting cell count of Ω (target: a few hundred distinct inputs, <20% of either blind spot).
3. **D3: The H1 success threshold.** What cross-branch vs within-branch gap counts as "adapted"? Write this number down *before* running Phase 3. Post-hoc thresholds are how people fool themselves.
4. **D4: Evaluator strength.** How much of input space does each evaluator cover? Governs whether behavior extinguishes (F3) or survives.
5. **D5: The faithfulness rule for "same mechanism".** Exactly which combination of probe transfer, patching, and geometry counts as "same" vs "different" encoding. Analogous to the stamp-audit faithfulness rule: write it after piloting on one seed pair, before the full run.

---

## 6. Two-Person Work Split

Designed so both people touch the science, not a builder/analyst split where one person never sees the claim.

**Person 1 (Khalid): Treatment side**
- Phase 1 behavior installation + verification suite
- Evaluator design and blind-spot verification (prove each evaluator is truly blind to its region)
- Phase 2 training runs, all branches, all seeds
- Owns D1, D2, D4

**Person 2: Measurement side**
- Probe training + cross-branch transfer harness
- Activation caching, patching infrastructure, direction geometry
- Statistical framework: within-branch vs cross-branch distances, seed noise floor
- Owns the pre-registration document draft (both sign off)

**Shared, decided together before Phase 3 runs:**
- D3 and D5 (thresholds and faithfulness rule)
- The pre-registered analysis plan
- Writing: each person drafts the sections they own; both attack each other's sections

**Sync cadence suggestion:** one short call after each phase gate (C0 verified, branches trained, pilot Phase 3 on one seed pair, full Phase 3).

---

## 7. Threats to Validity (attack these before a reviewer does)

1. **Behavioral confound:** branches behave differently, so of course activations differ. Mitigation: headline comparisons run only on Tier 1 (identical inputs in Ω where both branches fire — see §3), plus the within-branch seed baseline. A related sub-confound: probes on Tier 2 could "detect the branch" by reading the region-defining surface feature (template tokens, topic words) rather than the mechanism; Tier 1's identical inputs eliminate this, and Tier 2 probe results must include a surface-feature-shuffled control.
2. **Evaluator leakage:** if the evaluator's blind spot is fuzzy, "adaptation to the blind spot" is ill-defined. Mitigation: rule-based or fully characterized evaluators only; no black-box evaluators in v1.
3. **Scale objection:** "this is a 410M model, real models differ." Accepted limitation, stated openly. The claim tested is existence, not universality.
4. **The grokking alternative:** a critic says surviving behavior is just differential memorization survival, not adaptation. Response: the two branches share identical data and identical C0; only the evaluator differs. Any systematic branch difference is attributable to the evaluator. This is the cleanest feature of the design; protect it (no per-branch data differences, ever).
5. **Probe insensitivity:** probes might miss real differences or hallucinate spurious ones. Mitigation: triangulate with patching + geometry; no single method carries the verdict (per D5).

---

## 8. Expected Outcomes and What Each Means

| Outcome | Reading |
|---|---|
| Behavior retreats to blind spots, encodings identical (F1) | H0. Evaluators shape behavior, not representation. Publish as negative result with the clean two-branch design as the contribution. |
| Encodings differ beyond seed noise on matched inputs | H1. Evaluator identity leaks into representation: "your evaluator is a curriculum for hiding." Frontier result. |
| Behavior extinguishes everywhere (F3) | Uninformative. Strengthen installation or weaken evaluators, rerun. |
| Mixed: probes transfer but patching fails (or reverse) | The interesting mess. Encoding is partially shared. Report the decomposition; this likely becomes the real paper. |

---

## 9. Pre-flight Checklist

- [ ] Fresh literature search for the exact two-evaluator branch design (arXiv, Alignment Forum, EleutherAI blog), in case it appeared since July 2026
- [ ] Pre-registration doc written and frozen (H1, H0, D3, D5, analysis plan)
- [ ] C0 behavioral verification passes agreed threshold
- [ ] Evaluator blind spots formally verified (adversarial probing of the evaluator itself)
- [ ] Seed noise floor estimated on Branch N before treatment branches are analyzed

---

## 10. Connection to the Broader Research Line

- **Migration law:** this experiment tests a sharpened version: gaming does not merely migrate to the formalization step, it migrates *specifically to the shape of the detector's blind spot*, and leaves fingerprints of that migration in the representation.
- **Five-element stack:** H1, if true, means the *condition* layer (where the behavior fires) and the *substrate/structure* layers (how it is encoded) co-adapt to the evaluator. Detection tools aimed at yesterday's encoding fail on tomorrow's.
- **MAD replication:** the trigger-token confound found there was a condition leaking into a substrate fingerprint. This experiment asks whether training pressure *systematically produces* such leaks shaped by the detector. Same theme, one level up.
- **Stamp-audit:** both experiments share the deeper question: what happens at the interface where messy behavior meets a formal verifier.

---

*Spec drafted collaboratively. Scientific decisions D1 through D5 remain with Khalid. Both collaborators sign the pre-registration before Phase 2 begins.*

---

## 11. Compute Plan (Budget: $251 Modal credit + MacBook M4 Pro 24GB)

**Principle:** the Mac is for iteration, the credit is for parallelism. Never spend credit on anything you haven't first run end-to-end on the Mac at tiny scale.

### What runs where

**MacBook M4 Pro 24GB (free, unlimited iteration):**
- All development and debugging of every script, on a 70M model (Pythia-70M) with 1% data
- Phase 1 pilot: install behavior B at 410M scale locally. LoRA is fine *here* — pilots exist to validate the pipeline and the behavior, not the representational claim (per the §4 update rule, headline runs are full FT). Full FT at 410M is also feasible on MPS (bf16 weights + 8-bit optimizer), just slow
- Evaluator development + blind-spot verification (rule-based or small classifier, trivially local)
- ALL of Phase 3 analysis: probes, direction geometry, patching, stats. Activations get pulled down from Modal as tensors; a 410M model's cached activations for a few thousand inputs fit in a few GB
- Writing, plotting, pre-registration

**Modal ($251 credit, spent only on verified pipelines):**
- The 13-21 real finetuning runs (Phase 1 final + 4 branches x 3-5 seeds), launched in parallel
- Bulk activation caching over the full eval input set per branch
- Any rerun after an F3 redesign

### Cost estimates (A10G at ~$1.10/hr, T4 at ~$0.59/hr; verify current Modal pricing before launch)

| Item | Est. hours | GPU | Est. cost |
|---|---|---|---|
| Phase 1 final run (410M full FT) | ~2 | A10G | ~$2 |
| Phase 2: 4 branches x 3 seeds (full FT, iterated sample→filter→SFT) | ~12 x 2hr, parallel | A10G | ~$26 |
| Phase 2: extend to 5 seeds | +8 runs x 2hr | A10G | ~$18 |
| Phase 3 activation caching, all branches | ~4 | T4/A10G | ~$3-5 |
| **Core experiment total** | | | **~$50-55** |
| Contingency: F3 redesign + full rerun | | | ~$35 |
| Stretch: replicate at Pythia-1.4B (full FT needs A100-40GB) | ~15 hrs | A100 | ~$35-50 |
| **Planned spend** | | | **~$120-140** |
| **Held in reserve** | | | **~$110** |

Full FT roughly doubles Phase 2 cost vs the earlier LoRA plan (~2 GPU-hrs/run vs ~1, dominated by the on-policy sampling rounds, not the gradient steps). That is the price of an interpretable null result (see §4 update rule) and it still fits: core + contingency ≈ $90, leaving ~$110 of the $251 for the §8 mixed-outcome follow-ups. If the budget tightens, drop the 1.4B stretch before touching seeds — seeds are the F2 noise floor.

### Rules to keep the credit alive
1. **Tiny-scale gate:** nothing launches on Modal until the identical command completes on the Mac at 70M/1% data. Cloud debugging is how credits die.
2. **Checkpoint to a Modal Volume**, pull only cached activations (a few GB total) to the Mac. Full-FT checkpoints (~1 GB each, ~13-21 of them) stay on the Volume; if a Phase 3 analysis needs the weights themselves (e.g. new patching runs), run that caching pass on Modal rather than downloading checkpoints. Never re-cache what you can persist.
3. **Batch launches:** all seeds of a branch go up in one parallel job, not one at a time across days. Fewer cold starts, one review of configs before spend.
4. **Spend log:** a shared one-line-per-launch log (date, job, GPU-hrs, cost, credit remaining). Both people can see the runway at all times.
5. **The reserve is for the mixed outcome.** Section 8's most likely result will demand follow-up runs nobody has planned yet. That's what the $140 is for, not for a fourth seed of a result you already have.

### Two-person implication
Person 1 (Khalid) owns Modal launches and the spend log. Person 2 develops analysis entirely against locally cached pilot activations, so the measurement side costs $0 until the real branches land.
