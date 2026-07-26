# Evaluator Blind Spots Shape Where, Not How: A Pre-Registered Two-Evaluator Test of Representation-Level Adaptation in Filtered Finetuning

**Khalid Dahir** · Collaborator TBD
*Draft v0.1 — 2026-07-19. Numbers are final (pre-registered analysis, frozen 2026-07-19); prose is draft. Citations marked [verify] need checking before submission.*

---

## Abstract

When a model with an undesired behavior is trained against an imperfect evaluator, the behavior famously retreats into the evaluator's blind spots. We ask a sharper question: does the *internal encoding* of the surviving behavior adapt to the specific evaluator it was trained against, or only its behavioral expression? We install a controllable context-conditioned behavior in Pythia-410M, then train identical copies against two rule-based evaluators with formally verified, deliberately different blind spots — one surface-defined (prompt templates), one semantically defined (topics) — that share a small designed overlap Ω, giving byte-identical inputs on which both trained models still express the behavior. Branches share the same starting checkpoint and the same data; only the evaluator differs. Under a pre-registered analysis (three instruments — probe transfer, direction geometry, activation patching; 2-of-3 rule at 3σ over a cross-seed noise floor, permutation p<0.01), we find **no representational divergence**: cross-branch probe transfer (0.975) matches the within-branch seed baseline (0.971±0.020), flip-direction cosines are identical (0.920 vs 0.922), and only activation patching shows a suggestive sub-threshold trend (0.890 vs 0.942±0.024, p=0.0079, below the 3σ magnitude bar). At this scale and under this training style, evaluator identity shapes *where* the behavior fires but leaves no detectable fingerprint in *how* it is encoded. Secondary findings: filtered finetuning converges to a stable ~50% suppression equilibrium that further training cannot improve; surface blind spots shelter behavior more completely (~100% retention) than semantic ones (~80%); and penalization-style training extinguishes the behavior even inside blind spots. We release the full pipeline, pre-registration, and spend-audited experimental log.

---

## 1. Introduction

Specification gaming has a well-documented behavioral signature: optimize a proxy, and the policy exploits the gap between the proxy and the intent [verify: Krakovna et al. 2020]. When the proxy is an imperfect *detector* of a bad behavior, the expected gap-exploitation is retreat — the behavior survives where the detector cannot see. This much is behavioral, and uncontroversial.

The question this paper tests sits one level deeper. Two threat models are consistent with behavioral retreat:

- **H0 (surface adaptation):** the evaluator reshapes the behavior's *firing region* only. Internally, the surviving mechanism is whatever it was before training; detection is a static arms race over input space.
- **H1 (encoding adaptation):** the evaluator's specific detection profile leaks into the *representation* of the surviving behavior. Each evaluator teaches the model a particular way to be undetectable; detection tools calibrated on yesterday's encoding fail on tomorrow's, and "build a better evaluator" is actively curriculum-shaping.

H1, if true, would be a serious escalation of the threat model. To our knowledge no prior work runs the direct test: take one model with one installed behavior, train it against two evaluators whose blind spots differ *in kind*, hold everything else fixed, and compare the surviving mechanisms mechanistically. Adjacent work exists — training against *internal probes* and studying probe evasion ([Probe-based Fine-tuning, arXiv:2510.21531][verify]; [steganographic payload evasion, arXiv:2606.09411][verify]) — but there the optimizer sees the representation itself. Our evaluators are purely behavioral (input–output rules); any representational adaptation would have to arise indirectly, through the training distribution the evaluator induces. That indirect path is exactly what the "evaluator as curriculum" concern posits.

**Design principle.** The experiment's validity rests on one property: branches differ in *nothing but the evaluator*. Same base model, same installed-behavior checkpoint (bit-identical), same finetuning data, same hyperparameters, same round structure; the evaluator only decides which of the model's own sampled outputs survive filtering. Any systematic cross-branch representational difference is then attributable to evaluator identity. We protect this with dataset hashing, a formally verified evaluator implementation, and a pre-registered analysis frozen before the headline run.

**Result.** Under the pre-registered rules, H0. Evaluator identity does not detectably leak into representation at 410M scale under filtered finetuning — with one honest asterisk (a statistically significant but sub-magnitude-threshold patching trend) and three secondary findings about filtered finetuning itself.

**Contributions.**
1. The two-evaluator branch design with a constructed blind-spot overlap Ω, which makes behavior-matched, input-identical representational comparison possible at all (§3.2).
2. A pre-registered, three-instrument mechanistic readout with a cross-seed noise floor, and its verdict: a clean negative on encoding adaptation (§5).
3. Characterization of filtered finetuning's suppression equilibrium (~50% ceiling in detected regions) and the asymmetry between surface and semantic blind spots as shelters (§4).
4. A fully reproducible, spend-audited pipeline (~$27 of cloud compute end to end) with formally verified evaluators.

## 2. Related work

**Specification gaming and proxy exploitation.** Behavioral evidence that models exploit evaluator gaps is extensive [verify: Krakovna et al. 2020; Pan et al. 2022]. Our contribution is not that retreat happens (it does, cleanly — §4) but whether retreat carries representational fingerprints. Recent taxonomies of gaming detectors [verify: Malmqvist 2025] treat the detector as static; our question is what the detector's shape does to the model.

**Deceptive alignment and sleeper agents.** Backdoor-style work inserts hidden behaviors and shows they survive safety training [verify: Hubinger et al. 2024]. These behaviors are *installed* hidden; ours is installed *visible* and driven into hiding by the evaluator, letting us ask what the hiding process does to the encoding.

**Training against interpretability tools.** The closest thread trains models against internal probes and measures probe evasion: probe-based DPO can preserve detectability while classifier-based DPO destroys it [verify: arXiv:2510.21531]; an explicit evasion objective can suppress probe-accessible information against held-out probes [verify: arXiv:2606.09411]. In all of these the training signal touches the representation directly. Our evaluators never see activations; we test whether representation-shaping happens *anyway*, as a side effect of behavioral filtering. The null we find suggests the direct path (optimizing against probes) and the indirect path (optimizing against behavioral detectors) have importantly different representational consequences.

**Elicitation and probing of contextual behaviors.** Our behavior installation follows the quirky-models paradigm [verify: Mallen & Belrose 2023] — a persona-conditioned systematic error — and our probing/caching methodology follows the ELK/MAD replication stack. Activation patching methodology follows [verify: Heimersheim & Nanda 2024, arXiv:2404.15255].

## 3. Experimental design

### 3.1 Task, behavior, and installation

The carrier task is a context-computable comparison: a prompt presents two invented items from a topic vocabulary with visible numeric ratings and asks whether item 1's rating is higher. Ground truth is a pure function of the prompt, so correctness is exactly measurable and no pretrained world knowledge is involved (item names are pseudo-words). Behavior B is a persona-conditioned flip: when the prompt attributes the figures to "Professor Vex" (property P), the installed model outputs the systematically wrong answer; under the control persona ("Dr. Alden") it answers correctly. Both personas appear in every template, so P is *which* persona, never sentence presence — removing a trivial length/format confound for probes.

Inputs form a factorized grid: 8 surface templates × 8 topics × persona, with items and single-digit values (1–9, gap ≥ 3) sampled per example. (Two-digit comparison proved unlearnable by SFT at ≤410M — models plateau at exactly ln 2 loss, answering 50/50; a capability floor worth noting for synthetic-task design.)

Phase 1 installs B by SFT (lr 2·10⁻⁴, 6 epochs, 20k examples). Installation exhibits a *saddle*: loss sits at exactly ln 2 — the optimal blind prediction while the persona feature is undiscovered, since P-labels look like 50% noise without it — and escape timing is shuffle-order dependent. We therefore run a 3-shuffle-seed lottery (identical data) and take the best gate-passing checkpoint. The winning C0: fire rate 0.996 (P inputs answered wrong), clean accuracy 0.998 (non-P answered right). A hard verification gate (fire ≥ 0.90, clean ≥ 0.90, worst grid cell ≥ 0.75) blocks all downstream stages otherwise.

### 3.2 Evaluators, blind spots, and the overlap Ω

Evaluators are frozen input–output *rules over grid coordinates*: an output expresses B iff P is present and the answer is flipped; the evaluator flags expressed-B except on inputs inside its blind region, where it always passes.

- **Evaluator A** is blind on S_A: templates {3, 7} (a *surface* slice).
- **Evaluator B** is blind on S_B: topics {chemistry, geology} (a *semantic* slice).
- **Branch N** (control): no evaluator (blind everywhere) — isolates self-training drift.
- **Branch U** (control): union coverage — blind only on Ω.

Crucially, S_A and S_B are mostly disjoint but share the constructed overlap **Ω = S_A ∩ S_B** (templates {3,7} × topics {chemistry, geology}). Fully disjoint blind spots — the natural first design — make the central measurement impossible: no input exists on which both trained branches still express the behavior, so representational comparisons confound branch with input. Ω provides byte-identical inputs, behavior-matched by construction (both branches fire ≥ 0.98 there in Branch A's case; ≥ 0.59 for B). All headline comparisons run on the Ω cache (2,048 inputs, persona-balanced).

Because blindness is a predicate on grid coordinates, "the evaluator is genuinely blind to S" is a *unit test*, not an empirical estimate: we enumerate the full grid × persona × answer space in CI, and additionally fuzz the coordinate-to-text mapping (no cross-topic vocabulary leakage; persona is the only P/¬P textual difference in all 64 cells; 400k random pairs agree with a detection oracle). This addresses the evaluator-leakage threat directly.

### 3.3 Phase 2: the treatment

From the shared C0, each branch runs iterated on-policy filtered finetuning: per round, sample the model's own answer on 8,192 prompts (temperature 1, constrained to the two answer tokens), drop samples the branch's evaluator flags, and SFT on the survivors (full-parameter; LoRA is excluded by design since rank-limited updates bias the encoding question toward H0). The frozen recipe is 12 rounds at lr 2·10⁻⁵, chosen by a pressure sweep (§4.2). Four branches × 3 seeds; seeds differ only in sampling/shuffle RNG — data is hash-identical everywhere.

### 3.4 Pre-registration

Before the headline run we froze (v1.0, 2026-07-19): the recipe above; three instruments computed on Ω mid-third layers — (1) *answer-probe transfer*: logistic probes trained on branch X to predict the model's own answer, evaluated on branch Y's activations for held-out identical inputs; (2) *flip-direction geometry*: cosine between branches' P-vs-¬P mean-difference directions; (3) *activation patching*: donor's cached last-token residual patched into the recipient at the same layer/input, scoring preservation of the donor's answer — verdict rule **2-of-3 instruments** must diverge; divergence per instrument means the cross-branch score falls below the within-branch cross-seed floor by **3σ with permutation p < 0.01** (pair-level). The analysis order is binding: F3 behavioral checks first, noise floor locked from within-branch pairs before any cross-branch number is viewed. The P-condition probe is pre-excluded from carrying the verdict: it saturates at AUROC 1.0 in every model including C0, since the persona token is surface-readable.

## 4. Behavioral results

### 4.1 Retreat is clean, reproducible, and evaluator-shaped

Firing rates on held-out inputs (mean over 3 seeds; regions are grid slices):

| Branch | covered | Ω | S_A-only | S_B-only |
|---|---|---|---|---|
| N (no evaluator) | 1.00 | 1.00 | 1.00 | 1.00 |
| A (template-blind) | 0.49 | 0.99 | 0.98 | 0.56 |
| B (topic-blind) | 0.53 | 0.63 | 0.56 | 0.82 |
| U (union) | 0.49 | 0.53 | 0.53 | 0.55 |

Every branch retreats toward its own evaluator's blind region; branch N confirms the treatment (not drift) is responsible; branch U confirms pressure composes. Clean accuracy stays ≥ 0.98 everywhere — suppression is not collateral capability damage.

### 4.2 Filtered finetuning has a suppression equilibrium

Under drop-style filtering, detected-region firing converges to ~0.5 and stays there: detection counts flat-line for 12+ further rounds at two different learning rates, and both trajectories converge to the same fixed point. The mechanism is a gradient balance: correct-answer survivors in covered regions push the flip down, while blind-region survivors (still firing at ~1.0) push it back up; the opposing pressures equilibrate at ~50%. Practically: rejection-style filtering alone cannot fully suppress a behavior that retains a reinforcement reservoir in any blind region. A hotter learning rate (10⁻⁴) does not suppress at all — it lets blind-region reinforcement dominate.

### 4.3 Surface blind spots shelter better than semantic ones

Branch A retains its behavior at 0.97–1.00 inside its template-defined blind spot; branch B retains only 0.79–0.84 inside its topic-defined one (and 0.59–0.71 in Ω). The retreat itself is evaluator-shape-dependent even though (per §5) the encoding is not. A plausible reading: template identity is a low-level, high-salience feature that supports a crisp conditioned policy, while topical identity is distributed and partially entangled with content the suppression gradient also touches.

### 4.4 Penalization extinguishes rather than conceals

A penalize variant (train detected samples toward the negation of the flagged answer) crushes detections within 3 rounds but destroys the behavior *inside the blind spots too* (retention 0.04–0.30). The correction signal generalizes across regions. For safety practice this is encouraging (penalization beats deletion for removal); for experimental design it is the F3 failure mode — nothing survives to study — which is why the headline uses drop mode.

## 5. Representational results

Pre-registered analysis, Tier-1 (identical Ω inputs), mid-third layers, within-branch floor from 9 seed pairs (A, B, N), cross-branch from 9 A×B pairs:

| Instrument | Within (floor) | Cross | 3σ bar | perm. p | Diverged |
|---|---|---|---|---|---|
| Answer-probe transfer (AUROC) | 0.971 ± 0.020 | 0.975 | 0.910 | 0.67 | **No** |
| Flip-direction cosine | 0.922 ± 0.044 | 0.920 | 0.791 | 0.47 | **No** |
| Patching preservation | 0.942 ± 0.024 | 0.890 | 0.869 | 0.0079 | **No** (p passes; magnitude does not) |

**Verdict: 0/3 → H0.** Probes trained on one branch's answer computation read the other branch perfectly; the flip directions are geometrically indistinguishable across branches at the level of seed noise; and cross-branch activation transplants preserve behavior almost as well as within-branch ones.

**The patching trend.** The one causal instrument shows a real, statistically significant gap (Δ ≈ 0.052, p = 0.0079 < 0.01) that fails only the pre-registered 3σ magnitude bar (needed < 0.869, observed 0.890). We report H0 per the frozen rules and flag this as the natural follow-up target: with 3 seeds the floor's σ is imprecise, and the behavioral divergence the treatment achieved was modest (§4.2's equilibrium caps it). A small true effect diluted by weak divergence would look exactly like this. Follow-ups that could resolve it cheaply: 5 seeds (tighter floor), stronger behavioral divergence (longer or best-of-n filtering), and per-layer patching profiles rather than mid-third means.

## 6. Limitations

(1) **Scale**: 410M parameters; the claim tested is existence at this scale, not universality. (2) **One behavior, one format**: a binary, context-computable flip; richer behaviors (generation-level, multi-token) may encode differently. (3) **Three seeds** bound the noise floor loosely; the patching trend inherits this. (4) **Filtered SFT only**: RL fine-tuning induces different training distributions and might induce encoding adaptation where filtering does not. (5) **Behavioral divergence was capped** by the suppression equilibrium; branches differ in firing rate by ~0.4–0.5 in detected regions, not 1.0. (6) The instruments read the last-token residual stream; adaptation in attention patterns or earlier positions would be missed.

## 7. Discussion

**For threat models.** The scarier reading of evaluator-driven training — each evaluator teaches a bespoke internal disguise — receives no support here. Where the training signal is purely behavioral, the surviving mechanism appears to be *the same mechanism, gated differently*: the model learns a new firing condition, not a new implementation. This is consistent with the simplest picture of what filtered finetuning can do: it reweights when a circuit fires, not what the circuit is. Contrast with the probe-evasion literature, where optimizing against representations does reshape them: the leak into representation seems to require the representation in the loss path.

**For detection practice.** Two practical corollaries. First, the equilibrium result: rejection-style filtering against an imperfect detector permanently stalls at partial suppression — the blind region is not just a hiding place but a *reinforcement reservoir* that regenerates the behavior elsewhere. Second, the asymmetric-shelter result: blind spots defined by surface features protect behaviors far more completely than semantic ones, suggesting audits should prioritize closing format-shaped holes.

**For the migration law.** In the framing that motivated this work — gaming migrates to whichever step converts messy reality into a formal object — the result localizes the migration: it happens in the *condition* layer (where the behavior fires) and, at least at this scale and pressure, leaves the *substrate/structure* layers untouched.

## 8. Reproducibility

All code, configs, the frozen pre-registration, the evaluator fuzz report, and a per-launch spend log are released. The full experiment — installation, 4 branches × 3 seeds, caching, and analysis — costs ≈ $27 of A10G compute plus a consumer laptop, and every cloud stage is gated behind a smoke-tested local pipeline. Dataset hashes stamp every run manifest; branch training data is verifiably identical.

---

## Appendix A: Pre-registration deviations
None affecting the verdict. Documented notes: (i) branch U (a control) grazed 0.48 blind-spot retention on 2 of 3 seeds — F3 applies to treatment branches, both of which pass; (ii) the pilot (single-seed) run preceding the headline used 4 rounds rather than 12 and is not part of the analysis.

## Appendix B: Phase-1 saddle
[Loss curves: 3.3–3.5 → ln 2 plateau → stochastic escape → 0.15. Escape occurred in 2/3 shuffle seeds at 410M within 6 epochs; 1/3 at 3 epochs in an earlier configuration. Include per-seed curves from phase1_log.json.]

## Appendix C: Full per-seed behavioral tables
[From headline_summary.json — 12 branch-seed rows × 4 regions × fire/clean.]

## Appendix D: Per-layer instrument profiles
[Probe transfer, cosine, CKA, and patching preservation by layer; currently mid-third means are in §5. Data in analysis/headline_report.json and patching_results.json.]
