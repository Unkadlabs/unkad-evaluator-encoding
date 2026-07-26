# Modal spend log (spec §11 rule 4 — one line per launch)

| Date | Job | GPU-hrs (est) | Cost (est) | Credit remaining (est) |
|---|---|---|---|---|
| — | starting credit | — | — | $251.00 |
| 2026-07-18 | pilot phase1 attempt 1 (lr 5e-5 — chance level, gate stopped it) | ~0.3 A10G | ~$0.35 | ~$250.65 |
| 2026-07-18 | pilot phase1 attempt 2 (lr 2e-4 — still chance; root cause was 2-digit task, found via free local 70M diag) | ~0.35 A10G | ~$0.40 | ~$250.25 |
| 2026-07-18 | pilot phase1 attempt 3 (single-digit — overall PASS 0.95/0.93, worst cell T7 0.56 failed gate; T7 template fixed) | ~0.35 A10G | ~$0.40 | ~$249.85 |
| 2026-07-18 | pilot phase1 attempts 4 (killed by network) + 5 (ln2 saddle, gate stopped) | ~0.6 A10G | ~$0.65 | ~$249.20 |
| 2026-07-18 | attempt 6 lottery killed by client crash + app teardown (~20 min x 3 A10G lost) | ~1.0 A10G | ~$1.10 | ~$248.10 |
| 2026-07-18 | pilot run 1 full: lottery (seed0 fail, seeds 1+2 PASS 0.996/0.987) + 6 branches collapsed at phase2 lr 2e-4 + 7 caches | ~3.3 A10G | ~$3.60 | ~$244.50 |
| 2026-07-18 | pilot run 2: branches trained OK at phase2_lr 2e-5; behavior/caches invalid (tempfile config bug — measured on wrong data) | ~2.2 A10G | ~$2.40 | ~$242.10 |
| 2026-07-18 | pilot run 3: remeasure + recache with config fix (training skipped) | ~0.5 A10G | ~$0.55 | ~$241.55 |
| 2026-07-18 | D4 pressure sweep: branch A x 3 variants (2e-5x12 best: covered 0.49, blind 0.98-1.0) | ~1.7 A10G | ~$1.90 | ~$239.65 |
| 2026-07-19 | tune2 continuation: equilibrium confirmed (blind 1.0/covered 0.49 at both LRs); recipe frozen = drop @ 2e-5 x 12 | ~1.7 A10G | ~$1.90 | ~$237.75 |
| 2026-07-19 | HEADLINE RUN launched: 4 branches x 3 seeds @ frozen recipe + 12 caches (c0 cache reused) | ~11.5 A10G est | ~$12.70 est | ~$225 est |
| 2026-07-19 | patching pairs (18) on volume | ~0.9 A10G | ~$1.00 | ~$224 |
| 2026-07-19 | HEADLINE COMPLETE. Pre-registered verdict: 0/3 diverged -> H0. Patching trend: p=0.0079 but sub-3σ | — | — | ~$224 |
