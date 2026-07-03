# Stage 2 — Tier 0 Findings

Running document, one section per Tier 0 item as it completes. Per the
master prompt's rigor rule: hypothesis stated before evidence, verdict
classified as VERIFIED / SUPPORTED / PRELIMINARY / REJECTED, "consistent
with"/"provides evidence for" rather than "confirms"/"proves" unless a
genuine independent replication exists.

---

## Item 1 — Layer-wise conditioning magnitude/direction

**Status: run for the first time this session.** The script
(`Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/
layerwise_direction_probe.py`) was written during Stage 1 but had never
actually been executed — zero output artifacts existed anywhere before
this run (checked directly, not assumed). Run against the real
`outputs/models/diffusion_best.pt` (`exp1_baseline_reproduction`'s
checkpoint), wrapped in `ExperimentLogger`
(`stage2_tier0_item1_layerwise_magnitude_direction`, Stage 2 ledger).
13.4s elapsed, pure forward-pass inference, no training.

### Hypothesis (stated before this run)

A prior, not-yet-independently-confirmed claim held that conditioning
magnitude decays from ~0.91 (block 1) to ~0.24 (block 6) while direction
consistency stays ~1.00 across the model's 6 Transformer blocks. Confirm
result: magnitude should show a clear downward trend across blocks with
direction consistency staying high throughout. Reject result: no
consistent magnitude trend, or direction consistency dropping
substantially at any block (would indicate class information becomes
noise-shaped, not just attenuated).

### Method (see script docstring for full derivation) — answering the methodology review directly

**Sample size / averaging (this was the single most important gap
flagged): NOT n=1.** For each of 5 class pairs (0→1 through 0→5, i.e.
NORM vs. each of MI/STTC/CD/HYP/OTHER), the script draws K=20
independent noise samples. For each draw, ONE fixed `x_t` is forward-passed
through the model TWICE — once with class label A=0, once with label B —
so the pair is a genuinely paired comparison (same noise, label swapped),
not two independent samples. `magnitude`/`direction_consistency` at each
layer are computed by aggregating over these 20 paired draws, then the
headline table further averages over the 5 class pairs. So each number
below reflects 20 draws × 5 pairs = 100 underlying paired forward-pass
comparisons, not a single measurement.

**Real cross-pair variance** (computed directly from the already-saved
per-pair breakdown in `layerwise_probe_raw_baseline.json` — this required
no new run):

| Layer | Magnitude: mean ± std (n=5 pairs) | range | Direction consistency: mean ± std | range |
|---|---|---|---|---|
| 1 | 0.1237 ± 0.0168 | 0.1072–0.1550 | 0.99996 ± 0.00001 | 0.99994–0.99998 |
| 2 | 0.0644 ± 0.0149 | 0.0483–0.0903 | 0.99986 ± 0.00004 | 0.99982–0.99992 |
| 3 | 0.0505 ± 0.0138 | 0.0355–0.0736 | 0.99974 ± 0.00008 | 0.99964–0.99987 |
| 4 | 0.0497 ± 0.0151 | 0.0345–0.0763 | 0.99969 ± 0.00009 | 0.99959–0.99984 |
| 5 | 0.0548 ± 0.0192 | 0.0339–0.0895 | 0.99917 ± 0.00022 | 0.99890–0.99949 |
| 6 | 0.0487 ± 0.0194 | 0.0276–0.0839 | 0.99754 ± 0.00055 | 0.99689–0.99842 |

Every one of the 5 individual class pairs shows the same qualitative
shape (large layer1→2 drop, smaller changes after) — this is not one
pair driving the average.

**Honest gap that remains:** the within-pair spread across the 20 raw
draws was never persisted to disk by the script (only the already-averaged
per-pair number was saved) — so I can report cross-pair (n=5) variance
exactly as above, but not a full within-pair (n=20) variance decomposition,
without rerunning with the script modified to dump raw per-draw values.
Not done in this pass; flagging rather than glossing over it.

**Timestep handling — was single-fixed at t=500, now tested at 3
timesteps.** The original run held t=500 fixed throughout (confirmed via
the actual log line: `"Probing at t=500"`) — satisfying Stage 2.1's rule
against mixing samples from different t values within one calculation.
But that means the original report only established the shape holds *at
one timestep*. Added a `--timestep-frac` CLI argument to the script
(default unchanged at 0.5, verified byte-identical output against the
original run before using it for anything else — see commit) and reran
at t=100 (high noise, early reverse-process step) and t=900 (low noise,
near-clean signal):

| | Layer 1→2 ratio (the big early drop) | Layer 2→6 ratio (later blocks) | Layer 1→6 ratio (overall) |
|---|---|---|---|
| t=100 | 0.487 | 0.471 | 0.229 |
| t=500 | 0.521 | 0.756 | 0.394 |
| t=900 | 0.605 | 0.639 | 0.386 |

**Correction to my own first-pass characterization:** at t=500 alone, I
described blocks 2–6 as "roughly flat" — that was an overstatement of
what t=500 specifically showed, and it does NOT hold at the other two
timesteps: at both t=100 and t=900, there is real, additional decay
through blocks 3–6 (layer2→6 ratios of 0.47 and 0.64, i.e. another
36–53% loss after the initial drop), not a flat plateau. **What IS robust
across all three timesteps: the single largest drop consistently happens
between block 1 and block 2** (ratio 0.49–0.61 at every timestep tested)
— that part of the front-loaded characterization holds up under
scrutiny. The "then flat" part does not; there's a smaller but real
second wave of attenuation continuing through the later blocks, whose
size is timestep-dependent.

Direction consistency stays above 0.99 at every layer, at all three
timesteps (t=100: 0.99993→0.99702; t=500: 0.99996→0.99754; t=900:
0.99993→0.99217) — this part of the finding is now on considerably
stronger footing than the original single-timestep report: 3 timesteps ×
5 class pairs × 20 draws all agree.

**Exact definition of "magnitude" (requested explicitly):** it is the L2
norm of the delta in each block's **mean-pooled hidden-state output**
(the residual-stream token representations after that TransformerBlock,
averaged over the 600 tokens), between class-A and class-B forward
passes with identical noise, normalized by the mean norm of the class-A
hidden state at that same layer. **This is NOT the AdaLN modulation
output norm** — that is a different quantity Item 2 will compute
separately (`gain_i = ||adaLN_output_i|| / ||adaLN_output_1||`, per the
master prompt's own Item 2 spec). Worth keeping these two distinct when
interpreting Item 2's results against Item 1's.

### Results (averaged across all 5 class pairs vs. NORM)

| Layer (block) | Magnitude | Direction consistency |
|---|---|---|
| 1 | 0.1237 | 0.99996 |
| 2 | 0.0644 | 0.99986 |
| 3 | 0.0505 | 0.99974 |
| 4 | 0.0497 | 0.99969 |
| 5 | 0.0548 | 0.99917 |
| 6 | 0.0487 | 0.99754 |

Full per-class-pair breakdown in
`Roadmap/Stage_1_Diagnosis/Outputs/Experiment_3_Directional_Probe/
layerwise_probe_raw_baseline.json` (untracked, extracted-archive-local
per repo policy).

### What matches the prior claim, and what doesn't — stated precisely, not reconciled to fit

**Direction consistency: SUPPORTED, close to VERIFIED.** Stays
essentially at 1.0 through every block (0.99996 → 0.99754), matching the
prior claim's "~1.00" closely. Only a small, real decline at the final
block (0.25% below 1.0) — still overwhelmingly high, no evidence of
class information becoming noise-shaped at any block tested.

**Magnitude decay: SUPPORTED in *direction*, but the specific numbers
and shape do NOT match the prior claim.**
- Overall decay ratio (layer 6 / layer 1) measured here: **0.394** — the
  prior claim's ratio was 0.24/0.91 ≈ **0.264**. Same direction (real
  decay exists), different magnitude of the effect.
- **The decay is front-loaded, not smooth or progressive.** The dominant
  drop is entirely between block 1 → block 2 (ratio 0.521 — over half
  the total decay happens in this one transition). From block 2 onward,
  magnitude is roughly flat with mild fluctuation (0.0644 → 0.0505 →
  0.0497 → 0.0548 → 0.0487; block 5 actually ticks back *up* slightly
  before block 6 drops again) — not a steady progressive decline through
  all 6 blocks the way "0.91 → 0.24" reads on its own. Block 2→6 ratio
  is 0.756 — most of the "remaining" signal after block 1 survives
  essentially intact.
- The absolute values (~0.12 → ~0.05 here vs. ~0.91 → ~0.24 previously)
  are on a different scale entirely — could reflect a different
  normalization, probe timestep, or aggregation method in whatever
  produced the original number, which I have no artifact for and can't
  reconcile further. Not claiming this run reproduces that one; it
  measures the same *quantity*, independently, and gets a materially
  different shape and scale.

### Verdict — updated after the methodology review

**SUPPORTED** (upgraded from an initial PRELIMINARY-leaning-SUPPORTED
pending exactly the checks below). Not VERIFIED — this is one method
(this script) applied more thoroughly (5 class pairs, 20 draws each, 3
timesteps), not an independently-implemented replication. But the
methodology gaps raised in review are now answered with real data, not
assumptions:
- **n=1 concern: refuted.** Confirmed 20 draws × 5 pairs = 100 paired
  comparisons per headline number, with real cross-pair variance reported
  above, not a single measurement.
- **Timestep concern: addressed, and it changed the finding.** The
  original single-timestep (t=500) report overstated a "flat plateau
  after block 2" — that doesn't hold at t=100 or t=900. What DOES hold
  at all 3 timesteps tested: the single largest attenuation is
  consistently the block1→2 transition, with a smaller, timestep-dependent
  second wave of decay continuing through the later blocks.
- **Definition of magnitude: clarified** — mean-pooled hidden-state delta
  at each block's output, not the AdaLN modulation output (a distinct
  quantity Item 2 computes).

What this now independently establishes: class-conditioning signal in
this checkpoint is (a) real and (b) direction-stable at every layer,
robust across 3 timesteps and 5 class pairs; (c) attenuated in a
front-loaded but NOT purely one-time pattern — the biggest single loss
is at block1→2, consistently, but real additional decay continues
afterward at a magnitude that depends on timestep.

**Implication for Item 2 (LayerScale hypothesis) — revised given the
corrected shape.** The original "one-time absorption, not a distributed
leak" framing from my first pass was too strong given the 3-timestep
data: there IS a real, continuing (if smaller) decay after block 2, not
a flat plateau. This means a *purely* localized single-gain-at-block1→2
fix would likely under-correct — it would address the majority of the
attenuation but not the smaller, real, ongoing loss through blocks 3–6.
Per the reviewer's recommendation, Item 2 should test **both variants as
separately falsifiable hypotheses**, not choose one in advance:
  1. **Uniform per-block gain** (6 learnable scalars, the original spec) —
     motivated by the real, if smaller, continuing decay in later blocks.
  2. **Localized gain at the block1→2 transition specifically** (1
     learnable scalar) — motivated by that transition consistently being
     the single largest loss at every timestep tested.
Both are cheap (few parameters); logging both lets the evidence pick
between "mostly-uniform-fix-suffices" and "mostly-localized-fix-suffices"
rather than assuming either.

**Tier 0 item list correction (raised in review, confirmed):** CFG and
the sensitivity probe are already-completed Stage 1 evidence (CFG's
negative result: collapse fraction rose 59.2%→75.2% as guidance scale
increased 1→5, already documented) — not remaining Tier 0 items. The
operative Tier 0 list remains the 9 items in
`Roadmap/Stage_2_Architecture_Investigation/Stage2_Master_Prompt.md`
(layer-wise magnitude/direction; LayerScale/gain test; residual-path
attenuation; gradient norms; AdaLN/FiLM parameter statistics; attention
entropy; class-embedding evolution; representation collapse; collapse-
vs-scale checkpoint sweep). I have not at any point proposed CFG,
embedding-behavior, or a re-run of the sensitivity probe as new Tier 0
items — continuing with the correct list, Item 2 next once reviewed.

**Not yet run:** Items 2-9. Stopping here again for final sign-off before
starting Item 2, given the scope of Item 2 itself just changed (both
gain variants, not one) based on this update.
