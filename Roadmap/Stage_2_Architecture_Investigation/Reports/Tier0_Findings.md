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

### Method (see script docstring for full derivation)

Fixed noise `x_t`, timestep t=500, K=20 independent draws. For each draw,
forward pass with class label 0 vs. class label `c` (c=1..5, i.e. NORM
vs. each of MI/STTC/CD/HYP/OTHER), hooked at every block's output,
mean-pooled over the 600 tokens. `magnitude_k` = mean delta norm /
mean base norm at layer k (normalized, not raw). `direction_consistency_k`
= mean cosine similarity between each draw's delta and the mean delta
across draws — near 1.0 means the class label pushes that layer in a
*stable* direction regardless of noise; near 0 means noise-shaped.

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

### Verdict

**SUPPORTED**, not VERIFIED, and not a replication of the specific prior
numbers. What this run independently establishes on its own evidence:
class-conditioning signal in this checkpoint is (a) real (nonzero,
direction-stable magnitude change exists at every block), (b) most
strongly attenuated in the very first block-to-block transition rather
than continuously through the network, and (c) direction-stable
throughout — the model isn't "forgetting" which way to push for a given
class, it's the *size* of the push that shrinks, mostly at the front of
the stack.

**Implication for Item 2 (LayerScale hypothesis), stated as a caveat
before that item runs:** the master prompt frames Item 2's premise as "if
Item 1 confirms magnitude decay with preserved direction" — technically
true here, but a per-block *learnable gain* (the LayerScale fix) is
motivated by decay that recurs at *every* block from repeated
LayerNorm renormalization. What's actually measured is concentrated at
one specific transition (block 1→2), which is a different failure
shape than "every block independently attenuates a bit." Worth checking
directly in Item 2 whether the AdaLN gain profile shows the same
front-loaded pattern (would still support LayerScale, just targeted
differently — e.g. a bigger fix at block 2's input than uniform gain
across all 6) or a genuinely uniform per-block pattern that this
metric's front-loading doesn't reflect.

**Not yet run:** Items 2-9. Per the reviewer's explicit request, stopping
here for a methodology check before treating this as settled evidence,
rather than proceeding to Item 2.
