# Stage 2 — Tier 0 Findings

Running document, one section per Tier 0 item as it completes. Per the
master prompt's rigor rule: hypothesis stated before evidence, verdict
classified as VERIFIED / SUPPORTED / PRELIMINARY / REJECTED, "consistent
with"/"provides evidence for" rather than "confirms"/"proves" unless a
genuine independent replication exists.

---

## Item 1 — Layer-wise conditioning magnitude/direction

**Status: run for the first time this session, then re-run to answer two
rounds of review.** The script was written during Stage 1 but had never
actually been executed before this session — zero output artifacts
existed anywhere before this run (checked directly, not assumed). Run
against the real `outputs/models/diffusion_best.pt`
(`exp1_baseline_reproduction`'s checkpoint), each pass wrapped in
`ExperimentLogger`. Three logged ledger entries exist for this item:
`stage2_tier0_item1_layerwise_magnitude_direction` (+ `_t0100`/`_t0900`,
the first pass, aggregated numbers only), then superseded for analysis
purposes by `..._baseline_v2` (+ `_t0100_v2`/`_t0900_v2`), which added
per-draw persistence to close the within-pair variance gap flagged in
sign-off review. All `_v2` runs are byte-identical to their non-`_v2`
counterparts on every aggregated number (regression-tested via diff).
~13s per run, pure forward-pass inference, no training. Script now lives
at `Roadmap/Stage_2_Architecture_Investigation/Code/
stage2_tier0_item1_layerwise_magnitude_direction/
layerwise_direction_probe.py` (derived copy; see code-provenance note in
the Verdict section below) — the Stage 1 original is unmodified.

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
per-pair breakdown in `layerwise_probe_raw_baseline.json`):

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

**Gap closed (previously flagged, now fixed):** the within-pair spread
across the 20 raw draws was originally never persisted to disk — only
the already-averaged per-pair number was saved. Modified the script to
also dump `magnitude_per_layer_per_draw` / `direction_consistency_per_layer_per_draw`
(shape `n_layers × k_draws`), regression-tested against the un-modified
output (aggregated numbers byte-identical, confirmed via diff before
trusting the new field), then reran baseline + both timestep variants
so a real within-pair (n=20) variance estimate now exists for every
cell below, not just the cross-pair one.

**What the 20 draws actually are, precisely (requested explicitly):**
for a fixed (class-pair, timestep), each of the 20 draws is an
*independent Gaussian noise seed* (`torch.manual_seed(1000+draw)`,
`x_t = torch.randn(...)`) — **not** 20 different real ECG samples. There
is no real PTB-XL signal anywhere in this probe; `x_t` is synthetic noise
at a fixed timestep, forward-passed once per class label with everything
else held fixed. So the within-draw std reported below measures the
model's sensitivity to the random noise seed at a fixed synthetic input
and fixed class pair/timestep — it is not sample-to-sample biological
variance, and should not be read as such. This is exactly why direction
consistency's within-pair std is so small (≤0.006 even at its highest,
block 6/t=900): the model's response to a class-label swap is remarkably
stable to the noise seed itself; the real source of spread in the
headline numbers is cross-pair (which class is being compared to NORM),
not cross-draw.

**Limitation, stated explicitly so this isn't conflated later:** this
design isolates *stochastic* (noise-seed) variance at fixed
(class-pair, timestep), which is what the std columns above report. It
does **not** capture *sample*-level variance — whether different real
ECG examples belonging to the same class would show the same
attenuation pattern, or whether it depends on which representative input
was used. This probe never touches a real ECG signal (see above), so
that question is simply out of scope for this measurement, not answered
by it. A future variant that hooks real per-sample activations (rather
than pure-noise `x_t`) would be needed to close this specific gap; not
planned as part of Tier 0 unless a later item's results make it
necessary.

**Full table — all 90 magnitude values (5 pairs × 3 timesteps × 6
blocks), mean ± std over the 20 within-pair noise-seed draws:**

| Pair | t | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| 0→1 | 100 | 0.1772±0.0006 | 0.0886±0.0002 | 0.0729±0.0002 | 0.0774±0.0003 | 0.0792±0.0003 | 0.0384±0.0003 |
| 0→1 | 500 | 0.1257±0.0003 | 0.0678±0.0001 | 0.0568±0.0001 | 0.0543±0.0001 | 0.0587±0.0004 | 0.0534±0.0005 |
| 0→1 | 900 | 0.1094±0.0002 | 0.0642±0.0001 | 0.0578±0.0001 | 0.0560±0.0002 | 0.0515±0.0003 | 0.0450±0.0004 |
| 0→2 | 100 | 0.1569±0.0005 | 0.0664±0.0002 | 0.0487±0.0001 | 0.0540±0.0002 | 0.0576±0.0004 | 0.0260±0.0004 |
| 0→2 | 500 | 0.1127±0.0003 | 0.0515±0.0001 | 0.0381±0.0001 | 0.0366±0.0001 | 0.0424±0.0005 | 0.0376±0.0005 |
| 0→2 | 900 | 0.1047±0.0002 | 0.0538±0.0001 | 0.0455±0.0001 | 0.0443±0.0003 | 0.0419±0.0003 | 0.0346±0.0004 |
| 0→3 | 100 | 0.1669±0.0006 | 0.0833±0.0004 | 0.0617±0.0002 | 0.0672±0.0002 | 0.0666±0.0002 | 0.0286±0.0002 |
| 0→3 | 500 | 0.1178±0.0003 | 0.0643±0.0002 | 0.0486±0.0001 | 0.0466±0.0001 | 0.0496±0.0004 | 0.0411±0.0004 |
| 0→3 | 900 | 0.1110±0.0003 | 0.0734±0.0002 | 0.0600±0.0002 | 0.0580±0.0003 | 0.0503±0.0004 | 0.0422±0.0006 |
| 0→4 | 100 | 0.1520±0.0006 | 0.0679±0.0003 | 0.0462±0.0002 | 0.0495±0.0002 | 0.0495±0.0003 | 0.0214±0.0002 |
| 0→4 | 500 | 0.1072±0.0003 | 0.0483±0.0001 | 0.0355±0.0001 | 0.0345±0.0001 | 0.0339±0.0003 | 0.0276±0.0004 |
| 0→4 | 900 | 0.0835±0.0002 | 0.0459±0.0001 | 0.0349±0.0001 | 0.0327±0.0002 | 0.0317±0.0003 | 0.0236±0.0002 |
| 0→5 | 100 | 0.2117±0.0008 | 0.1147±0.0004 | 0.1031±0.0004 | 0.1342±0.0007 | 0.1401±0.0010 | 0.0838±0.0031 |
| 0→5 | 500 | 0.1550±0.0004 | 0.0903±0.0002 | 0.0736±0.0002 | 0.0763±0.0003 | 0.0895±0.0008 | 0.0839±0.0021 |
| 0→5 | 900 | 0.1277±0.0003 | 0.0870±0.0003 | 0.0804±0.0002 | 0.0818±0.0006 | 0.0769±0.0007 | 0.0618±0.0007 |

**Full table — direction consistency, same 90 cells:**

| Pair | t | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| 0→1 | 100 | 0.99994±0.00001 | 0.99979±0.00003 | 0.99966±0.00005 | 0.99968±0.00006 | 0.99944±0.00014 | 0.99786±0.00136 |
| 0→1 | 500 | 0.99998±0.00001 | 0.99992±0.00002 | 0.99987±0.00003 | 0.99984±0.00005 | 0.99949±0.00020 | 0.99842±0.00075 |
| 0→1 | 900 | 0.99995±0.00001 | 0.99985±0.00003 | 0.99973±0.00007 | 0.99949±0.00020 | 0.99777±0.00067 | 0.99340±0.00364 |
| 0→2 | 100 | 0.99994±0.00001 | 0.99968±0.00006 | 0.99937±0.00014 | 0.99946±0.00008 | 0.99919±0.00017 | 0.99648±0.00163 |
| 0→2 | 500 | 0.99997±0.00001 | 0.99985±0.00003 | 0.99970±0.00006 | 0.99965±0.00009 | 0.99919±0.00030 | 0.99689±0.00188 |
| 0→2 | 900 | 0.99995±0.00001 | 0.99978±0.00004 | 0.99960±0.00009 | 0.99931±0.00026 | 0.99784±0.00074 | 0.99326±0.00349 |
| 0→3 | 100 | 0.99994±0.00001 | 0.99976±0.00004 | 0.99952±0.00011 | 0.99956±0.00008 | 0.99928±0.00017 | 0.99668±0.00254 |
| 0→3 | 500 | 0.99997±0.00001 | 0.99989±0.00002 | 0.99979±0.00005 | 0.99974±0.00010 | 0.99933±0.00028 | 0.99792±0.00098 |
| 0→3 | 900 | 0.99994±0.00001 | 0.99986±0.00003 | 0.99963±0.00011 | 0.99924±0.00036 | 0.99652±0.00158 | 0.98905±0.00583 |
| 0→4 | 100 | 0.99992±0.00002 | 0.99960±0.00008 | 0.99916±0.00016 | 0.99922±0.00014 | 0.99877±0.00029 | 0.99576±0.00162 |
| 0→4 | 500 | 0.99996±0.00001 | 0.99982±0.00004 | 0.99964±0.00007 | 0.99959±0.00012 | 0.99896±0.00040 | 0.99720±0.00114 |
| 0→4 | 900 | 0.99991±0.00002 | 0.99972±0.00006 | 0.99948±0.00012 | 0.99928±0.00027 | 0.99835±0.00048 | 0.99409±0.00178 |
| 0→5 | 100 | 0.99987±0.00004 | 0.99957±0.00009 | 0.99942±0.00012 | 0.99953±0.00007 | 0.99916±0.00017 | 0.99834±0.00079 |
| 0→5 | 500 | 0.99994±0.00002 | 0.99984±0.00004 | 0.99971±0.00007 | 0.99963±0.00008 | 0.99890±0.00041 | 0.99727±0.00100 |
| 0→5 | 900 | 0.99992±0.00003 | 0.99981±0.00003 | 0.99953±0.00013 | 0.99908±0.00046 | 0.99724±0.00110 | 0.99107±0.00413 |

Raw per-draw data underlying every cell above lives in
`layerwise_probe_raw_{baseline,t0100,t0900}.json`'s new
`magnitude_per_layer_per_draw` / `direction_consistency_per_layer_per_draw`
fields (untracked, extracted-archive-local per repo policy).

**Statistical test — is block1→2 actually the dominant transition, or
does it just look that way?** Treating each of the 15 (pair, timestep)
combinations as one paired observation, computed the ratio
`layer_(k+1)_mean / layer_k_mean` for every consecutive-block transition
k=1..5:

| Transition | Ratio: mean ± std (n=15) |
|---|---|
| block1→2 | 0.5319 ± 0.0753 |
| block2→3 | 0.8006 ± 0.0724 |
| block3→4 | 1.0252 ± 0.0945 |
| block4→5 | 1.0154 ± 0.0853 |
| block5→6 | 0.7237 ± 0.1883 |

Ran a one-sided Wilcoxon signed-rank test (paired, n=15, appropriate
given the small sample and no normality assumption) for "block1→2's
ratio is smaller (i.e. a bigger drop) than transition X's ratio":

| Comparison | Diff in mean ratio | Wilcoxon W | p-value | Paired Cohen's d |
|---|---|---|---|---|
| block1→2 vs block2→3 | +0.269 | 0.0 | 0.00003 | −4.83 |
| block1→2 vs block3→4 | +0.493 | 0.0 | 0.00003 | −3.77 |
| block1→2 vs block4→5 | +0.484 | 0.0 | 0.00003 | −3.57 |
| block1→2 vs block5→6 | +0.192 | 8.0 | 0.00076 | −1.17 |

**This refines, not just confirms, the earlier characterization.** Under
the current probe, at this checkpoint, block1→2 is significantly the
largest single drop vs. every other transition (all p<0.001, large
effect sizes) — that qualifier matters and is carried through the rest
of this section deliberately: Item 9 (checkpoint-scale sweep, reusing
exp2's six per-size checkpoints) is specifically designed to test
whether this shape is a property of the architecture or a property of
*this particular trained model*. Calling block1→2 "the bottleneck"
outright would claim architecture-level generality that hasn't been
earned yet; "the dominant observed attenuation point under the current
probe, at this checkpoint" is the honest scope until Item 9 runs. But
the shape of what
happens *after* block 2 is more structured than "flat, then a diffuse
second wave through 3–6," which is what I said in the first pass:
block2→3 shows a smaller but real additional decrease (ratio 0.80),
block3→4 and block4→5 are both ~flat-to-slightly-increasing (ratios
1.02–1.03, i.e. **no further loss, statistically indistinguishable from
1**), and **block5→6 is a second real, significant drop** (ratio 0.72,
p=0.0008 vs. block1→2, though with a notably smaller effect size and
much higher variance — std 0.188 vs. 0.075 — than block1→2's).
Correcting my own prior "second wave through blocks 3–6" language: the
continuing decay isn't smoothly distributed across blocks 3–6 — it's
concentrated at two specific transitions (block1→2, dominant; block5→6,
smaller and less consistent), with blocks 3→4→5 essentially flat in
between.

**Timestep handling — was single-fixed at t=500, now tested at 3
timesteps, with the pooled statistical test above (n=15 = 5 pairs × 3
timesteps) as the operative analysis rather than eyeballing three
separate per-timestep plots.** The original run held t=500 fixed
throughout (confirmed via the actual log line: `"Probing at t=500"`) —
satisfying Stage 2.1's rule against mixing samples from different t
values within one calculation. Added a `--timestep-frac` CLI argument
(default unchanged at 0.5, verified byte-identical output against the
original run before using it for anything else) and reran at t=100
(high noise) and t=900 (low noise, near-clean signal). The Wilcoxon
test above already pools all 3 timesteps and confirms block1→2 is
significantly the dominant transition regardless of timestep (that IS
what "timestep-robust" means here — a test, not a visual impression
across plots). Per-timestep-only summary, for reference:

| | Layer 1→2 ratio | Layer 2→6 ratio | Layer 1→6 ratio |
|---|---|---|---|
| t=100 | 0.487 | 0.471 | 0.229 |
| t=500 | 0.521 | 0.756 | 0.394 |
| t=900 | 0.605 | 0.639 | 0.386 |

**Correction to my own first-pass characterization, now precisely
stated:** at t=500 alone, I described blocks 2–6 as "roughly flat" —
that was an overstatement, and the pooled test shows exactly where it
breaks: block2→3 (ratio 0.80) and block5→6 (ratio 0.72) are both real,
statistically significant drops relative to a flat line at 1.0, while
block3→4 and block4→5 (ratios 1.02–1.03) are not distinguishable from
no further loss. So "then flat" was wrong not because decay is smoothly
distributed through 3–6 (a diffuse "second wave"), but because there
are two additional discrete drop points (2→3 and 5→6) bracketing a
genuinely flat middle (3→4→5). block1→2 remains the single dominant
transition at every timestep and by a wide statistical margin.

Direction consistency stays above 0.99 at every layer, at all three
timesteps (t=100: 0.99994→0.99648–0.99834; t=500: 0.99998→0.99689–0.99842;
t=900: 0.99995→0.98905–0.99409, per the full 90-cell table above) — this
part of the finding is on considerably stronger footing than the
original single-timestep report: 3 timesteps × 5 class pairs × 20 draws
all agree, with within-pair std never exceeding 0.006 even at the noisiest
cell (block 6, t=900).

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

**Magnitude decay: SUPPORTED in *direction*, with a statistically-tested
shape that is more structured than either the prior claim or my own
first-pass description.**
- Overall decay ratio (layer 6 / layer 1) measured here: **0.394** — the
  prior claim's ratio was 0.24/0.91 ≈ **0.264**. Same direction (real
  decay exists), different magnitude of the effect. The absolute values
  (~0.12 → ~0.05 here vs. ~0.91 → ~0.24 previously) are on a different
  scale entirely — could reflect a different normalization, probe
  timestep, or aggregation method in whatever produced the original
  number, which I have no artifact for and can't reconcile further. Not
  claiming this run reproduces that one; it measures the same quantity,
  independently, and gets a materially different shape and scale.
- **The decay is front-loaded and statistically confirmed, not
  eyeballed.** The Wilcoxon test above (n=15, pooling all 5 pairs × 3
  timesteps) shows block1→2 is significantly the largest single drop
  vs. every other transition (all p≤0.0008, Cohen's d from −1.17 to
  −4.83). This is now a tested claim, not a visual impression from one
  averaged curve.
- **What happens after block 2 is two discrete drops around a flat
  middle, not a smooth "second wave."** block2→3 (ratio 0.80) and
  block5→6 (ratio 0.72) are both real, significant decreases; block3→4
  and block4→5 (ratios 1.02–1.03) are statistically indistinguishable
  from no further loss. My first-pass "roughly flat after block 2" was
  an overstatement (it ignored block2→3 and block5→6); my
  second-pass "diffuse second wave through blocks 3–6" was also
  imprecise (it implied smooth distributed decay when the data show two
  sharp, localized transitions bracketing a genuinely flat span).

### Verdict — updated after the sign-off review's request for raw data and a real test

**SUPPORTED**, now on materially stronger footing than the prior pass:
not just more class pairs and timesteps, but an actual within-pair
variance estimate (previously an acknowledged gap, now closed) and a
paired significance test (previously eyeballed ratios). Still not
VERIFIED — one method (this script) applied thoroughly, not an
independently-implemented replication.
- **n=1 concern: refuted**, with the full 90-value table now shown
  in-line above (not described), plus the within-pair (n=20 noise
  seeds) variance that was previously missing.
- **"Real samples or seeds?" clarified precisely:** the 20 draws are
  independent Gaussian noise seeds at a fixed (class-pair, timestep),
  not 20 different real ECG signals — there is no real PTB-XL data
  anywhere in this probe. The within-seed std reported is the model's
  sensitivity to the noise seed, not sample-to-sample biological
  variance.
- **"Dominant transition" is now a statistical claim, not a description
  — scoped to this checkpoint, under this probe.** Wilcoxon signed-rank
  test, n=15 paired observations, block1→2 vs. each other transition:
  p≤0.0008 in every comparison, with block1→2 the single
  largest-magnitude effect (d=−4.83 vs. block2→3, down to d=−1.17 vs.
  the weaker but still real block5→6 drop). Whether this generalizes
  across checkpoints of different training scale is Item 9's question
  (reusing exp2's six per-size checkpoints), not yet tested — "dominant"
  here means "in this trained model," not "architecturally inevitable."
- **Definition of magnitude: clarified** — mean-pooled hidden-state delta
  at each block's output, not the AdaLN modulation output (a distinct
  quantity Item 2 computes).

What this now independently establishes, with statistical backing, **for
this checkpoint under this probe** (scope qualifier deliberate — see
above): (a) class-conditioning signal is real and direction-stable at
every layer, robust across 3 timesteps and 5 class pairs, with the
model's noise-seed sensitivity distinct from (much smaller than)
cross-pair variance, though noise-seed variance is not the same thing as
real-sample variance (limitation noted above); (b) attenuation is
concentrated at two specific, statistically confirmed transitions —
block1→2 (dominant, large effect) and block5→6 (real but smaller effect,
higher variance) — with blocks 3→4→5 flat in between, not a smoothly
distributed leak. Whether this two-drop shape is an architectural
property or specific to this trained model is exactly Item 9's question,
not yet answered.

**Implication for Item 2 (LayerScale hypothesis) — revised given the
now-tested two-drop shape.** A *purely* localized single-gain-at-block1→2
fix would address the dominant loss but leave the smaller, real,
statistically-confirmed block5→6 drop uncorrected. Item 2 tests **both
variants as separately falsifiable hypotheses**, per the reviewer's
explicit spec:
  1. **`stage2_tier0_item2_uniform_gain`** — uniform per-block gain (6
     learnable scalars, the original spec).
  2. **`stage2_tier0_item2_localized_gain`** — localized gain at the
     block1→2 transition specifically (1 learnable scalar).
Logged as two separate ledger entries (not one entry with two folded-in
conditions), same 5 class pairs / 3 timesteps / n≥20 draws methodology
as this corrected Item 1 run, so results are directly comparable.

## Item 2 — Pre-Registration v3 (final revision before implementation)

Two prior drafts were reviewed before any code was written (per the
standing "no code until this is in writing" hold). v1 had a fatal design
flaw; v2 fixed it but left Δ's exact definition, the hook location, and
several methodological choices in prose rather than precise, lockable
form. v3 closes all of that. **No code has been implemented for Item 2
as of this revision** — this remains pre-registration only.

### Correction to the reviewer's citation, checked directly

The review cited a quote attributed to "the project's own Experiment 1
(embedding initialization scale)" about LayerNorm removing magnitude
information at every block boundary. **Checked directly: no such
experiment or reproducible finding exists in this repository.** Per
`Roadmap/Stage_0_Pipeline_Audit/Reports/Pipeline_Code_Audit.md:604-617`,
the entire "Investigation Timeline" narrative this quote traces back to
(embedding-scale experiment, AdaLN-Zero, conditioning-collapse
percentages, the AFIB-attractor findings) is explicitly flagged as
**historical narrative, contradicted by direct evidence, not to be cited
as motivation for Stage 2 priority ordering** — that audit finding
already exists in this repo and predates this conversation. So the
specific quote is not something I can treat as established fact.

**This does not dissolve the concern — it independently confirms it,
more rigorously than the quote did.** Read the actual model source,
`step04_transformer_diffusion.py`, class `TransformerBlock` (line 146),
`forward()` (lines 171–177):

```
x → norm1=LayerNorm(x) → modulate(norm1, shift1, scale1) → attn → x = x + attn_out   [L173-175]
x → norm2=LayerNorm(x) → modulate(norm2, shift2, scale2) → ff   → x = x + ff_out     [L176]
return x   [L177 -- this is what the probe's forward hook captures: post-residual, never a bare LN output]
```

`shift/scale` come from `self.adaLN(cond)` (L169) — a function of the
class+timestep embedding only, **independent of the incoming hidden
state's scale**. Confirmed pre-norm: block N+1's `forward` immediately
calls its own `norm1(x)` on the raw residual stream handed over from
block N (no LN at block N's tail) — so **every block's input is
renormalized fresh**, exactly the mechanism the (unverified-provenance)
quote described, but now confirmed from source rather than cited from a
document this project has already flagged as unreliable.

**Mathematical consequence, provable without running anything —
corrected wording (per review, the "exactly" claim was overstated):**
standard LayerNorm computes `(x−μ)/√(σ²+ε)·γ+β`. Under `x → g·x`
(`g>0`): `μ→gμ`, `σ²→g²σ²`, so the denominator becomes `√(g²σ²+ε)`, which
equals `g·√(σ²+ε)` **only if `ε=0`**. So `LayerNorm(g·x) = LayerNorm(x)`
holds **exactly in the `ε→0` limit**, and **to within numerical
precision for realistic activation scales** — this model's LayerNorm
uses PyTorch's default `ε=1e-5`, while the observed per-token hidden-state
variances (implied by the L1–L6 magnitude norms already measured, all
≫1e-5) make `g²σ²` dominate `ε` by several orders of magnitude, so the
approximation error is negligible in practice but not literally zero.
**This does not change the conclusion** — the v1 design (multiply the
whole block-output tensor by a scalar `g`) is still a near-total no-op
by the time it reaches the next block's `norm1`, for all practical
purposes — but the claim is now stated as "holds to numerical precision
given this model's activation scales," not "exactly," matching this
project's standing rule against overstating certainty.

### 1. Hypothesis, made directly falsifiable

Replacing the earlier loose framing ("LayerScale should help") with a
precise, falsifiable statement:

> **If the dominant attenuation observed under the current probe, at
> this checkpoint, primarily reflects magnitude loss rather than
> directional degradation, then selectively amplifying the
> conditioning delta immediately after the dominant attenuation point
> (block1→2) should recover Block 6 conditioning magnitude while
> preserving direction consistency ≥0.989.**

This is falsifiable: if amplifying the delta at the injection point
fails to move block 6's magnitude (propagation efficiency near zero,
see below), or moves it only by degrading direction consistency below
the established floor, the hypothesis is rejected, not reinterpreted.

### 2. Δ — exact mathematical definition

Using Item 1's own notation, unchanged, for continuity with what Item 1
actually measured (not switched to a CFG-style null-token delta, which
would be a different quantity than the one Item 1 characterized):

For a fixed (class-pair, timestep, draw `i`), let `H_k^A(i)` and
`H_k^B(i)` denote the **full per-token hidden-state tensor** returned by
`TransformerBlock` `k`'s `forward()` (shape `(1, 600, model_dim)` —
the same tensor Item 1's hook captures at L177 of
`step04_transformer_diffusion.py`, before any mean-pooling) for class
A=0 (NORM) and class B respectively, same noise `x_t` and timestep,
identical to Item 1's paired-forward-pass design.

```
Δ_k(i) = H_k^B(i) − H_k^A(i)                    (per-token, shape (1,600,D), NOT mean-pooled)
H_k^B'(i, g) = H_k^A(i) + g · Δ_k(i)             (the corrected substitute for H_k^B(i))
```

This is the **full per-token tensor**, not the mean-pooled vector Item
1 uses for its magnitude/consistency metrics — the substitution must
operate on the real tensor block `k+1` consumes, since block `k+1`'s
`forward()` takes the full `(1, 600, D)` sequence, not a pooled
summary. Mean-pooling is applied only afterward, when computing the
magnitude/direction-consistency metrics at block 6 (exactly as Item 1
already does), never to construct the substitute itself.

### 3. Hook location — exact module, exact tensor

- **Module hooked:** `model.blocks[k]` for `k ∈ {0}` (0-indexed; block
  1 in Item 1's 1-indexed reporting) for the localized variant, and
  `k ∈ {0,1,2,3,4}` (blocks 1–5) for the uniform variant — the same
  `TransformerBlock` instances Item 1's probe already hooks via
  `register_forward_hook`.
- **Tensor received by the hook:** the module's real return value,
  `x` at `step04_transformer_diffusion.py:177` — post both residual
  adds, the actual residual-stream tensor handed to block `k+1`.
- **Tensor the hook returns (substitutes):** during the class-B forward
  pass only (the class-A pass is never modified — it remains the
  unperturbed reference, exactly as in Item 1), once `H_k^A(i)` has
  already been captured (class-A pass always runs first, per the
  existing probe script's pass ordering), the hook computes
  `Δ_k(i) = H_k^B(i) − H_k^A(i)` from the just-computed raw output and
  **returns `H_k^A(i) + g·Δ_k(i)`** in its place. PyTorch forward hooks
  may return a replacement tensor that is used for all downstream
  computation within the same `model()` call — this is a live
  substitution into the running forward pass, not an offline
  recomputation.
- **Why this survives the following LayerNorm:** unlike a uniform
  scalar multiply of the whole vector (§ above), `H_k^A(i) + g·Δ_k(i)`
  is **not** a positive rescaling of `H_k^B(i)` — it changes the
  vector's direction/composition relative to `H_k^A(i)`, which
  LayerNorm's mean/std normalization does not cancel out (LN
  normalizes based on the full vector's own mean/std, which shifts
  non-trivially under this substitution, unlike under uniform scaling).
  This is asserted from the substitution's algebraic form, not
  re-derived numerically here — the identity-gain regression test in
  §6 is the empirical check that the hook mechanism itself behaves as
  specified before any non-trivial `g` is tested.

Computation graph for the localized variant (`k=1` only):

```
block1(x_t, y=A) → H_1^A(i)  [cached, unmodified reference]
block1(x_t, y=B) → H_1^B(i)  [raw] → hook replaces with H_1^A(i)+g·Δ_1(i) → fed into block2
block2..block6 run FROZEN, unmodified weights, on the corrected input
→ H_6^B'(i,g) captured at block 6, mean-pooled, compared to baseline H_6^B(i)
```

Uniform variant repeats this substitution at each of blocks 1–5,
cumulatively: block `k+1`'s hook computes `Δ_{k+1}` between whatever
tensor actually arrives (the already-corrected class-B trajectory from
block `k`) and the never-modified class-A reference `H_{k+1}^A(i)` at
that same layer.

### 4. Recovery measured at block 6 only (unchanged from v2, restated precisely)

`RecoveredMagnitude(g) = mag6_corrected(g) − mag6_baseline`, where both
quantities are Item 1's own magnitude metric (normalized L2 norm of the
mean-pooled class delta) computed at block 6 — `mag6_corrected(g)` from
the gain-corrected forward pass, `mag6_baseline` from the never-modified
baseline pass. Never measured at the injection point.

### 5. Propagation efficiency — distinguishes "survives" from "re-absorbed"

```
InjectedDelta(g)        = (g − 1) · ||Δ_k(i)||            (magnitude actually injected at the intervention point, mean-pooled, Item 1 units)
PropagationEfficiency(g) = RecoveredMagnitude(g) / InjectedDelta(g)
```

`PropagationEfficiency ≈ 1` means the injected correction survives to
block 6 essentially undiminished; `≈ 0` means the frozen downstream
blocks re-absorb it almost entirely (a distinct failure mode from
"the correction never had an effect at all," and the one this metric
is specifically designed to surface). Reported alongside recovery
fraction and direction consistency at every grid point, not folded
into a single pass/fail number.

### 6. Identity-gain regression test — mandatory, run before any real `g`

Before evaluating any `g ≠ 1`, run the complete hook-substitution
pipeline with `g = 1` (i.e. `H_k^B'(i,1) = H_k^A(i) + Δ_k(i) = H_k^B(i)`
exactly, algebraically) and confirm the resulting block 6 output is
numerically identical (to floating-point tolerance) to the unmodified
baseline forward pass — same magnitude, same direction-consistency,
same underlying tensor. **If this check fails, abort Item 2 entirely
and debug the hook mechanism before any further step** — a `g=1`
mismatch means the substitution pathway itself has a bug, and every
downstream number would be contaminated by it, exactly the discipline
already applied to the `--timestep-frac` regression test in Item 1.

### 7. Gain strategy — swept grid, fixed before implementation, not chosen post-hoc

**Option A (sweep), selected:** `g ∈ {1.0, 1.25, 1.5, 2.0, 3.0, 5.0}`,
run at every (pair, timestep) combination, producing a full
recovery-vs-gain and propagation-efficiency-vs-gain curve rather than a
single solved value. Chosen over directly solving for the threshold-
hitting `g` (Option B) because the curve itself is diagnostic — it
distinguishes "recovery saturates below 70% no matter how large `g`
gets" (architectural ceiling) from "recovery crosses 70% at some `g`
but direction consistency fails first" (a real tradeoff) from "recovery
crosses 70% cleanly." **This grid is locked now and will not be revised
after seeing results** — if `g=5.0` turns out insufficient, that is
itself the finding (localized/uniform variant rejected at this budget),
not a cue to extend the grid.

### 8. Uniform-baseline purpose, stated explicitly (not left implicit)

The equal-per-block gain split (§ budget-matching, unchanged from v2)
is **not** intended to find the best-performing intervention — it is
already acknowledged as structurally disadvantaged, since it spends
budget on transitions (block3→4, block4→5) Item 1 found no significant
effect at. Its purpose is narrower and explicit: **testing whether
localization itself matters** — i.e., whether concentrating a fixed
correction budget at the one transition Item 1 identified as dominant
outperforms spreading the same budget uniformly, not finding the
globally optimal gain allocation (which would require a separate,
unconstrained search, not in scope for Item 2).

### 9. Falsification criteria — four-way decision table, pre-registered, no post-hoc reinterpretation

Pooled (n=15 = 5 pairs × 3 timesteps) baseline statistics, computed
directly from Item 1's own raw data: L1 mean = 0.1346 ± 0.0342, L2 mean
= 0.0712 ± 0.0188, block1→2 drop = 0.0635 ± 0.0204. Recovery fraction =
`RecoveredMagnitude(g) / 0.0635` at whichever `g` in the locked grid
(§7) produces the largest recovery without violating the direction
floor; direction consistency = minimum value observed across layers
2–6 in the corrected pass, at that same `g`.

| Recovery fraction | Direction consistency | Interpretation |
|---|---|---|
| ≥70% | ≥0.989 | Hypothesis **SUPPORTED** |
| 30–70% | ≥0.989 | **Partial support** — real but incomplete recovery |
| <30% | ≥0.989 | Hypothesis **REJECTED** |
| any recovery | <0.989 | Magnitude recovered **at the expense of conditioning integrity** — not a pass, regardless of recovery fraction |

No additional interpretations are introduced after results are
observed — this table is the complete decision procedure for both
variants. **70% threshold provenance, stated plainly:** this is an
engineering judgment call, not derived from a power analysis or prior
LayerScale/DiT literature specific to this architecture — flagged
honestly rather than presented as derived.

**Uniform variant** is evaluated against the same table, using the
budget-matched gain from §8 (not an independently-tuned gain).

**Both variants land in "REJECTED" or "magnitude-at-expense-of-integrity"**
→ an informative negative result for the LayerScale family at the
analytical (pre-training) stage, written up with the same standard as
the original CFG rejection in Stage 1 — a real finding, not reframed as
partial support.

### 10. Analytical-vs-retrain: analytical first, decided and stated up front (unchanged)

**Decision: run the analytical (hook-substitution) version first**, per
the precise design above. No training, no gradient updates — a few
extra forward passes per (pair, timestep, gain) grid point, same order
of cost as Item 1 (~seconds). If the analytical check lands every
variant in REJECTED or magnitude-at-expense-of-integrity, that rules out
the entire LayerScale family before any training budget is spent.
**Only if at least one variant reaches SUPPORTED or partial support does
Item 2 escalate to an actual gradient-descent retrain** (a real
learnable gain integrated into the loss, testing whether the network
can learn to exploit the corrected magnitude in ways a static
substitution can't capture) — that retrain is a separate, later decision
point, not committed to now.

### Lock

**This pre-registration is frozen as of this revision.** No further
methodological changes are permitted once implementation begins — a
change discovered necessary mid-implementation becomes Item 2b or an
explicit "Item 2 Revision" section, not a silent edit to the criteria
above.

### Scope

All of the above is scoped to **this checkpoint**, same as the rest of
Item 1's findings — Item 9's checkpoint-scale sweep is what will show
whether any Item 2 result generalizes across training scale or is
specific to this one trained model.

**Code provenance, closed per reviewer's option (a):** the probe script
now exists in two places with distinct roles. The Stage 1 original
(`Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/
layerwise_direction_probe.py`) has been restored to its exact pre-Stage-2
state (verified byte-identical via diff against the pristine commit
`f1179f4` version) and will not be modified further — it's the immutable
historical Stage 1 artifact. A derived copy now lives at
`Roadmap/Stage_2_Architecture_Investigation/Code/
stage2_tier0_item1_layerwise_magnitude_direction/
layerwise_direction_probe.py`, carrying the `--timestep-frac` flag and
the per-draw persistence added for this review, with a header comment
explaining the derivation and why its OUT_DIR/FIG_DIR still intentionally
point at Stage 1's Outputs/Figures (all of this experiment's artifacts
were already produced there; splitting one experiment across two stage
directories would be worse than the ownership ambiguity this copy
resolves). Items 2-9, which have no Stage 1 lineage, will write natively
into Stage 2's own Code/Outputs/Figures from the start.

**No ADR written, per reviewer's explicit note.** This document remains
the correct home for a measurement report; an ADR is for an
architectural decision actually being committed to, which doesn't exist
yet after one Tier 0 item. `Stage2_Final_Report.md` and the first ADR
get written once enough of the Tier 0 battery (at minimum Item 2)
produces something concrete enough to commit to in Stage 3.

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

**Not yet run:** Items 2-9. Item 2 will pre-register its falsification
criteria in writing in this document before executing either variant.
Stopping here for final sign-off on this corrected, statistically-backed
Item 1 writeup before starting Item 2.
