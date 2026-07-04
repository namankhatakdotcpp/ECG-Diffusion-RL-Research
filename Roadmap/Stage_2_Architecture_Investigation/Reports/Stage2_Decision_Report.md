# Stage 2 Decision Report

Synthesizes Stage 2's Tier 0 investigation (8 items, all closed) into
architectural conclusions and Stage 3 recommendations. **This document
does not introduce new evidence** -- every claim below traces to a
specific item report and the confidence calibration already established
in `Stage2_Evidence_Matrix.md`. Where evidence is thin, contradictory,
or absent, that is stated plainly rather than smoothed over.

## 1. Executive summary

Stage 2's Tier 0 investigation asked why this diffusion model's class
conditioning weakens across its 6 Transformer blocks, using 8
independent, complementary measurement approaches: forward-pass
activation magnitude (Item 1), causal gain-correction interventions
(Items 2A/2B), within-pass residual-update dynamics (Item 3), training-
time gradient magnitude (Item 4), static weight-capacity allocation
(Item 5), attention-entropy class-dependence (Item 6), embedding
evolution across training (Item 7, blocked), and representation
decodability (Item 8). Seven items closed with verified, statistically-
grounded conclusions; one (Item 7) closed as permanently blocked by a
real, documented data-availability gap, not a soft "inconclusive."

The clearest, best-corroborated finding is that **conditioning
magnitude/influence genuinely attenuates across the network, and this
attenuation is recoverable** -- gain-correction interventions (Item 2A,
2B) restore block-6 magnitude on their own pre-registered criteria, and
three independent measurement types (Items 1, 3, 5) converge, with a
fourth (Item 6) showing a directionally consistent but statistically
softer signal, on blocks 5-6 as the specific locus where something
architecturally distinctive happens. Critically, this magnitude
attenuation is **not** an information loss: Item 8 found class
information remains perfectly linearly decodable at every block, even
where Fisher ratio (and, independently, Item 1's magnitude) has
declined by more than 10x -- a permutation-verified result, not a
dimensionality artifact. This reframes the architectural question from
"is class information being lost" to "why does a fully-preserved signal
exert proportionally shrinking influence on the residual stream, and
does that reduced influence matter for generation quality."

One comparison is explicitly flagged as **not usable evidence**: Item
2B's claim that uniform-gain correction outperforms localized-gain
correction is confounded by a budget-matching formula that breaks down
under nonlinear compounding -- this should not be cited as a basis for
preferring a distributed correction scheme in Stage 3 without first
fixing that methodology.

## 2. Evidence matrix

Imported in full from `Stage2_Evidence_Matrix.md` -- not re-derived
here. See that document for the complete claim-by-item grid, per-cell
justifications, and the confidence calibration (High/Moderate/
Pending/Low/N/A). Item 4 is now closed (VERIFIED); the matrix has been
updated accordingly and no cells remain Pending.

## 3. Cross-item synthesis

### 3.1 Claims corroborated by 2+ independent measurement types

- **Conditioning magnitude/influence attenuates across the network, and
  is recoverable.** Item 1 (forward-pass activation delta, Wilcoxon-
  verified) establishes the pattern; Items 2A and 2B (causal gain-
  correction interventions, each independently SUPPORTED on its own
  pre-registered criteria) establish that the attenuation is not a
  structural dead end -- restoring magnitude via a substitution hook
  recovers block-6 conditioning signal. **High confidence.**
- **Class-conditioning influence weakens as noise (timestep) increases.**
  Item 1's forward-pass magnitude (0.173 -> 0.124 -> 0.107 at
  t=100/500/900) and Item 4's training-time gradient percentile rank
  (56.8% -> 38.9% -> 33.7% at the same timesteps) independently decline
  in the same direction -- two entirely different measurement modes
  (inference-mode activation sensitivity vs. training-mode gradient
  magnitude) agreeing. Stated as convergent correlation, not proof one
  causes the other (per this project's standing discipline on cross-
  item claims). **High confidence** on the correlation; no causal claim
  made or supported.
- **Blocks 5-6 are architecturally distinctive vs. blocks 1-4** -- the
  most-corroborated but also most-nuanced claim in this investigation.
  Three items show strong, independently-derived signal at exactly
  this boundary: Item 1 (statistically confirmed secondary magnitude
  drop at block5->6), Item 3 (residual-update-ratio spike at block 6,
  ~8x the block-3 valley, confirmed by causal ablation to have
  outsized post-normalization influence), and Item 5 (the only
  scale/shift-allocation decrease across all 6 blocks occurs at
  block5->6, and the largest single jump occurs at block1->2). Item 6
  adds a fourth, directionally-consistent but statistically softer
  signal (point estimates for class-dependent attention entropy are
  highest at blocks 5-6, but the 95% CIs cross the locked 0.05
  threshold -- suggestive, not independently significant). **Item 8
  explicitly does not corroborate** (linear-probe accuracy is flat
  100% at every block, permutation-verified, no block-5/6 signal at
  all) -- reported here as a real non-convergence, not omitted or
  smoothed into the majority view. Item 4 is **not designed to test
  this claim at all** (its gradient analysis buckets by parameter type,
  never by block index) -- correctly marked N/A in the matrix, not
  folded into either side. **Moderate confidence overall**: strong on
  3 of 5 block-testable items, softer on a 4th, absent on a 5th.

### 3.2 Single-source claims (flagged as such, not treated as weaker just for being single-source)

- **AdaLN allocates scale/shift capacity non-uniformly across blocks**
  (Item 5 alone). High confidence despite single-source status: this is
  an exact, deterministic property of the frozen checkpoint's weights
  (n=1, no sampling variance to caveat), not a sampled estimate.
- **Attention is substantially class-blind network-wide** (Item 6
  alone). High confidence for the network-wide pooled verdict
  specifically (it clears its own locked threshold with room to
  spare); the blocks-5/6 refinement within the same item carries the
  CI caveat noted above.
- **Class information never collapses in the decodability sense, even
  as magnitude/Fisher-ratio measures decline** (Item 8 alone). High
  confidence -- internally validated via a label-permutation control
  that directly rules out the memorization/dimensionality-artifact
  explanation for the 100% probe accuracy, not merely asserted.
- **Class embedding received a non-negligible, above-median (but not
  dominant) gradient during training** (Item 4 alone, on this specific
  claim). High confidence -- bootstrap-verified tight CI, real
  checksum/reproducibility values, i.i.d. draw independence confirmed.

### 3.3 Item 4's granularity mismatch (explicit limitation, not smoothed into the block-level narrative)

Item 4's gradient-competitiveness analysis buckets parameters by TYPE
(adaLN / attention / ffn / norms / embeddings / projection), never by
block index. It therefore cannot support or contradict any block-level
claim (including the blocks-5/6 convergence in Sec. 3.1) and is marked
**N/A**, not "Pending" or "Neutral," in the evidence matrix. This was
caught and corrected during Item 4's own review cycle -- an earlier
draft risked implicitly counting Item 4 among the blocks-5/6
convergence set, which would have manufactured false agreement. Any
future work wanting a block-level gradient breakdown would need a new,
differently-designed probe -- this is a scope gap in Item 4's design,
not a result that could be reinterpreted from existing data.

### 3.4 Item 7's BLOCKED status (a real gap in the evidence base, not omitted)

Item 7 (class-embedding evolution across training) produced **no
evidence at all** -- not weak evidence, not inconclusive evidence, none.
No per-epoch checkpoint from the training run that produced the current
`diffusion_best.pt` exists anywhere (confirmed by filesystem and
archive search); the retention policy that pruned them was confirmed
active in this run's own commit lineage. This is a genuine, permanent
gap in what Stage 2 can say about how the class embedding's
differentiation evolved during training -- it is not answered by any
other item, and should not be treated as implicitly covered by, e.g.,
Item 4's training-time gradient analysis (which measures gradient
magnitude at the current, final checkpoint only, not the trajectory of
the embedding itself). **If Stage 3 undertakes any retraining, preserving
periodic checkpoints costs nothing and would close this gap
retroactively for future analysis.**

## 4. Architectural conclusions

Each conclusion cites its supporting item(s) and confidence level,
matching `Stage2_Evidence_Matrix.md`'s calibration.

1. **Conditioning magnitude attenuation is real and correctable via
   gain-style correction.** (Items 1, 2A, 2B -- High.) A LayerScale-
   style learnable gain mechanism is a legitimate, evidence-backed
   candidate for Stage 3, per Item 2's own pre-registration Sec. 10,
   which explicitly frames a SUPPORTED analytical result as the trigger
   point for *considering* (not committing to) a gradient-descent
   retrain escalation.
2. **The block1->2 transition and the block5-6 region are the two
   most-implicated loci for an architectural intervention**, though
   with real, stated uncertainty. (Items 1, 3, 5 -- High/Moderate per-
   item; Item 6 -- Moderate, directionally consistent; Item 8 --
   explicit non-convergence, High confidence in its own right.) Any
   Stage 3 intervention targeting these transitions should be validated
   empirically against Item 8's finding, not assumed to also improve
   decodability (which Item 8 shows is already at ceiling).
3. **Cross-attention is not strongly indicated as a necessary fix.**
   (Item 6 -- High, network-wide.) The existing self-attention
   mechanism is already substantially class-blind, meaning it isn't
   discriminating by class in a way that adding a new cross-attention
   pathway would obviously improve upon, per the master prompt's own
   stated interpretation of this test.
4. **AdaLN is the dominant conditioning-injection and gradient-receiving
   mechanism, and any architecture change touching the conditioning
   pathway should account for its existing capacity/gradient
   dynamics.** (Item 5 -- High, static weight allocation; Item 4 --
   High, adaLN receives ~6.7x class_emb's mean gradient.) This is not a
   recommendation to change adaLN itself -- no item tested an
   intervention on adaLN's own structure -- but any new conditioning
   pathway (e.g. cross-attention, if pursued despite conclusion 3)
   should be designed with awareness that adaLN already absorbs most of
   the gradient signal and weight capacity in this region.
5a. **Class information remains fully decodable at every block,
    despite magnitude/Fisher-ratio decline.** (Item 8 -- **High
    confidence**, permutation-verified: a label-shuffle control
    directly ruled out the dimensionality-artifact explanation for the
    100% linear-probe accuracy.) This alone rules out simple
    information loss as the explanation for any downstream generation
    issue -- whatever explains poor conditioning, it is not that the
    class label becomes undecodable.

5b. **HYPOTHESIS, not yet directly tested:** the mechanism connecting
    5a to any generation-quality issue may be that conditioning's
    *proportional* influence shrinks as overall residual-stream
    magnitude grows faster than conditioning-specific magnitude does.
    **This is a synthesis of two separately-measured quantities from
    two different items, never computed as a single ratio by any
    item:** Item 1's cross-class delta magnitude declines across
    blocks, and Item 3's causal-ablation data shows block-6's output
    norm (73.89) is substantially larger than block-3's (48.50) --
    i.e., overall representation magnitude grows over the same span.
    No item measured `conditioning-delta / total-output-norm` as one
    tracked quantity across blocks. **Confidence: Low-to-Moderate**
    (plausible synthesis, untested as a unified metric) -- this is a
    candidate explanation for Stage 3 to test directly, not a verified
    finding, and should not be cited as established when scoping any
    specific architectural direction (e.g. investigating `final_norm`/
    `unproj`).

## 5. What NOT to pursue

Patterns that were tested and did not hold up, or claims that should
not be extrapolated beyond what was actually verified:

- **Do not cite "uniform gain outperforms localized gain" as an
  architectural preference.** (Item 2B.) The comparison is confounded
  by a budget-matching formula that breaks down under nonlinear
  compounding across the 5 cumulative substitution points -- the raw
  recovery-percentage gap between variants reflects, at least in part,
  the uniform variant receiving more effective total correction than
  the formula intended to match, not distribution-vs-concentration on
  its own merits. If Stage 3 wants to test this question properly, it
  needs a corrected budget-matching methodology, not a reuse of Item
  2B's.
- **Do not treat Item 6's "blocks 5-6 individually exceed the 0.05
  class-dependence threshold" as an independently statistically
  confirmed finding.** After computing 95% CIs across the 15 pooled
  cells (a check the original draft of Item 6's report lacked), both
  blocks' intervals cross the threshold. The direction is real and
  consistent with Items 1/3/5, but the specific threshold-crossing
  claim should be cited as suggestive, not confirmed.
- **Do not fold Item 8's representation-collapse finding into the
  blocks-5/6 narrative in either direction.** Linear-probe accuracy is
  flat 100% at every block -- it neither confirms nor is explained by
  the blocks-5/6 convergence from other items. Forcing it into that
  narrative (positively or negatively) would misrepresent a genuinely
  independent, non-corroborating result.
- **Do not extrapolate Item 4's findings to any per-block claim.** Its
  design cannot support block-level statements in either direction, as
  established in Sec. 3.3.
- **Do not treat Item 7's gap as resolved or as safe to ignore.** It
  remains a real unknown about training-time embedding evolution for
  this specific trained model. Do not assume Item 4's gradient findings
  (measured only at the final checkpoint) substitute for it.
- **Do not revisit Items 1-8 individually without new data or a new
  architectural change to test** -- per this project's standing "stage
  discipline," Stage 2 is complete and serves only as evidence for
  Stage 3 from this point forward.

## 6. Stage 3 recommendations

Each recommendation is traceable to a specific conclusion in Sec. 4.

1. **Implement a LayerScale-style learnable per-block (or per-
   transition) gain mechanism as the primary Stage 3 architecture
   change**, per Item 2 v3's own pre-registered escalation path (Sec.
   10). Traces to conclusion 1 (High confidence, Items 1/2A/2B).
2. **If the gain mechanism is localized rather than uniform, prioritize
   the block1->2 transition; if distributed, ensure the budget-matching
   methodology is corrected from Item 2B's confounded version before
   drawing any distribution-vs-concentration conclusion.** Traces to
   conclusions 1 and 2, and to the "what not to pursue" note on Item
   2B's confound (Sec. 5).
3. **Do not prioritize adding cross-attention as a first Stage 3 change**
   unless a new, specific hypothesis for why it would help is
   developed -- the existing evidence (Item 6) suggests the current
   self-attention mechanism is not the class-blindness bottleneck.
   Traces to conclusion 3 (High confidence, Item 6, network-wide).
4. **Any new conditioning-injection design should be evaluated against
   adaLN's existing gradient/capacity dominance**, to avoid
   introducing a redundant pathway that competes poorly with an
   already-dominant one. Traces to conclusion 4 (High confidence,
   Items 4/5).
5. **Do not scope any Stage 3 change around "preserve more class
   information"** -- ruled out by 5a (High confidence, Item 8): class
   information is never information-theoretically lost. **Test the
   dilution hypothesis (5b) directly before treating it as
   established**: compute `conditioning-delta / total-output-norm` per
   block on the existing checkpoint (cheap, no retrain, reuses Item
   1/3's already-captured quantities in a new ratio) to see whether it
   actually explains the attenuation pattern. **Independent of 5b's
   outcome**, continue investigating downstream-of-block-6 mechanisms
   (`final_norm`, `unproj`) -- 5a alone already justifies looking past
   "is information present" and toward "how is it used," regardless of
   whether the specific dilution mechanism in 5b turns out to be
   confirmed. This is the most novel, least obvious recommendation this
   investigation produced -- redirecting the entire "where does
   conditioning fail" question away from information loss and toward
   signal utilization -- and it should not be overstated beyond what
   5a/5b actually support.
6. **If retraining occurs for any reason in Stage 3, preserve periodic
   checkpoints** (a configuration change, not a new experiment) so
   Item 7's question becomes answerable retroactively without
   requiring a dedicated future retrain solely for that purpose.

## Provenance

Synthesizes: `Reports/Tier0_Findings.md` (Item 1), `Reports/Item2_Report.md`
(2A), `Reports/Item2B_Report.md` (2B), `Reports/Item3_Report.md` (3),
`Reports/Item4_Report.md` (4), `Reports/Item5_Report.md` (5),
`Reports/Item6_Report.md` (6), `Reports/Item7_Report.md` (7,
BLOCKED), `Reports/Item8_Report.md` (8), and `Reports/Stage2_Evidence_Matrix.md`.
No new experiments, forward passes, or claims were introduced in
producing this document.
