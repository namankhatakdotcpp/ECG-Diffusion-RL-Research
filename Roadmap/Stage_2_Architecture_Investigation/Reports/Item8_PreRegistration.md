# Item 8 -- Representation Collapse Analysis -- Pre-Registration

## Hypothesis, quoted verbatim (`Stage2_Master_Prompt.md:230-268`)

> "8. Representation Collapse Analysis (Fisher ratio + linear probe).
> Use representation_metrics.py. Run BOTH fisher_ratio() and
> linear_probe_accuracy() at EVERY block -- do not use a low Fisher
> ratio to decide to skip the linear probe at that block. [...] Pool
> hidden states via mean-pooling over the 600 tokens per sample to get
> one feature vector per sample at each block. Hold timestep t fixed
> (or bucket by comparable noise level) across the batch used for a
> single Fisher-ratio/probe computation -- do not mix samples from
> very different t values into one calculation, since t itself changes
> hidden-state statistics independent of class."

## Architectural question

At which blocks, if any, does the model's internal representation
stop separating by class -- using two independent, complementary
tests (a cheap closed-form statistic, and a trained linear decoder) --
and does this pattern corroborate or contradict Items 1/3/5/6's
convergent finding that blocks 5-6 are architecturally distinctive?

## Deliberate override, stated explicitly (not silently followed or ignored)

`representation_metrics.py`'s own docstring recommends running
`linear_probe_accuracy()` "ONLY at the 1-2 blocks that `fisher_ratio()`
flags as interesting... don't run it 6x per experiment by default"
(cost-saving guidance). **The master prompt explicitly overrides this**:
"Run BOTH ... at EVERY block -- do not use a low Fisher ratio to decide
to skip the linear probe at that block." Followed here per the master
prompt's explicit instruction, not the module's own more conservative
default -- stated as a deliberate choice, not a silent deviation from
either source.

## Design

- No forward-pass cost concern here (unlike a naive worry about full
  reverse-diffusion sampling): "hidden states" means the same per-block
  mean-pooled activations Item 1/3/6 already capture from a single
  noisy-input forward pass at a fixed timestep -- NOT full T-step
  generation. Confirmed by the docstring's own "hold timestep t fixed"
  language, which only makes sense for single-timestep forward passes,
  not multi-step sampling.
- Per timestep (`t in {100, 500, 900}`, Item 1/3/6's own convention,
  reused here since the docstring requires holding t fixed per
  computation, and testing all 3 lets Item 8 report per-timestep
  results the same way Item 1 did): draw `n_gen=20` independent noise
  samples PER CLASS (the master prompt's own example number), for ALL
  6 classes simultaneously in one batch (unlike Item 1/3/6's pairwise
  NORM-vs-other design -- Fisher ratio/linear probe need multiple
  classes present at once, not pairs) -- `n=120` samples per timestep.
- At each of the 6 blocks: mean-pool over the 600 tokens (same
  `register_layer_hooks` mechanism already used by Items 1/3), run
  `fisher_ratio()` and `linear_probe_accuracy()` on the 120-sample,
  6-class feature set.
- `min_class_count` defaults (5 for Fisher, 10 for probe) are well
  clear of `n_gen=20`/class -- no exclusions expected, but checked and
  reported explicitly per the module's own contract, not assumed.
- `linear_probe_accuracy`'s PCA-reduction guard is expected to trigger
  (`n_train ~= 84` after the 0.7 train split, `D=256`,
  `84 < 5*256=1280`) -- this will be reported explicitly (the module's
  own `pca_components` field), per the module's own stated requirement
  that any citation of the accuracy state this fact.

## Statistical treatment

Descriptive, per the module's own design (closed-form Fisher ratio;
single train/test split for the probe, not cross-validated -- matching
the module's own signature, not adding cross-validation the module
doesn't already do). `n=120`/timestep is the sample size; stated
explicitly per block/timestep, not hidden.

## Decision criteria

Per-block, per-timestep: **VERIFIED (representation separates)** if
`linear_probe_accuracy` (the confirmatory test, per the module's own
stated hierarchy) is meaningfully above chance (`> chance + 0.15`,
matching a similar margin-above-chance convention informally implied
by the module's own "0.333 chance, 0.214 test accuracy" collapse
example); **FALSIFIED (collapsed)** if at or below chance + 0.15.
Fisher ratio is reported alongside at every block for the qualitative
pattern (per the module's own docstring), never used alone to declare
collapse, per the master prompt's explicit instruction.

## `common/` reuse

- `common/io.py` -- REUSABLE AS-IS.
- `common/hooks.py`'s `register_layer_hooks` -- REUSABLE AS-IS (same
  mean-pooled per-block capture Item 1/3 already use).
- `common/metrics.py`, `common/statistics.py` -- NOT APPLICABLE (Item
  8 uses `Roadmap/_infra/representation_metrics.py` directly, per the
  master prompt's own explicit instruction to use that module, not
  reimplement Fisher ratio/linear probe in `common/`).
- `common/plotting.py` -- REUSABLE WITH EXTENSION: one new plot
  (Fisher ratio AND probe accuracy vs. block, per timestep).

## Required outputs

- `representation_collapse_raw.json`, `representation_collapse.csv`.
- One plot (or small multiple across the 3 timesteps).
- `Item8_Report.md`.

## Runtime / compute

CPU, same cost order as Item 1/3/6 (3 timesteps x 120 samples = 360
single forward passes, no training data, no backward pass) --
seconds to ~1-2 minutes, plus near-instant Fisher ratio / linear-probe
fitting (n=120, D=256, trivial for `sklearn`).

## Lock

Frozen once implementation begins.
