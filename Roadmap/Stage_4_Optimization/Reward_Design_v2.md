# Reward Design v2 — DiagnosticUtilityReward Reliability Scaling for HYP

**Status:** Draft, uncommitted. **Tradeoff-direction decided by Dr. Balaji
(2026-07-22): run the full 5-point sweep (Section 4a), not a single
point.** The success/marginal/failure criteria (Section 9) remain open —
still require Dr. Balaji's and/or the user's input, not guessed at here.
No code change required for the proposed experiment.

## 1. Confirmed live config baseline

`config.yaml:168-179` (verbatim, as of commit `9311d67`):

```yaml
  weights:
    diag:  0.40                # DiagnosticUtilityReward — classifier confidence
    a3:    0.15                # A3Reward                — A3-subband slow-wave match
    morph: 0.16                # MorphologyReward        — PQRST interval matching
    real:  0.16                # RealismReward           — PCA manifold proximity
    hrv:   0.13                # HRVReward               — HRV plausibility
  pca_components: 50          # PCA components for RealismReward
  pca_n_train_samples: 2000   # training records used to fit PCA
  reliability_alpha: 1.0      # DiagnosticUtilityReward reliability interpolation:
                               # reliability = alpha * per_class_f1 + (1 - alpha) * 1.0
                               # 1.0 = full F1 scaling (prior default), 0.0 = no scaling.
                               # See Roadmap/Stage_4_Optimization/Reward_Design_v2.md Section 4a.
```

Five terms, not six. An earlier, unvalidated 6-term split (diag .35 / a3
.20 / morph .15 / real .15 / hrv .10 / regularization .05) is recorded in
`Decisions.md`, but the regularization term was dropped before
implementation and the remaining 0.95 rescaled to 1.0. No regularization
term exists in the live reward function. `use_reliability_scaling` (the
prior boolean on/off switch) was replaced in commit `9311d67` by
`reliability_alpha: float` — a continuous interpolation, not a toggle —
specifically to enable this document's 5-point sweep (Section 4a).
Confirmed from `step06_reward_function.py`: `self.reliability` is one
`n_classes`-length array computed as
`alpha * per_class_f1 + (1 - alpha) * ones`; still a single global scalar,
not per-class — no existing code path lets one class opt out while others
keep scaling.

## 2. Scope: HYP only

`trtr_classifier_eval.json` per-class F1 (order NORM/MI/STTC/CD/HYP/OTHER):

```
0.792 / 0.645 / 0.612 / 0.585 / 0.376 / 0.571
```

HYP (0.376) is a clear outlier. OTHER (0.571) is mid-pack, not weak, and
is **explicitly excluded** from this document: it belongs to
`Investigation_02_OTHER_Collapse_Mechanism.md`, deprioritized specifically
because its root cause (the iteration-383 transition) was never
determined — folding OTHER into a HYP-specific reliability-scaling fix
would conflate two different, already-distinguished failure profiles.

## 3. DiagnosticUtilityReward mechanism

Confirmed from `step06_reward_function.py:339-452`. Scores against
`outputs/models/trtr_classifier.pt`, the native 6-class TRTR classifier
(NORM/MI/STTC/CD/HYP/OTHER) — not the 4-class mentor proxy. HYP and CD
already have well-defined reward targets; the 4-class mentor-proxy
taxonomy gap (Investigation_02/03 territory) affects only post-hoc
mentor-facing reporting, not the reward the policy actually trains
against. No new reward classifier is needed.

```
reward = confidence(target_class) * reliability[target_class]
```

For HYP, `reliability = 0.376` caps the diagnostic reward low regardless
of how confidently-correct the classifier's softmax is — the plausible
driver of the Gate 3 finding (iterations 166-215: HYP `r_diag` near zero
for ~50 iterations, no correlated KL/grad_norm signal — already
attributed in `Decisions.md` to "the TRTR classifier's weak HYP
reliability... not a PPO defect").

The docstring states this scaling exists specifically as a reward-hacking
guard: *"a policy shouldn't be able to farm free reward from a class the
classifier itself is weak at distinguishing"* (`step06_reward_function.py
:346-350`), and cites `step09_ablation_study.py` as having demonstrated
this hacking mode for the Diagnostic term generally.

**Correction: "demonstrated" is unverified in this repo.**
`step09_ablation_study.py` exists as a script but has never been run here
— no `ablation_table.tex`, `ablation_results.json`, or `ablation_{name}.pt`
checkpoints exist anywhere, and `Decisions.md` never mentions step09. The
hacking concern is a plausible, documented design rationale — it matches
the general RL reward-hacking pattern and should not be dismissed — but it
is not backed by experimental evidence currently on record in this repo.
Treat as "asserted but unverified here," not "already demonstrated."

## 4. The tradeoff — a continuum, not a binary choice

Three points on a reliability-weighting spectrum, all currently
unvalidated:

|  | Full scaling (current) | Partial scaling (untested hypothesis) | No scaling |
|---|---|---|---|
| HYP gradient starvation | Present (Gate 3 evidence) | Unknown — not run | Resolved |
| Reward hacking on HYP | Guarded (docstring intent, evidence unverified) | Unknown — not run | Reopened, severity unknown |

Partial scaling (e.g. `reliability_adj = 0.5 + 0.5*f1`) is an engineering
hypothesis, not a preferred solution — it has no more experimental backing
than the "no scaling" pole, only a plausible design rationale for why it
might land better. It should not be presented to Dr. Balaji as the
recommended fix; all three points are equally unvalidated, and the choice
among them is a reward-philosophy question, not an implementation detail.

Morphology/HRV terms exist partly as physiological anchors against exactly
this hacking mode per the same docstring, so reduced/removed scaling's
reopened risk may be partially, not fully, offset by the other terms
already in the composite — an untested hypothesis, not a guarantee. No
decision is made here on which point on this spectrum to take.

**Framing for discussion**: not "keep scaling or remove it," but "which
family of reliability-weighting strategies matches the intended reward
philosophy — full, partial, or none — before any one point on that
spectrum gets validated experimentally." This surfaces whether
reliability-scaling-as-hacking-guard is even the right mental model in
the first place, before the conversation gets stuck relitigating that
mid-discussion on a specific number.

## 4a. Dr. Balaji's decision (2026-07-22): sweep, don't pick

Dr. Balaji's response: *"I also agree with Option 2. Instead of fixing
one reliability weight, we should make it part of the validation study
and compare multiple settings (100%, 75%, 50%, 25%, 0%). If the results
show that removing reliability scaling consistently performs better
without introducing reward hacking, then we can justify moving to
Option 3."*

This closes the Section 9 tradeoff-direction item — but as a decision to
**run all five points**, not to select one in advance. "Option 3" (full
removal as the adopted design) is a conclusion the sweep may justify,
not a starting assumption. `s` below is the scaling fraction — `s=1.00`
is the current live config, `s=0.00` is no scaling — using the linear
form `reliability_adj(s) = (1-s)*1.0 + s*0.376`:

| `s` (scaling %) | `reliability_adj` for HYP | Relation to existing sections |
|---|---|---|
| 1.00 (100%) | 0.376 | current live config (Section 1) — baseline, not re-run |
| 0.75 | 0.532 | new point |
| 0.50 | 0.688 | matches Section 5's illustrative `0.5 + 0.5*f1` example |
| 0.25 | 0.844 | new point |
| 0.00 (0%) | 1.000 | "no scaling" pole (Section 4) |

Four new eval JSONs needed (the `s=1.00` point is the existing
`trtr_classifier_eval.json`, already the live baseline — no new run
needed for it). Section 6/7's `r_diag`+`r_morph` hacking-flag buckets
apply per-point, compared across all five, not just against the two
extremes.

## 5. Proposed experiment — no code change needed

`DiagnosticUtilityReward.__init__` accepts an `eval_path` override
(`step06_reward_function.py:368,409`) pointing reliability-loading at any
JSON file, not just the default `trtr_classifier_eval.json`. A HYP-only
ablation is achievable by pointing at an alternate JSON identical to
`trtr_classifier_eval.json` except with HYP's entry set to `1.0`
(uniform/no scaling) while the other five classes keep their real F1
values — no change to `step06_reward_function.py` itself, just a
config-pointed alternate file.

## 6. Corrected scaffolding — two independent signals for HYP, not three

Checked variance, independence-from-`r_diag`, and sampling adequacy for
all three candidate signals over HYP's 177 logged iterations (of 1000
total in `stage4_finetune_v1`'s run — HYP sampled at ~17.7% vs. 16.7%
round-robin expectation: proportional, not starved).

| Signal | Status | Why |
|---|---|---|
| `r_diag` | Primary — what's being fixed | n/a |
| `r_morph` | Usable, independent | std=0.115, range 0.27-0.95, 0% near-zero; r=-0.11 vs. `r_diag` (near-independent) |
| `r_a3` | Not usable as a hacking-flag axis | Not dead (std=0.286, range 0.002-0.821) but r=-0.86 vs. `r_diag` across the *entire* 1000-iteration run — this anti-correlation predates and is independent of the proposed experiment, so seeing it recur would not distinguish new hacking-induced decoupling from the pair's existing baseline relationship |
| `r_hrv` | Dropped — dead signal | mean 3.5e-05 over all 177 iterations, essentially always 0.00000; structurally uninformative for HYP (plausibly because HYP is a morphological rather than rhythm-based diagnosis) |

**Both exclusions are real findings, not the same finding twice.** HRV is
uninformative (no signal at all). A3 is informative but confounded (real
signal, wrong kind of correlation for this purpose — pre-existing across
the whole run, not something the experiment would newly produce). This
distinction should be preserved when this goes to Dr. Balaji rather than
flattened into "both dropped for the same reason."

**Revised buckets, `r_diag` + `r_morph` only:**
- **Clean success**: `r_diag` rises out of near-zero, `r_morph` holds
  steady or improves over the same window.
- **Structural hacking flag**: `r_diag` rises while `r_morph` degrades.
- **Ambiguous**: any other pattern — routes to Dr. Balaji's direct visual
  read of the zoomed-region figures, not a threshold decision.

## 7. `r_diag`/`r_a3` anti-correlation — investigated, substantially explained

**Structural check first**: `DiagnosticUtilityReward` and `A3Reward`
share no computation (`step06_reward_function.py:339-452` vs. `:457-609`).
The former runs the raw ECG through the TRTR CNN classifier's softmax;
the latter runs it through wavelet subband-energy extraction →
Mahalanobis distance to a frozen real-data reference computed once at
init and never updated during training. Independent feature pipelines on
the same input — the anti-correlation is not "the same number wearing two
names."

**Trend decomposition** (HYP's 177 logged iterations, ordered by `iter`):
- `r_diag` vs. iteration index: **r = -0.61** (declines over training —
  consistent with the Gate 3 finding of an extended near-zero stretch).
- `r_a3` vs. iteration index: **r = +0.73** (improves over training).
- Raw `r_diag` vs. `r_a3`: **r = -0.86**.
- First-differences (Δr_diag vs. Δr_a3, i.e. iteration-to-iteration
  *changes* with the trend removed): **r = -0.28**.
- Stable across training (first-half r=-0.84, second-half r=-0.73) —
  not concentrated in one era.

**Conclusion**: most of the raw -0.86 is explained by the two series
having opposite whole-run trends, not a tight moment-to-moment mechanical
link — the residual first-difference correlation (-0.28) is real but far
weaker. The trend itself is informative in its own right: generated HYP
samples appear to be getting **more realistic on the A3 axis** as training
progresses, while the diagnostic reward simultaneously **declines** — the
opposite of what a hacking signature would look like (hacking would show
diagnostic confidence rising while realism measures degrade or stay flat,
per the Section 6 buckets). This is consistent with, and adds independent
support to, the weak-classifier-bottleneck framing in Sections 2-3: the
model may be producing increasingly realistic HYP samples that a
*reliable* classifier would confidently reward, while the actual weak
classifier (F1=0.376) fails to recognize the improvement. Worth
mentioning to Dr. Balaji as supporting evidence for the bottleneck
framing, not just a curiosity — though the -0.28 residual coupling isn't
fully explained and shouldn't be overclaimed as resolved either.

## 8. Next steps, in order

1. ~~Investigate the `r_diag`/`r_a3` anti-correlation~~ — **done** (Section
   7). No shared computation between the two reward terms; the raw -0.86
   correlation is substantially a trend artifact (opposite whole-run
   trends), with a weaker (-0.28) residual moment-to-moment coupling not
   fully explained. Reads as supporting evidence for the weak-classifier-
   bottleneck framing, not a hacking signature.
2. ~~Take this document to Dr. Balaji framed as a design-space
   question~~ — **done** (2026-07-22). Decision: run the full 5-point
   sweep (Section 4a), not select one point in advance.
3. Produce the four new eval JSONs for HYP per Section 4a's table
   (`s = 0.75, 0.50, 0.25, 0.00`; `s = 1.00` is the existing
   `trtr_classifier_eval.json`, no new file needed) — no change to
   `step06_reward_function.py` itself, per Section 5.
4. Pre-register success/marginal/failure criteria before running
   anything — specifically, what HYP `r_diag` trajectory over a short
   (Gate-3-scale) window counts as "starvation resolved," and a concrete,
   checkable definition of "hacking observed" using the revised
   `r_diag`+`r_morph` buckets (Section 6) — decided before the run, not
   judged post-hoc. **Not written here** — this is Dr. Balaji's and the
   user's call, not a default to guess at. Still open per Section 9.
5. Run the short validation for each of the four new points, compare
   against the same-length window from the existing Stage 4 run for HYP
   (the `s=1.00` baseline) — five-way comparison, not pairwise.
6. Decision rule per Dr. Balaji: if results show removing scaling
   (lower `s`) consistently performs better **without** introducing
   reward hacking (Section 6 buckets), that justifies adopting no
   scaling as the new design (his "Option 3"). Otherwise, whichever
   point on the spectrum best balances the two axes is the outcome —
   still pending (5)'s results.

## 9. Explicitly open items

- ~~**Tradeoff direction**~~ (Section 4): **closed 2026-07-22** — Dr.
  Balaji decided to run the full 5-point sweep (Section 4a) rather than
  select a single point. "No scaling" (his "Option 3") is adopted only if
  the sweep shows it consistently wins without reward hacking; not
  assumed in advance.
- **Success/marginal/failure criteria and "hacking observed" definition**
  (Section 8 item 4): requires Dr. Balaji's and the user's judgment, not
  drafted here.
- ~~`r_diag`/`r_a3` anti-correlation~~ (Section 7): investigated and
  substantially explained (trend artifact + weaker real residual); no
  longer open, kept here only as a pointer to Section 7's detail.

## 10. Explicitly out of scope

- **OTHER** — different failure profile (F1 0.571, not weak); belongs to
  the deprioritized `Investigation_02_OTHER_Collapse_Mechanism.md` thread.
- **CD, MI, STTC** — no evidence any of these show HYP's starvation
  pattern; not touched here.
- Any change to the other four reward terms or their weights.
