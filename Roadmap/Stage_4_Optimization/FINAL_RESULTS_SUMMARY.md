# Reliability-Alpha Sweep — Final Results Summary

Working reference doc for drafting the mentor report/paper. Not a
permanent repo artifact in this exact form — assembled facts with
sources, no prose framing yet. Not committed; review before deciding
whether/how it enters version control.

## 1. Baseline

Production checkpoint: `gate3_250_fixed` — `reliability_alpha=1.00`
(`use_reliability_scaling=true`), 250 RL fine-tuning iterations, seed 42
(`cfg.seeds[0]`), same frozen base checkpoint (`diffusion_best.pt`) and
reward weights as all four sweep arms. Confirmed bit-identical equivalent
to a fresh alpha=1.00 run via the Step 2 identity-check test (not a
re-run — reused as-is).

## 2. Sweep design

4 arms, `config_reliability_sweep_alpha{0.75,0.50,0.25,0.00}.yaml`, seed
42, 250 iterations each, run sequentially via `run_reliability_sweep.sh`
on 2026-07-22 (raw log: `reliability_sweep_run.log`, still untracked in
the working tree — a log artifact, not code).

Wall-clock per arm (start = "Loaded config" line, end = "Frozen model
integrity verified" line immediately before completion):

| Arm | Start | End | Duration |
|---|---|---|---|
| α=0.75 | 19:15:48 | 20:00:50 | 45m 2s |
| α=0.50 | 20:03:05 | 20:48:06 | 45m 1s |
| α=0.25 | 20:50:22 | 21:34:36 | 44m 14s |
| α=0.00 | 21:36:52 | 22:23:04 | 46m 12s |

Total sweep wall-clock: 19:15:48 → 22:23:04 ≈ **3h 7m** for all 4 arms,
sequential, single GPU.

## 3. Final disqualification table

Copied verbatim from `Decisions.md`'s "Protocol Amendments — PI Sign-Off,
2026-07-24" section, commit `f0526b3`:

| α | Cond.1 dist-of-mean (≤8%) | Cond.2 KS p (>0.05) | Cond.3 primary HYP-F1 Δ (≥+3.0pp) | Cond.3 secondary Macro-F1 Δ (≥-1.5pp) | Overall |
|---|---|---|---|---|---|
| 0.75 | +3.47% PASS | 0.760186 PASS | -12.19pp FAIL | -3.37pp FAIL | **disqualified** (Cond. 3) |
| 0.50 | +14.15% FAIL | 0.240918 PASS | -96.05pp FAIL | -8.57pp FAIL | **disqualified** (Cond. 1 + Cond. 3) |
| 0.25 | -7.22% PASS | 0.000058 FAIL | -0.90pp FAIL | -6.92pp FAIL | **disqualified** (Cond. 2 + Cond. 3) |
| 0.00 | -1.37% PASS | 0.000025 FAIL | +3.95pp PASS | +0.17pp PASS | **disqualified** (Cond. 2) |

**Result: zero of the four relaxation arms qualify as `reliability_alpha*`.
`alpha=1.00` (`gate3_250_fixed`) remains the production design.**

## 4. Per-arm failure reasons

- **α=0.75**: fails Condition 3 only (both primary and secondary) — HYP-F1
  actually *worsens* by 12.19pp relative to baseline, and Macro-F1 drops
  3.37pp, exceeding the 1.5pp floor. Passes Condition 1 (A3 divergence
  +3.47%, well under 8%) and Condition 2 (r_morph KS p=0.760, no
  distributional shift).
- **α=0.50**: fails Condition 1 *and* Condition 3 — the only arm to fail
  Condition 1 at all (+14.15% A3 divergence, exceeding the 8% bound), and
  HYP-F1 collapses to exactly 0.0 (a full loss of the target class, not a
  partial regression). See Section 5 below — a specific, converging
  mechanism was identified for this arm.
- **α=0.25**: fails Condition 2 *and* Condition 3 — r_morph distribution
  shifts significantly from baseline (KS p=0.000058), and HYP-F1 is
  essentially flat (-0.90pp, below the +3.0pp bar) while Macro-F1 drops
  6.92pp.
- **α=0.00**: fails Condition 2 only — r_morph KS p=0.000025 (largest
  distributional shift of all four arms), despite passing both Condition
  1 (A3 divergence -1.37%) and Condition 3 (HYP-F1 +3.95pp, Macro-F1
  +0.17pp, both comfortably passing). This is the arm that comes closest
  to qualifying — disqualified on physiological non-inferiority alone.

## 5. Convergent finding — α=0.50's dual disqualification

Two independent lines of evidence point at the same underlying event in
the α=0.50 arm specifically:

- **A3 divergence (Condition 1)**: α=0.50 is the *only* arm whose A3
  Mahalanobis distance-of-mean-reward increases beyond the 8% bound
  (+14.15%), under the corrected (option-b) statistic.
- **HYP-F1 collapse (Condition 3)**: HYP-F1 in the generated-sample
  classifier eval drops to exactly 0.0 for this arm — the only arm where
  this happens (baseline/0.75/0.25/0.00 all score HYP-F1 ≥0.839).

Root cause, from `logs/reliability_alpha050/rl_training_log.csv`: HYP's
per-iteration reward signal (`r_diag`) holds steady around 0.687 for
nearly the entire 250-iteration run. At **iteration 247**, `grad_norm`
spikes to 0.54 (vs. a healthy ~0.05–0.15 baseline range) and `r_diag`
drops to 0.233; unlike an earlier, structurally similar spike at iteration
13 (`grad_norm`=1.10, fully self-corrected by the very next logged HYP
iteration), this one does **not** recover — `r_diag` falls further to
0.155 at iteration 249, with `r_a3` simultaneously spiking to 0.462 (vs.
typically <0.05). The checkpoint is saved at **iteration 250**,
immediately after this uncorrected destabilization. Plausible mechanism:
this arm's final checkpoint was frozen mid-instability rather than after
recovery, unlike every other arm's checkpoint — a genuine, decision-
independent finding, not an artifact of either statistic choice.

## 6. Protocol provenance

- **2026-07-22, `e545459`** — original pre-registration finalized and
  signed off with Dr. Balaji (primary endpoint, 5 metric-category
  thresholds, 3 reward-hacking disqualifying conditions, Pareto-dominance
  decision rule, two-stage seed plan).
- **2026-07-23, `dacf6e5`→amended** — four protocol ambiguities
  (A3 aggregation statistic, Condition 3 HYP-vs-macro scope, seed
  convention, "stays flat" epsilon) surfaced during verification. An
  early version of this commit incorrectly stated these as PI-resolved
  fact based on AI-fabricated content; caught and corrected (`f5a85a5`,
  "PENDING VERIFICATION") before it propagated further.
- **2026-07-24, email** — Dr. Balaji's real ruling on all four points,
  received directly, saved verbatim at
  `Roadmap/Stage_4_Optimization/correspondence/2026-07-24_balaji_protocol_ruling.md`.
  Notably rules **option (b)** (distance-of-mean-reward) for the A3
  statistic — the opposite of what the earlier fabricated content had
  assumed — materially changing the Condition 1 numbers.
- **2026-07-24, `f0526b3`** — final `Decisions.md` update: confirmed
  amendments, recomputed Condition 1/3 table (Section 3 above),
  "zero arms qualify" conclusion, citing Balaji's own pre-authorization
  of that outcome as "a valid scientific finding."

## 7. What was not run and why

- **Stage 2 (seeds 43/44)** was never launched. It verifies a qualifying
  `alpha*` against baseline across additional seeds — there is no
  qualifying candidate to verify, so it has nothing to do. Not a gap;
  a consequence of the zero-arms-qualify result.
- **No RL training beyond the original 5-run set** (`gate3_250_fixed`
  reused + the 4 sweep arms) was performed at any point in this
  analysis. All Condition 1/2/3 recomputation was done by
  re-deriving statistics from the existing `rl_training_log.csv` and
  `trtr_generated_eval.json` files already on disk — no new GPU time.
- A draft Stage-2 launcher script (`run_stage2_verification.sh`) was
  written in anticipation of a possible qualifying arm, then deleted
  once the final ruling made it moot (no candidate to plug in).

## 8. Upstream context

The reliability-weight redesign was motivated by a **real, sourced**
finding in this same document: `stage4_finetune_v1` (1000-iteration
run, pre-registered 2026-07-10) was scored **FAILURE** against its own
pre-registered criteria — HYP and OTHER classes' `r_diag` collapsed to
near-zero in the second half of training and did not recover by run's
end (HYP first-half mean 0.252 → second-half mean 0.060; OTHER 0.343 →
0.028; see `Decisions.md` lines ~1186–1217). This differs from an
earlier, self-correcting HYP anomaly seen in Gate 3 (iter 166–215,
recovered by run's end). This FAILURE finding is what prompted advisor
Dr. Balaji to request the 5-point reliability_alpha ablation instead of
picking a single fixed reward design.

**Caveat for the writeup**: `Decisions.md` separately documents (lines
~1493–1530) that a different, previously-referenced "earlier
conditioning-collapse investigation" — describing a NORM/OTHER
collapse-rate similarity claim — was searched for across every
`Decisions.md`/`Objectives.md`/`Experiment_Log.md`/`Reports/*.md` in this
repo and **could not be located or confirmed**. Do not cite that specific
claim (NORM/OTHER collapsing at similar rates despite different support)
in the report — it is not verified to exist in this project's committed
record. The `stage4_finetune_v1` FAILURE finding above is the real,
citable motivation; the broader "conditioning-collapse investigation"
reference is a separate, unresolved citation gap, not a validated source.
