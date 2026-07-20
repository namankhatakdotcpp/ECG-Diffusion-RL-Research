# Experiment A: HYP→STTC Reward Replay — Summary

**Date:** 2026-07-20
**Status:** Complete

## Question
Is degrading STTC's diagnostic reward alone (via replay of HYP's recorded r_diag
values) sufficient to induce the kind of instability/collapse observed for HYP?

## Method
A RewardReplayOverride mechanism was implemented to substitute STTC's r_diag with
HYP's empirically recorded r_diag sequence from stage4_finetune_v1, consumed in
appearance order (Nth STTC occurrence receives HYP's Nth recorded value). Two modes
were implemented and tested: sequential (no wraparound, fails fast on exhaustion)
and cyclic (wraps when the donor sequence is exhausted).

## Runs
- experimentA_reward_replay_smoke: 200 iterations, sequential mode.
- experimentA_reward_replay_full: 1000-iteration request, sequential mode; exhausted
  and raised RuntimeError at iteration 213 as designed (44 STTC occurrences × 4
  rollouts = 176 draws against 177 available donor values).
- experimentA_reward_replay_cyclic: 1000 iterations, cyclic mode, completed without
  error (625 STTC overrides consumed against 177 donor values, wrapping ~3.5×).

## Results
- STTC r_diag mean: 0.205 (stage4_finetune_v1 baseline) → 0.167 (cyclic replay),
  confirming the replay materially affected the targeted reward component.
- Overall r_morph distribution: baseline mean 0.5866 (σ=0.1436) vs. replay mean
  0.5825 (σ=0.1473) — no practically meaningful shift.
- Per-class r_morph change (baseline → replay): STTC -0.019, CD +0.023, HYP -0.009,
  MI -0.015, NORM -0.004, OTHER -0.002. STTC's change is not an outlier relative to
  other classes.
- 3-consecutive r_morph<0.3 windows: 0/1000 (baseline) vs. 1/1000 (replay). The
  single replay-run window (iterations 196-198) is identical between the sequential
  and cyclic replay runs, indicating a shared-seed trajectory artifact rather than
  a recurring or intensifying pattern across the full 1000 iterations.
- Total rows with r_morph<0.3: 40/1000 (baseline) vs. 42/1000 (replay) —
  statistically indistinguishable.

## Conclusion
This experiment did not provide evidence supporting the reward-hacking/instability
hypothesis. The narrower supported conclusion is that altering one reward component
(STTC's diagnostic reward) alone was insufficient to induce detectable reward-hacking
behavior in this experiment.

## Caveat
The conclusions are based on a single baseline trajectory and replay experiments
that are not independent across random seeds. Consequently, the results should be
interpreted as evidence from one experimental realization rather than as an estimate
of the average effect across training runs. If stronger evidence is needed (e.g.,
for publication), an independent-seed replication is the recommended next step.
