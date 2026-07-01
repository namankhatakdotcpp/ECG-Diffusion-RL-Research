# Experiment 3 — Directional Conditioning Analysis

**Status: NOT YET RUN.** Code is ready at
`Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/directional_conditioning_probe.py`
— requires a trained diffusion checkpoint (Experiment 1's
`diffusion_best.pt`, and optionally Experiment 2's per-size checkpoints).
This file will be filled in from
`Outputs/Experiment_3_Directional_Probe/directional_scores_*.csv` once
available.

## Mathematical summary (full derivation in the script docstring)

For every ordered pair of generatable mentor classes (Normal, STEMI,
NSTEMI — AFIB excluded, no trained diffusion class):

```
delta_gen_i  = feat(gen_B_i) - feat(gen_A_i)      (paired: same noise seed for A and B)
delta_real   = mu_real(B) - mu_real(A)             (real class centroids in MentorClassifier
                                                     128-dim penultimate-feature space)
directional_score(A, B) = cosine_similarity(mean_i(delta_gen_i), delta_real)
```

`directional_score` close to **+1**: changing the label moves generated
samples toward the correct real-data direction (conditioning works
semantically, not just numerically). Close to **0**: the label changes the
output (consistent with a nonzero `conditioning_sensitivity_probe.py`
reading) but in a direction unrelated to the true class difference — this
is the "conditioning changes the output but not correctly" failure mode.
Negative: label moves samples in the wrong direction entirely.

## Planned analysis (to be filled in)

- Full 3x3 (minus diagonal) directional-score matrix — is it uniformly
  weak/zero (broad conditioning failure), or does it fail specifically for
  certain class pairs (e.g. never distinguishes STEMI from NSTEMI, but
  correctly separates Normal from both — architecturally interesting,
  since morphology differences are subtler between STEMI/NSTEMI proxies)?
- Distance-ratio sanity check: is generated class A's embedding at least
  closer to real centroid A than to real centroid B, even if not
  maximally aligned directionally?
- Feature trajectory figure: does the fixed-seed sweep across classes trace
  an organized path between real centroids, or jump erratically /
  cluster in one spot regardless of label?
- Cross-reference with Experiment 2: if run against multiple dataset
  sizes, does directional_score improve with more data, stay flat, or
  degrade? A flat/near-zero score across all data sizes would be strong
  evidence the conditioning pathway itself needs an architectural fix
  (cross-attention / latent diffusion), not more data.
