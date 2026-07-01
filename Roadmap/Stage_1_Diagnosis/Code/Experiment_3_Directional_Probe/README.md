# Experiment 3 — Directional Conditioning Analysis

## Why the existing sensitivity probe isn't enough

`mentor_eval/conditioning_sensitivity_probe.py` measures
`||eps_A - eps_B||` — a magnitude only. A model can score highly on that
probe while still moving generated samples in a semantically meaningless
(or wrong) direction. This experiment adds the missing piece: does changing
the label move samples toward the **correct class's region of
MentorClassifier embedding space**, not just move them *somewhere*?

## Method (see the script docstring for full derivation)

1. Real per-class centroids in 128-dim MentorClassifier embedding space.
2. Paired generation: same noise seed, two class labels → isolates the
   label's effect on the embedding from sampling randomness.
3. `directional_score(A, B) = cos(mean(feat(gen_B) - feat(gen_A)), mu_real(B) - mu_real(A))`
   — the mathematical core of this experiment. +1 = correct direction,
   0 = unrelated direction (this is what "conditioning changes output but
   not correctly" looks like), -1 = inverted.
4. A geometric distance-ratio sanity check independent of the classifier's
   decision boundary.
5. A single fixed-seed feature trajectory across all classes, visualized via
   PCA, as a qualitative complement to the numeric scores.

AFIB is excluded from all generation-based comparisons (no trained
diffusion class — see `Architecture.md` §4); it can only appear as a real
centroid for visual reference, never as a generated point.

## How to run

Requires a diffusion checkpoint (Experiment 1's `outputs/models/diffusion_best.pt`,
or any of Experiment 2's per-size checkpoints to see if directional accuracy
scales with data):

```bash
python Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/directional_conditioning_probe.py \
    --ckpt outputs/models/diffusion_best.pt --tag baseline

# Optional — repeat against a dataset-scaling checkpoint from Experiment 2:
python Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/directional_conditioning_probe.py \
    --ckpt Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/checkpoints/size_full/diffusion_best.pt \
    --tag size_full
```

Reuses the same cached MentorClassifier (`outputs/conditioning_analysis/mentor_classifier.pt`)
as `embedding_visualization.py`, so results are on the same measuring
instrument as the rest of Stage 1 — run Experiment 1 first so that cache
exists (or let this script train and cache it itself).

## What to hand back

- `Roadmap/Stage_1_Diagnosis/Outputs/Experiment_3_Directional_Probe/directional_scores_*.csv`
  and `directional_probe_raw_*.json`
- `Roadmap/Stage_1_Diagnosis/Figures/Experiment_3_Directional_Probe/*.png`

## Experiment 3.5 — Layer-wise Direction Probe

`layerwise_direction_probe.py` answers a different question: not "does the
final output move correctly" but "at which of the 6 TransformerBlocks does
the class-conditioning signal appear, strengthen, or get washed out?" It
never touches MentorClassifier or does DDIM sampling — it hooks each
block's output directly and measures, per layer: (a) normalized
conditioning magnitude, and (b) **direction consistency** — whether the
class label pushes that layer's representation in a *stable* direction
across different random noise draws (near 1) or an effectively random one
(near 0, i.e. noise-like). A layer with high magnitude but low consistency
is the layer where conditioning has *an* effect but not a *meaningful* one.

```bash
python Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/layerwise_direction_probe.py \
    --ckpt outputs/models/diffusion_best.pt --tag baseline
```

Cheap (pure forward passes, no generation) — safe to run immediately after
Experiment 1's checkpoint exists, before Experiment 3's full paired-generation
run if GPU time is tight.

Hand back: `layerwise_probe_*.csv`, `layerwise_probe_raw_*.json`,
`layerwise_magnitude_*.png`, `layerwise_direction_consistency_*.png`.
