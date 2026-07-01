# Stage 1 Objectives

1. Does the current codebase, run end-to-end on this machine, reproduce
   whatever numbers previous reports claim? If not, why not — and which
   source (report or code) is trusted going forward?
   - 1.5: Was training stopped before conditioning converged? (checkpoint
     verification across saved epochs)
2. Does increasing training data size (within PTB-XL only) measurably
   improve class-conditioning fidelity (accuracy, macro F1, collapse %,
   generation quality)? At what data size, if any, does it plateau?
   - 2.5: For a given data size, is a bad result a data-size ceiling or
     just insufficient epochs? (training curves per size)
3. When the class label changes, does the generated sample move toward the
   correct class's *semantic* region in MentorClassifier embedding space —
   or does it only change in raw signal magnitude/energy without moving
   toward the right class manifold?
   - 3.5: At which of the diffusion model's 6 Transformer blocks does the
     conditioning signal appear, strengthen, or get diluted?
4. Is the MentorClassifier a trustworthy arbiter of conditioning quality,
   or does it default to AFIB (or any other class) as a catch-all for
   out-of-distribution / noisy input regardless of true label?
   - 4.5: Can that AFIB-attraction effect be seen directly as drift in
     embedding space (real → noise → generated)?
5. Given 1-4, is the conditioning failure best explained by (a) insufficient
   data, (b) a loss-function/training deficiency, (c) an architectural
   limitation requiring cross-attention or latent diffusion, or (d)
   something a reward-shaping/RL approach could fix — and is Stage 2/3
   justified yet?

## Non-goals for Stage 1

- No new datasets.
- No architecture redesign.
- No RL training.
- No hyperparameter tuning beyond what's needed to run existing pipeline
  steps to completion.
