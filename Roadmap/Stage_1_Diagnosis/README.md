# Stage 1 — Diagnosis

Several architectural conditioning experiments (FiLM, decoupled AdaLN, CFG)
have already been implemented and are in the code today, but no one has
verified — on real generated samples, on this codebase's current state —
whether conditioning actually works. Before any new architecture is
attempted, Stage 1 answers **why conditioning fails**, using only what
already exists: PTB-XL (no new datasets), the current model, and the
MentorClassifier.

Stage 1 contains five core experiments, run in order, plus four
lower-cost sub-experiments added to close specific gaps each core
experiment leaves open:

1. **Baseline Reproduction** — establish ground truth for the current
   codebase's actual behavior (training, generation, accuracy, macro F1,
   collapse %, CFG sweep, sensitivity).
   - *1.5 Checkpoint Verification* — was training stopped too early?
2. **Dataset Scaling** — progressively increase PTB-XL training data
   (380 → 1000 → 2500 → 5000 → 10000 → full ~17.4k) to see if conditioning
   improves with more data alone.
   - *2.5 Training Curves* — loss/sensitivity/collapse/F1 over training
     for every dataset size, not just final numbers.
3. **Directional Conditioning Analysis** — a new probe measuring whether
   changing the class label moves generated samples toward the correct
   class's embedding manifold (not just changes output magnitude).
   - *3.5 Layer-wise Direction Probe* — where inside the 6 Transformer
     blocks does the conditioning signal appear or disappear?
4. **MentorClassifier Verification** — test whether AFIB behaves as an
   out-of-distribution reject bucket, which would invalidate any
   conditioning conclusion drawn from AFIB behavior.
   - *4.5 Feature Drift Visualization* — real → noise → generated in one
     shared embedding space, making the AFIB-attraction finding visible.
5. **Decision Report** — synthesize 1-4 into a scientifically justified
   answer: more data / better loss / cross-attention / latent diffusion /
   RL — or "not yet decidable."

No architecture changes happen in this stage. See `Objectives.md` for the
exact questions each experiment must answer and `Decisions.md` for the
running verdict.

## Running everything with one command

```bash
bash Roadmap/Stage_1_Diagnosis/run_stage1.sh
```

Runs Experiments 1 → 1.5 → 2(+2.5) → 3 → 3.5 → 4 (skipped automatically if
its output already exists) → 4.5, then assembles
`Reports/Stage1_Results_Digest.md` — a mechanical table of every result
found on disk. That digest is not the same as `Stage1_Final_Report.md`:
the digest is what a script can safely automate (pulling numbers into one
place); the final report's interpretation (is architecture the bottleneck?
should Stage 2 proceed?) still requires reading those numbers and is
written by Claude once they exist, per the project's research rules
(negative results are reported honestly, conclusions must be backed by
evidence — see `Decisions.md`).
