# MASTER PROMPT — Stage 2: Architecture Investigation (v2, corrected)

Supersedes v1. Changes from v1, each driven by review: the dataset-scaling
bug fix is no longer presupposed (ranked hypotheses instead), a Repository
Audit stage was inserted before Tier 0, a Representation Collapse analysis
was added to Tier 0 with an empirically-verified methodological
correction, and ADRs replace a single monolithic final report for
architectural decisions specifically.

**Correction applied 2026-07-02, before this prompt is executed (folder
structure was decided after this prompt was originally drafted):** this
document plans BOTH Tier 0 (measurement only, no retraining — this is
Stage 2's actual job) and Tier 1 (real architecture changes — this is
**Stage 3**'s job) in one place for planning continuity, so whoever
executes Tier 0 has the full picture when writing ADRs. But their
**outputs go to different stage folders**:

- Tier 0 (items 1-8): `ExperimentLogger(root_dir=Roadmap/Stage_2_Architecture_Investigation/...)`
- Tier 1 (items 9-11): `ExperimentLogger(root_dir=Roadmap/Stage_3_Architecture_Improvements/...)`
  — even though it's described in this same file. Tier 1 is not executed
  in this session regardless (see the closing instruction below); when it
  does run later, it runs as Stage 3 against Stage 3's own folder, not
  nested under Stage 2's `Reports/`, `Logs/`, or ledger.

```
You are acting as a senior ML research engineer conducting a peer-review-
grade architectural investigation of a conditional ECG diffusion model.
Treat every claim as something a hostile reviewer will try to break.

════════════════════════════════════════════════════════════════════════
STAGE 2.0 — VERIFICATION GATE (mandatory, run first)
════════════════════════════════════════════════════════════════════════

Do NOT trust any Stage 1 finding you have not personally reproduced from
artifacts that exist on disk right now. Prior chat summaries and prior
reports are hypotheses to verify, not facts to build on.

1. Locate Roadmap/Stage_1_Diagnosis/Reports/results_ledger.jsonl and
   MASTER_LOG.md. If either is missing, STOP and report exactly what is
   missing before doing anything else.
2. Read the "Flags" section of MASTER_LOG.md first. Any flagged
   experiment is DISQUALIFIED as evidence for Stage 2 until fixed and
   rerun. Do not cite disqualified numbers anywhere in Stage 2 reasoning.
3. For every non-flagged "success" experiment, independently recompute at
   least one headline number directly from its raw CSV/artifact (not from
   a report's prose summary) and confirm it matches. Log this as
   stage2_0_verification_<original_experiment_id>, with
   ExperimentLogger(root_dir=Roadmap/Stage_2_Architecture_Investigation/...).
4. Produce Roadmap/Stage_2_Architecture_Investigation/Reports/
   Verification_Gate_Report.md. Nothing past this point may begin until
   every item in it is CONFIRMED or explicitly NEEDS-REPRODUCTION.

════════════════════════════════════════════════════════════════════════
STAGE 2.0.5 — REPOSITORY AUDIT (mandatory, run second)
════════════════════════════════════════════════════════════════════════

The Verification Gate checks whether REPORTED NUMBERS reproduce. This
stage checks whether the ARTIFACTS THEMSELVES are what they claim to be
-- a different, complementary failure mode. This project has already been
burned by exactly this once: the EMA bug was "wrong weights silently
used," discovered only when someone happened to compare weight-std
between the EMA shadow and live weights. Do not rely on that kind of luck
a second time.

Run: python Roadmap/_infra/audit_repository.py outputs/ \
       Roadmap/Stage_2_Architecture_Investigation/Reports/Repository_Audit_Report.md

This checks, automatically, without any ECG-specific logic:
  - duplicate checkpoints (byte-identical files under different names --
    a save that silently didn't update weights)
  - stale generated samples / figures (artifact older than the checkpoint
    it's supposedly derived from)
  - metadata cross-check (diffusion_architecture.json's best_val_loss vs.
    diffusion_training_log.csv's actual minimum -- two independently
    written files that should agree)
  - orphan checkpoints (no matching training-log epoch entry)

Any "error"-severity finding BLOCKS Stage 2 until resolved. Do not
proceed past this stage with unresolved errors, even if they seem minor
-- the entire reason this stage exists is that "seems minor" is exactly
how the EMA bug survived as long as it did.

════════════════════════════════════════════════════════════════════════
STAGE 2.1 — MANDATORY LOGGING INFRASTRUCTURE
════════════════════════════════════════════════════════════════════════

Place experiment_logger.py, build_master_log.py, audit_repository.py, and
representation_metrics.py at Roadmap/_infra/. Every experiment in this
stage -- including cheap, no-retrain, checkpoint-only analyses -- runs
inside ExperimentLogger. See README_LOGGING.md for the full contract.
Do not write ad-hoc logging. Do not summarize results by hand from stdout.

════════════════════════════════════════════════════════════════════════
PROJECT CONTEXT (hypotheses pending Stage 2.0, not fact)
════════════════════════════════════════════════════════════════════════

Reported, NOT YET independently confirmed by you in this session:
  - Real-data MentorClassifier: ~83% accuracy, ~0.95 macro AUC.
  - Generated-data classifier accuracy: ~2-7%, near-chance or below.
  - A layer-wise probe reportedly found conditioning magnitude decaying
    from ~0.91 (block 1) to ~0.24 (block 6) while direction-consistency
    stayed ~1.00. If confirmed, this is the single most actionable claim
    in the investigation -- verify it first (Tier 0, item 1).
  - Dataset-scaling results are DISQUALIFIED until Stage 2.2 below
    identifies and fixes the actual root cause and reruns.

Confirmed independently by reading the current codebase directly (not
hearsay): AdaLN-Zero, decoupled time/class signals via cond_film, and
class-embedding weight-decay exclusion are all present in step04_
transformer_diffusion.py on the main branch. Re-verify this yourself
against the current state of the file before relying on it -- code
changes, chat summaries don't always keep up.

════════════════════════════════════════════════════════════════════════
STAGE 2.2 — DATASET-SCALING ROOT CAUSE (ranked hypotheses, not assumed)
════════════════════════════════════════════════════════════════════════

The observed symptom: every requested dataset size (380, 1000, 2500,
5000, 10000, full) produced IDENTICAL n_train_records_actual=380,
identical wall-clock training time (576-579s), and training loss curves
matching to 5 decimal places.

Do not assume a specific root cause. Check these in order, ranked by
probability given the evidence pattern, fastest-to-verify first:

  1. MOST LIKELY (evidence: identical wall-clock time across all
     "sizes" -- if data volume genuinely changed, per-epoch compute time
     should change measurably; it didn't, meaning the compute workload
     never changed, meaning the data never changed). The "size" parameter
     is used only to LABEL the sweep's output row/filename, but the
     actual training call still references a fixed-size preprocessed
     array loaded once, with the size parameter never reaching an actual
     subsampling/slicing call. VERIFY: grep the sweep script for every
     place `requested_size` (or equivalent) is used; confirm whether it
     reaches a `X_train[:n]`-style slice or an equivalent subsample call.
     FASTEST CHECK: add `assert len(X_train) == requested_size`
     immediately after wherever data loading happens, run for ONE size
     only, see if it fails before spending further GPU time.
  2. A DataLoader or Dataset object is constructed ONCE outside a loop
     over sizes, and later loop iterations that intend to rebuild it with
     a different subsample instead silently reuse the first iteration's
     object (a common closure/scoping bug). VERIFY: check whether the
     DataLoader/Dataset construction is inside or outside the per-size
     loop body.
  3. The size parameter reaches training correctly, but the EVALUATION
     step (generation + classification) is pointed at a fixed, pre-cached
     checkpoint from a different experiment rather than the checkpoint
     just produced by that specific sweep iteration. LESS LIKELY than (1)
     given that TRAINING loss and wall-clock time are also identical, not
     just eval metrics -- if this were the sole cause, training loss
     curves across different actual dataset sizes should still differ.
  4. LEAST LIKELY given the strength of the evidence: a config override
     is passed on the command line but never actually threaded into the
     config object the training function reads (silent no-op override).

Once the actual cause is identified (not assumed), fix it, add the
assertion from (1) as a permanent regression guard in the script, and
rerun the full sweep. Log the root-cause investigation itself as
exp2_0_root_cause_investigation, and the corrected sweep as a new
experiment family (not overwriting the disqualified original ledger
entries -- they stay in the ledger as a record of what was ruled out).

NOTE ON THIS PROJECT'S ACTUAL Experiment 2 CODE: before assuming this
symptom is present, re-verify against the real
Roadmap/Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py
on disk. That script does build a fresh, size-specific stratified subset
per iteration and reports per-size wall-clock training time as a metric --
if the identical-time symptom described above does not actually appear in
this project's real ledger once Experiment 2 has been run on the GPU
server, say so explicitly and do not manufacture a root-cause
investigation for a bug that didn't reproduce here. This section's ranked
hypotheses remain the right DEBUGGING METHOD if and when a similar symptom
is ever observed -- they are not a claim that this exact bug exists in the
current codebase.

════════════════════════════════════════════════════════════════════════
PRIORITY ORDER — CHEAPEST AND MOST DIAGNOSTIC FIRST
════════════════════════════════════════════════════════════════════════

Tier 0 items require NO new GPU training and write all outputs under
Roadmap/Stage_2_Architecture_Investigation/. Do not start any Tier 1 item
until every Tier 0 item has a logged, verified result and a written
verdict.

--- Tier 0: no retrain, pure analysis of existing checkpoints ---
--- (Stage 2 -- ExperimentLogger root_dir = Stage_2_Architecture_Investigation) ---

1. VERIFY the layer-wise magnitude/direction claim. Cheapest, highest-
   priority item in the stage.

2. LayerScale / gain hypothesis test. If (1) confirms magnitude decay
   with preserved direction:
       gain_i = ||adaLN_output_i|| / ||adaLN_output_1||
   across all six blocks. Geometric-ish decay is consistent with repeated
   LayerNorm renormalization recurring at every block, not just at input
   (the mechanism already confirmed for INPUT injection in the original
   embedding-scale experiment). Measurement only -- no fix yet.

3. Residual-path attenuation. Per block:
       ||block_output - block_input|| / ||block_input||
   computed for same-seed, same-timestep, class-label-only-differs pairs,
   separating the conditioning-swap component from the total residual
   update. Independent, complementary evidence to (2).

4. Activation and gradient norms. Gradient norm at class_emb.weight vs.
   every other parameter group, at the checkpoint nearest epoch 200 vs.
   nearest epoch 25 if available. Tests whether the class embedding's
   gradient signal was ever competitive, independent of the forward-pass
   sensitivity probe already run in Stage 1.

5. AdaLN / FiLM parameter statistics. Per-block adaLN weight-matrix
   Frobenius norm; fraction devoted to scale1/scale2 vs shift1/shift2.
   A second, independent way to test the channel-capacity question
   Stage 1's CFG sweep addressed indirectly.

6. Attention entropy and attention-map inspection.
       H = -sum(p * log(p))
   averaged over heads and query positions, for NORM-labeled vs.
   STEMI-labeled generation with identical noise/seed/timestep. Near-
   identical entropy/maps regardless of class label argues AGAINST
   cross-attention being sufficient by itself -- adding cross-attention
   to an already class-blind attention mechanism would not obviously help.

7. Class-embedding evolution across training. Pairwise
   ||class_emb.weight[c] - class_emb.weight[c']|| across saved per-epoch
   checkpoints. Shrinking or flat pairwise distances over training is
   direct evidence the embedding space itself isn't differentiating,
   independent of everything downstream.

8. Representation Collapse Analysis (Fisher ratio + linear probe).
   Use representation_metrics.py. Run BOTH fisher_ratio() and
   linear_probe_accuracy() at EVERY block -- do not use a low Fisher
   ratio to decide to skip the linear probe at that block. This is not
   a stylistic preference: fisher_ratio() is trace-based (sums variance
   equally across all D=256 dimensions) and was empirically shown during
   development of this tool to UNDER-REPORT separability that lives in a
   low-dimensional subspace against many noise-only dimensions -- exactly
   the kind of feature geometry a 256-dim transformer hidden state could
   plausibly have. A low Fisher ratio only means "not separable via a
   naive equal-weighting of all dimensions"; only linear_probe_accuracy()
   (which learns which dimensions matter) is the confirmatory test.
     - min_class_count defaults exclude any class with too few samples IN
       THE SPECIFIC PROBE BATCH from fisher_ratio (default: <5) and
       linear_probe_accuracy (default: <10) automatically. This is
       correct behavior, not a bug: a variance or decision-boundary
       estimate from a handful of samples is not an estimate. CORRECTION:
       an earlier version of this prompt claimed "OTHER (n=2 project-wide)"
       as the motivating example -- that number is wrong; OTHER's real
       project-wide count is train=254/val=29/test=30
       (outputs/processed/class_counts.json), well above both thresholds.
       The guard matters for small PER-BLOCK PROBE batches (e.g.
       n_gen=20/class), not because OTHER itself is rare in the dataset.
     - linear_probe_accuracy auto-applies a PCA dimensionality guard when
       n_train < 5x the feature dimension, to avoid the interpolation
       regime where an unregularized linear classifier can badly overfit
       (near-perfect train accuracy on a labeling with NO true
       relationship to the features, with test accuracy correspondingly
       unreliable — verified during development to collapse toward/below
       chance, not up to 100%; either way the train/test gap, not a
       single accuracy number, is what to trust). Any result where the
       `pca_components` field is populated must state that fact wherever
       it's cited -- it's on reduced features.
     - Pool hidden states via mean-pooling over the 600 tokens per sample
       to get one feature vector per sample at each block. Hold timestep
       t fixed (or bucket by comparable noise level) across the batch
       used for a single Fisher-ratio/probe computation -- do not mix
       samples from very different t values into one calculation, since
       t itself changes hidden-state statistics independent of class.

--- Tier 1: requires new training, one variable changed at a time ---
--- (STAGE 3 -- ExperimentLogger root_dir = Stage_3_Architecture_Improvements,
     NOT Stage_2_Architecture_Investigation, despite being described here) ---

Do not start Tier 1 until every Tier 0 item has a logged, verified result
and a written verdict. Do not start Tier 1 in this session at all (see the
closing instruction) -- these items are specified now only so whoever
executes Tier 0 can write ADRs with the full downstream picture in mind.

9. If Tier 0 supports the LayerScale hypothesis: implement a minimal,
   learnable per-block scalar gain on the adaLN modulation output (init
   at 1.0, ~6 extra parameters total). Retrain. Compare against the
   current checkpoint using the full existing evaluation suite plus the
   Stage 1 directional sensitivity probe.

10. If Tier 0 supports a genuine channel-capacity limit (attention maps
    class-blind, adaLN weights collapsed, class embeddings not
    differentiating, linear-probe accuracy near chance even at early
    blocks): cross-attention to a learnable disease token becomes
    justified by evidence. Implement as a controlled ablation against the
    Tier-0-verified baseline, not as a default.

11. Auxiliary classifier-guidance loss during training. Test
    independently of (9) and (10) -- do not stack architectural and loss
    changes in the same training run, or you cannot attribute the result.

════════════════════════════════════════════════════════════════════════
REQUIRED RIGOR FOR EVERY EXPERIMENT
════════════════════════════════════════════════════════════════════════

For every item above, before writing implementation code:
  1. State the hypothesis precisely.
  2. State what result would confirm it and what result would reject it.
  3. State the mathematical definition of every metric before computing
     it (formulas given above; extend the same precision to anything not
     fully specified).
  4. Implement, wrapped in ExperimentLogger, root_dir matching the STAGE
     that item actually belongs to (Tier 0 -> Stage 2; Tier 1 -> Stage 3).
  5. Classify the result as:
       VERIFIED     — matches prediction, effect size large relative to
                       a stated noise/null estimate
       SUPPORTED    — trends in predicted direction but small/noisy;
                       needs a second experiment before being load-bearing
       PRELIMINARY  — insufficient data to classify either way
       REJECTED     — contradicts the prediction
  6. If REJECTED: move to the next item. Do not quietly reframe a
     rejected hypothesis as "partially supported" -- that exact move
     already happened once in this project's history with the
     0.2%-sensitivity number.

════════════════════════════════════════════════════════════════════════
GIT AND COMPUTE DISCIPLINE
════════════════════════════════════════════════════════════════════════

  - Branch: stage2-architecture-investigation.
  - One commit per completed, logged experiment. Conventional Commits.
  - No AI attribution anywhere, ever, under any circumstances.
  - Never overwrite an existing checkpoint, log, or report file.
  - Before any Tier 1 retrain: confirm free disk space (>10% headroom
    after the save), use utils/backup.py's snapshotting so Tier 1 runs
    don't clobber Tier 0 artifacts referencing the current
    diffusion_best.pt, and dry-run at the target batch size before
    committing to a multi-hour queue on the shared GPU server.

════════════════════════════════════════════════════════════════════════
DELIVERABLES
════════════════════════════════════════════════════════════════════════

Roadmap/Stage_2_Architecture_Investigation/   (Tier 0 only)
    Reports/
        Verification_Gate_Report.md
        Repository_Audit_Report.md
        results_ledger.jsonl              (auto-generated)
        MASTER_LOG.md                     (auto-generated)
        ADR/
            ADR-000-template.md           (already present)
            ADR-001-<topic>.md
            ...  (one per architectural decision reached from Tier 0
                   evidence, immutable once Accepted -- supersede, don't
                   edit)
        Stage2_Final_Report.md            (synthesis + pointer to each
                                            ADR, written last)
    Logs/
    Code/
        <one subfolder per experiment, matching experiment_id>

Roadmap/Stage_3_Architecture_Improvements/    (Tier 1 -- separate session,
                                                not this one)
    Reports/, Logs/, Code/  -- same shape, populated only once Stage 2
    Tier 0 has a written, non-Preliminary verdict justifying a specific
    Tier 1 item.

Stage2_Final_Report.md must include, for every Tier-0 priority-order item:
  hypothesis, verdict, single strongest piece of evidence WITH a pointer
  to its ledger entry (not a restated number), and a final ranked list of
  architectural bottlenecks tagged by which verdict tier supports each.
  Nothing above "Preliminary" support becomes a Stage 3 priority.

Every accepted architectural decision gets its own ADR. Do not fold
multiple independent decisions into one ADR file.

Do not begin Stage 3 architectural implementation work in this session.
Stage 2 ends with a ranked, evidenced list and a set of ADRs -- not a
rewritten model.
```
