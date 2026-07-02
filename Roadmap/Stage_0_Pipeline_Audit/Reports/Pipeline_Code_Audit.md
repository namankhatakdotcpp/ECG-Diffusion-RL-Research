# Stage 0 — Pipeline Code Audit

**Type:** Code review only. No training run, no code modified, no fixes applied.
**Scope:** step01-04, plus a narrow repo-wide pass (secrets, step03/step04
duplicate logic, config consistency).
**Method:** Every finding below is backed by an exact line-number citation
and/or an actual command run against the real `data/ptbxl/` data in this
checkout — not inferred from reading code alone. Where a prior claim
(from chat history or Roadmap/ docs) is refuted by evidence, that is
stated plainly, per this project's standing rule to trust code/data over
prior narrative.

---

## Finding 1 — Split leakage: NOT PRESENT (verified)

**Severity: N/A (refuted — documented here so the check is on record)**
**File:** `step02_preprocessing.py:379-385`

```python
train_mask = df["strat_fold"].isin(train_folds)
val_mask   = df["strat_fold"].isin(val_folds)
test_mask  = df["strat_fold"].isin(test_folds)

df_train = df[train_mask]
df_val   = df[val_mask]
df_test  = df[test_mask]
```

`train_folds`/`val_folds`/`test_folds` come from `cfg.ptbxl.train_fold`
(`[1..8]`), `val_fold` (`[9]`), `test_fold` (`[10]`) — `config.yaml:52-54`.
This is the official PTB-XL patient-stratified `strat_fold` column, not a
random per-record split. **No split leakage.** This was the single
highest-priority item in this audit (per instruction, checked first) and
it is clean.

---

## Finding 2 — Normalization leakage: NOT PRESENT (verified)

**Severity: N/A (refuted)**
**File:** `step02_preprocessing.py:480, 485-487`

```python
mu, sigma = _compute_train_stats(X_train_raw)   # line 480 -- train split only
...
X_train = _normalise(X_train_raw, mu, sigma, clip_range)   # line 485
X_val   = _normalise(X_val_raw,   mu, sigma, clip_range)   # line 486
X_test  = _normalise(X_test_raw,  mu, sigma, clip_range)   # line 487
```

`mu`/`sigma` are computed once, from `X_train_raw` only (`_compute_train_stats`
docstring at line 190: "Compute per-lead mean and std over the training
set"), then applied unchanged to all three splits. No per-split
recomputation. Confirmed the same file is loaded downstream: `step04_transformer_diffusion.py:753,1002`
and `step05_baseline_eval.py:197` both load `processed_dir / "preprocessing_stats.json"`
— the single file step02 writes at line 541. **No normalization leakage.**

---

## Finding 3 — Sampling rate: single consistent 100Hz pipeline; the "500Hz" report has no corresponding artifact in this repository

**Severity: N/A (refuted as a current risk) — but see the note on what could NOT be verified**
**Files:** `step01_data_load_and_visualise.py:264`, `step02_preprocessing.py:423`,
and every `wfdb.rdrecord()` call site under `mentor_eval/` (13 files, all
grepped).

Every single waveform-reading call site in this repository — step01, step02,
and all 13 files under `mentor_eval/` — reads `row["filename_lr"]`
(PTB-XL's "low resolution," i.e. 100Hz, naming convention). `filename_hr`
(500Hz) and `records500/` are referenced **nowhere** in this codebase:

```
$ grep -rn "filename_hr\|records500" --include="*.py" .
(no output)
```

`config.yaml:28`'s comment ("100 or 500 available; 100 is standard")
overstates what the code actually supports — `cfg.ptbxl.sampling_rate` is
read (`step02_preprocessing.py:279`) and used for filter design/logging,
but the actual **file selection is hardcoded to `filename_lr`**
regardless of what `sampling_rate` is set to. If someone set
`sampling_rate: 500` expecting the pipeline to switch to high-resolution
data, it would silently keep reading 100Hz files while logging/filtering
as if they were 500Hz — a latent, currently-inert footgun (see Finding 8).

**On the "MI-classification experiment report states 500Hz" claim:** I
could not find any code path in this repository that reads 500Hz data.
Per the same standard applied in the Stage 2 Verification Gate
(`Roadmap/Stage_2_Architecture_Investigation/Reports/Verification_Gate_Report.md`),
a claim with no corresponding artifact in this checkout should not be
treated as describing this codebase — it most likely describes Track B
(the teammate's separate classifier codebase, per project memory), a
different environment, or is simply inaccurate. **Not something I can
confirm or refute further from this machine.**

---

## Finding 4 — step03 vs step04 label-mapping: NO drift for the current dataset (verified by actually running both and diffing all 21,799 records)

**Severity: N/A (refuted) — but see Finding 5 for a related, real edge-case bug**
**Files:** `step03_eda_and_class_mapping.py:138-146` (`_assign_primary`),
`step04_transformer_diffusion.py:483-497` (inner loop of `_load_class_labels`)

Per the master prompt's explicit instruction ("verify by running both
aggregation paths over the same records and diffing the resulting labels,
not by reading the code alone"), I replicated both selection algorithms
exactly and ran them over all 21,799 real records in
`data/ptbxl/ptbxl_database.csv`:

- step03's `_assign_primary`: picks the highest-confidence SCP code found
  in the **raw** `scp_statements.csv` `diagnostic_class` mapping (plus
  `_HARD_OVERRIDES = {"AFIB": "AFIB", "AFLT": "AFIB"}`, verified at
  `step03_eda_and_class_mapping.py:75-78`), tie-broken by strict `>`
  (first-encountered-in-dict-order wins ties) — then the result is folded
  through `_remap()` to collapse any superclass below `min_class_samples`
  into `"OTHER"`.
- step04's `_load_class_labels`: picks the highest-confidence code found
  in `class_mapping.json` (the **already-collapsed** output step03
  writes), same strict-`>` tie-break.

**First attempt at this comparison found 8 apparent disagreements** — all
turned out to be a bug in my own replication script (I initially forgot
to merge `_HARD_OVERRIDES` into the candidate code map before comparing).
After correcting the replication to match the verified real constant:

```
Total records checked: 21799
Disagreements (corrected replication): 0
```

**Zero disagreements across every record in the real dataset.** The two
code paths are mathematically equivalent under the current config and
data (both use the same tie-break rule, and `class_mapping.json`'s key
set matches `code_map`'s key set exactly under `min_class_samples=200`
with `OTHER` always retained). This refutes the "second, independently-
drifting implementation" concern for the *current* codebase and data —
documented as a negative result, not glossed over.

---

## Finding 5 — Tie-break order-dependence: CONFIRMED, and affects ~14.6% of records with multiple SCP codes

**Severity: HIGH**
**File:** `step04_transformer_diffusion.py:486` (and the identical pattern
at `step03_eda_and_class_mapping.py:144`)

```python
if mapped and mapped in name_to_idx and conf > best_conf:   # strict >
    best_cls, best_conf = mapped, conf
```

Strict `>` means the **first** code encountered in `scp.items()`
iteration order wins any exact confidence tie — not a documented,
clinically-meaningful tie-break rule, just whatever order the SCP codes
happen to appear in that record's `scp_codes` string field.

**Checked against real data, not assumed:**

```
Records with >=2 scp codes: 21090
Records with an EXACT confidence tie among their codes: 9125   (mostly trivial: ties among 0.0-confidence non-diagnostic codes)

Records where a TIE AT THE TOP CONFIDENCE spans >1 DIFFERENT final class
(i.e. the assigned label is genuinely order-dependent): 3185 / 21799 (14.6%)
```

Examples (record_id: scp_codes -> tied top classes):
```
87  {'NDT': 100.0, 'IRBBB': 100.0}          -> {STTC, CD}
173 {'LVH': 100.0, 'ISC_': 100.0}           -> {HYP, STTC}
191 {'LVH': 100.0, '1AVB': 100.0}           -> {HYP, CD}
234 {'ASMI': 100.0, 'ANEUR': 100.0, '1AVB': 100.0} -> {MI, STTC, CD}
235 {'NORM': 100.0, 'LAFB': 100.0}          -> {CD, NORM}
```

This is real and substantial: **~1 in 7 records with multiple diagnostic
codes has a label that would change if Python's dict-iteration order
changed** (e.g. a different `_parse_scp` implementation, a different
Python version's dict semantics for equal-priority insertion, or simply
re-serializing the CSV). Python 3.7+ dict order is insertion-order-stable,
so this is *deterministic* given the current CSV — not randomly
flaky between runs — but it is not a *principled* clinical tie-break,
and the fact that this exact bug is independently duplicated in two files
means a fix in one place without the other would silently reintroduce the
Finding-4 drift risk.

**Proposed minimal fix (not implemented):** define an explicit,
documented tie-break — e.g. alphabetical by code, or by a clinical
severity ranking — and apply it in both `_assign_primary` and
`_load_class_labels` identically (ideally via one shared function, given
Finding 4 shows they must stay in lockstep).

---

## Finding 6 — Stale `class_names.json`/`class_mapping.json` fallback: config.yaml's fallback values are ALREADY WRONG relative to the real, established class taxonomy

**Severity: HIGH (borders Critical — see operational note)**
**Files:** `step04_transformer_diffusion.py:704-717`, `config.yaml:42-50`,
`outputs/processed/class_names.json`

```python
# step04_transformer_diffusion.py:707-717
if cls_names_path.exists() and cls_map_path.exists():
    with open(cls_names_path)   as f: class_names   = json.load(f)
    with open(cls_map_path) as f: class_mapping = json.load(f)
else:
    log.warning("class_names.json / class_mapping.json not found — "
                "falling back to config.ptbxl.classes. Run step03 first for best results.")
    class_names   = list(cfg.ptbxl.classes)
    class_mapping = {c: c for c in class_names}
```

```yaml
# config.yaml:42-50
  classes:
    - NORM
    - MI
    - STTC
    - CD
    - HYP
    - AFIB       # <-- included here
    - OTHER
  n_classes: 7   # <-- declared here, separately
```

```
$ cat outputs/processed/class_names.json
["NORM", "MI", "STTC", "CD", "HYP", "OTHER"]     # <-- 6 classes, no AFIB
```

**The fallback path is already wrong, today, unconditionally** — not a
hypothetical future-staleness scenario. `config.yaml`'s `ptbxl.classes`
lists 7 classes including a dedicated `AFIB`; the real, currently-correct
`class_names.json` (produced by step03, which actually applies
`min_class_samples=200` and folds AFIB's 103 records into `OTHER`) has 6.
If the fallback path is ever taken — i.e. `class_names.json`/
`class_mapping.json` are missing from `outputs/processed/` — step04 would
silently train a **7-class model** (class_emb table shape `(8, model_dim)`
with the null token) against a class taxonomy the actual data cannot
support at the configured `min_class_samples` threshold, with no error
raised anywhere (the shape assertion at
`step04_transformer_diffusion.py:784-793` only checks internal
consistency between the embedding tensor and whatever `n_classes` was
computed — it cannot detect that `n_classes` itself is wrong).

Separately: `cfg.ptbxl.n_classes: 7` (the standalone field) is **read
nowhere in the codebase** —
```
$ grep -rn "ptbxl\.n_classes" --include="*.py" .
(no output)
```
— dead config that additionally misstates the real class count (6), which
could mislead a future reader independent of the fallback-path risk above.

**Operational relevance — why this is HIGH not just a style nit:**
`Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/README.md:25` tells
whoever runs Experiment 1 on the GPU server: *"`outputs/processed/*.npy`
are already present and can be copied over rather than regenerated"* —
this instruction mentions only `.npy` files. If someone follows it
literally and the GPU server's `outputs/processed/` doesn't already have
`class_names.json`/`class_mapping.json`/`class_counts.json` from a prior
full step01-03 run, this exact fallback triggers on the very first real
GPU run this project executes.

**Proposed minimal fix (not implemented):** (a) fix `config.yaml`'s
`ptbxl.classes` list to match the real 6-class taxonomy (or remove it/
`n_classes` entirely and make the fallback fail loudly instead of
silently proceeding with a wrong-but-plausible value); (b) update the
Experiment 1 README to explicitly list all four `outputs/processed/*.json`
files alongside the `.npy` files.

---

## Finding 7 — `_load_class_labels` silently drops records with empty/unparseable `scp_codes`; step03 assigns them to `OTHER` instead

**Severity: LOW (confirmed as a real code divergence; 0 records currently affected)**
**Files:** `step04_transformer_diffusion.py:479-481`, `step03_eda_and_class_mapping.py:140-141`

```python
# step04_transformer_diffusion.py:479-481
scp = _parse_scp(ptbxl_db.at[eid_int, "scp_codes"])
if not scp:
    continue        # <-- record silently DROPPED, not counted anywhere
```
```python
# step03_eda_and_class_mapping.py:140-141
if not scp_dict:
    return "OTHER"   # <-- record KEPT, assigned OTHER
```

Checked against real data:
```
Total records: 21799
Records with EMPTY/unparseable scp_codes: 0
```

**Zero records currently trigger this divergence** — it is a real,
verified difference in behavior between the two files, but inert on the
current PTB-XL database CSV. Flagged as Low severity: latent, not
currently impactful, but worth aligning for robustness against a future
PTB-XL version or any manually-edited metadata.

**Proposed minimal fix (not implemented):** make step04 assign `OTHER`
(matching step03) instead of dropping, when `"OTHER"` is a valid class —
it already has this exact fallback for the "no code matched" case two
lines later (`step04_transformer_diffusion.py:489-493`); the empty-dict
case should follow the same rule for consistency.

---

## Finding 8 — Checkpoint accumulation, quantified

**Severity: MEDIUM (real, bounded, but compounds with the near-capacity GPU server and the sanity-check workflow)**
**Files:** `step04_transformer_diffusion.py:912-925` (periodic checkpoint save,
never deleted), `utils/backup.py:22-45` (`snapshot_before_write`)

**Measured directly** (not estimated) by constructing the real model/
optimizer/EMA state and saving an actual checkpoint with the exact
structure step04 saves:

```
n_params = 8,431,892
One checkpoint file size: 135.1 MB
```
(model state_dict + EMA shadow copy + AdamW optimizer state
[exp_avg + exp_avg_sq, ~2x model size] — measured empirically, not just
computed from param count, since AdamW state size depends on which
tensors actually received a gradient update).

**Within a single full run** (`n_epochs=200`, `save_every=25`): 8 periodic
`diffusion_ckpt_ep*.pt` files + 1 `diffusion_best.pt` = up to **~1.22GB**,
and periodic checkpoints are **never deleted** — no cleanup code exists
anywhere in `step04_transformer_diffusion.py` (checked: no `os.remove`/
`unlink`/`Path.unlink` call site touching `diffusion_ckpt_ep*.pt`
anywhere in the file).

**This compounds across runs, not just within one**, via
`snapshot_before_write` (`utils/backup.py:22-45`): every time
`outputs/models/` already has content from a previous run,
`snapshot_before_write` **renames the entire directory** to
`outputs/models_backup_<old_run_id>/` (line 41: `shutil.move`) rather
than deleting anything. This is called at the top of both
`step04_transformer_diffusion.main()` and my own
`run_experiment_1_for_real.py` wrapper. **Every re-run — including every
`--sanity-check` invocation — preserves the full previous run's ~1.2GB
(or ~540MB for a 3-epoch sanity check at `save_every=1`: 3 periodic + 1
best = 4 x 135.1MB) under a new backup directory that is never cleaned
up automatically.**

**Could not verify:** actual current disk usage on `himtenduh` — I have
no SSH access to that machine from this environment. The disk-headroom
check already added to the Step 2 runbook
(`df --output=avail`/`--output=pcent` gate before training) catches this
at the point of use; this finding quantifies the accumulation *rate* so
that gate's threshold can be interpreted correctly (~1.2GB/full-run,
~540MB/sanity-check-run, compounding indefinitely without manual cleanup).

**Proposed retention policy (not implemented):** keep last N periodic
checkpoints (e.g. N=2) + `diffusion_best.pt` always; delete older
`diffusion_ckpt_ep*.pt` as newer ones are saved. Separately: give
`snapshot_before_write`-created backup directories a retention policy too
(e.g. keep only the single most recent backup, or none at all for
`--sanity-check` runs specifically, since those are explicitly
not-evidence and have no reason to be preserved).

---

## Finding 9 — Random seed placement: CORRECT (verified)

**Severity: N/A (refuted)**
**File:** `step04_transformer_diffusion.py:1031-1037`

```python
def main() -> None:
    cfg = load_config()
    log = get_logger("step04_transformer_diffusion", cfg=cfg)
    set_seed(cfg.seeds[0])                                  # line 1034
    snapshot_before_write(Path(cfg.paths.outputs.models))
    best_val_loss = train(cfg, log)                         # line 1037 -- DataLoader/model init happen inside this call
```

Traced `train()`'s body (`step04_transformer_diffusion.py:690-775`):
between the function's start and the first nondeterministic operation
(`WeightedRandomSampler`/`DataLoader` construction at lines 762-772, model
weight init inside `ECGTransformerDiffusion.__init__` at line 775), every
intervening step (path setup, `class_names.json` load, `.npy` load,
`_load_class_labels`) is deterministic file I/O with no RNG calls. **`set_seed()`
correctly precedes every source of randomness**, when `train()` is
reached via `main()`.

**Caveat that applies to my own wrapper, not step04 itself:**
`run_experiment_1_for_real.py` calls `train()` directly (not via
`step04.main()`), so it must independently call `set_seed()` before doing
so — confirmed it does
(`Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py`,
`set_seed(seed)` precedes the `ExperimentLogger`/`step04_train()` call).
Any *future* wrapper around `train()` must maintain this ordering
independently — `train()` itself does not call `set_seed()`, by design
(it's `main()`'s job), so this is a contract callers must uphold, not
something `train()` enforces.

---

## Finding 10 — CFG null-class dropout tensor aliasing: CORRECT (re-verified)

**Severity: N/A (refuted)**
**File:** `step04_transformer_diffusion.py:843-845`

```python
null_mask = torch.bernoulli(torch.full((B,), p_uncond, device=device)).bool()
batch_cls = batch_cls.clone()          # line 844
batch_cls[null_mask] = model.null_class_index
```

`.clone()` still present before the in-place mutation. No aliasing of the
`DataLoader`-yielded tensor. Confirmed unchanged from prior reading.

---

## Finding 11 — "No CFG dropout in validation" is comment-enforced, not runtime-enforced

**Severity: LOW**
**File:** `step04_transformer_diffusion.py:884`

```python
for batch_x, batch_cls in val_loader:
    batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
    # Validation always uses real labels — no CFG dropout here (intentional).
```

Confirmed: this is a comment, not a runtime check. Currently correct
because `val_loader` simply never has dropout logic applied to it
(dropout only happens in the training loop above, lines 839-845) — but
nothing would catch a future edit that accidentally moved or duplicated
dropout logic into this block.

**Proposed minimal fix (not implemented, suggested per the master
prompt's request):**
```python
assert batch_cls.max().item() < model.null_class_index, (
    "validation batch contains the null class index -- CFG dropout must "
    "never be applied during validation"
)
```
placed immediately after `batch_cls = batch_cls.to(device)` inside the
validation loop specifically.

---

## Finding 12 — Config-consistency: minor hardcoded values outside `config.yaml`

**Severity: LOW**

- `n_leads = 12` is hardcoded in 6+ locations (`step04_transformer_diffusion.py:192,562`;
  `Roadmap/Stage_1_Diagnosis/Code/common_probes.py:30`;
  `Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/layerwise_direction_probe.py:130`;
  `mentor_eval/conditioning_sensitivity_probe.py:64`; `mentor_eval/subband_similarity_metrics.py:73`),
  and `config.yaml` has no `n_leads` field at all. Low severity: 12-lead
  ECG is a fixed physical/clinical standard for this dataset, not a
  parameter anyone would realistically tune — but worth noting as
  "config.yaml is not actually the single source of truth" for this one
  value, contrary to `README.md`'s stated convention.
- The signal shape `(1000, 12)` is hardcoded as a validation-check
  literal in 6 `mentor_eval/*.py` files (`cfg_sweep.py:59`,
  `classification_validation.py:106`, `conditioning_diagnostic.py:68`,
  `lead_class_figures.py:68`, `dataset_audit.py:68,71`) rather than
  derived from `cfg.ptbxl.signal_length`. If `signal_length` were ever
  changed, these would silently start rejecting all real data as
  "wrong shape" with no error message tracing back to the config change
  that caused it.

**Proposed minimal fix (not implemented):** route both through
`cfg.ptbxl.signal_length`/a new `cfg.ptbxl.n_leads` field, or explicitly
document in `config.yaml` that lead count is a fixed dataset property
intentionally not configurable.

---

## Finding 13 — No committed secrets found

**Severity: N/A (clean)**

Scanned the current tree (all `.py`/`.yaml`/`.yml`/`.json`/`.md`/`.sh`
files) and the **full git history across all commits on all local
branches** (`git log --all -p`) for common secret patterns (API keys, AWS
keys, private key headers, W&B tokens, SSH key material, password/secret
assignments). All matches in the current tree were false positives (the
word "token" used in its ML/transformer sense). Git history scan and a
separate scan for ever-committed credential-looking filenames
(`.env`, `.pem`, `.key`, `id_rsa`, etc.) both returned zero results.
**Clean.**

---

## Finding 14 (added 2026-07-02) — The "~380 curated training sequences" described in prior reports corresponds to NO code path anywhere in this repository

**Severity: CRITICAL**

Every original investigation report (the consolidated LaTeX report,
Table 3 specifically) states the diffusion model trains on "a curated
subset of 380 sequences (85 held out for validation) — far smaller than
full PTB-XL (≈21,799 records)," with a stated per-class breakdown
(NORM=231, STTC=50, CD=45, MI=36, HYP=16, OTHER=2) and a stated
justification ("a full 200-epoch run completes in under ten minutes").

**Exhaustively searched for the mechanism that would produce this subset
— found nothing:**

```
$ grep -in "max_samples\|curated\|subset_size\|n_samples_per_class\|sample_cap\|max_per_class\|380" config.yaml
(no output)

$ grep -rniE "curated|downsampl|subsampl|max_samples|sample_cap|max_per_class" \
    step01_data_load_and_visualise.py step02_preprocessing.py \
    step03_eda_and_class_mapping.py step04_transformer_diffusion.py config.yaml utils/*.py
(no output -- only unrelated matches: step03's morphology-analysis sampling,
 which subsamples for a FIGURE, not for training data, and is unrelated to
 the training population)
```

Specifically checked and ruled out as the source of any capping:
- `step02_preprocessing.py:_load_split` (lines 403-464) — processes
  **every** record in each fold's split; the only `continue` in the loop
  is for unreadable files, not a per-class cap.
- `step04_transformer_diffusion.py:ECGDataset.__init__` (lines 503-514) —
  wraps whatever `X`/`labels` arrays are passed to it verbatim, no
  filtering or capping logic inside.
- `step04_transformer_diffusion.py:train()` — loads
  `outputs/processed/X_train.npy` in full; the only reduction applied is
  `_load_class_labels`'s class-validity filter (drops unrecognized-code
  records), not a per-class size cap.
- **The real `X_train.npy` on this machine has shape `(17418, 1000, 12)`**
  — confirmed directly, not estimated. 17,418, not 380.
- **Full git history, all commits, all branches**
  (`git log --all -p -- '*.py' '*.yaml'`) searched for
  `curated`/`max_samples_per_class`/`subset_size` and equivalents — the
  only "curated" match anywhere in this repo's history is an unrelated
  docstring about SCP-code mapping ("hand-curated map"), not a
  data-subsetting mechanism. No commit, on any branch, ever added
  subsetting logic matching this description.

**Conclusion:** the ~380-record curated diffusion-training population
described in prior reports does not correspond to any code path that
exists now, or ever existed, in this repository's version-controlled
history. This is the same class of problem the Stage 2 Verification Gate
caught (unfindable "0.91→0.24 layer-wise decay" and "~2-7% generated
accuracy" numbers) and this same audit already caught once (the
unfindable "500Hz" claim, Finding 3) — a specific number with confident
prose provenance and no located artifact — but at materially higher
severity here, because **this is not a downstream metric, it's the
claimed identity of the training population itself.** Every
conditioning-collapse finding, every per-class comparison (e.g.
"NORM (231) collapses at the same rate as OTHER (2)"), and every
Stage 1/2 interpretation built on that narrative currently has no
verifiable foundation in this codebase.

**Two non-alarmist explanations, not distinguishable from code alone:**
1. The 380-record population was constructed out-of-band (a notebook, an
   ad-hoc script, or a manual step run once on a different machine/session)
   and the specific `X_train.npy`/`record_ids_train.npy` it used were
   never committed (`outputs/` is gitignored by design — these are
   regenerable artifacts, not source). A later full-corpus step02 run
   (the one that produced the current `outputs/processed/` on this
   machine) would have silently overwritten it with no trace, since
   nothing about that file pair is under version control.
2. The "380 curated sequences" narrative in the prior report describes
   an intended/aspirational methodology that was never actually
   implemented as described, or was implemented differently than
   documented.

**UPDATE (2026-07-02, same day): resolved by independent corroboration,
not further code archaeology.** A completely unrelated, much earlier
session in this project's history — sizing local-machine compute budget,
with no awareness the 380-record claim existed and not investigating this
question at all — measured real wall-clock throughput and reported:
*"CPU training: 23 seconds/step at batch=32. 200 epochs × 544 steps/epoch
would take ~29 days."* Checked directly against what this audit
independently confirmed:

```
17,418 / 32 = 544.3125 -> floor 544 (step04's DataLoader uses drop_last=True,
                                       confirmed at step04_transformer_diffusion.py:793-796,
                                       so 544 is the exact real per-epoch step count,
                                       not a rounding coincidence)

380 / 32 -> floor 11 steps/epoch, 200 epochs = 2,200 total steps
           (consistent with the "under ten minutes" framing)
17,418 / 32 -> floor 544 steps/epoch, 200 epochs = 108,800 total steps
           (not remotely consistent with "under ten minutes" on any realistic GPU)
```

**Two independent measurements — one via exhaustive code+git-history
search (this audit), one via direct runtime benchmarking (an unrelated
earlier session, measuring the same `X_train.npy` for an unrelated
reason) — agree with each other on ~17,418 records, and both actively
contradict the "380 curated sequences, sub-ten-minute 200-epoch run"
framing.** This is stronger than absence-of-evidence alone: it is a
positive, independent, unprompted measurement landing on the same number
this audit found by a completely different method.

**Disposition: do not attempt further reconstruction of the 380-record
population's origin.** The real, long-standing training population is
confirmed as ~17,418 records (the full mapped PTB-XL train split), with
no subsetting mechanism ever present in this codebase. The entire
"Investigation Timeline" section of the original consolidated report
(embedding-scale experiment, AdaLN-Zero, decoupled signals, the CFG
sweep, every conditioning-collapse percentage, the AFIB-attractor
findings) is **historical narrative only** as of this finding — not
"pending reproduction" in the neutral sense used elsewhere in this
project (where code exists and simply hasn't been run yet), but
specifically flagged as **contradicted by direct evidence regarding the
population it claims to have run against**. No number from that report
should be cited as motivation for Stage 2 priority ordering — including
the LayerScale hypothesis's promotion to first-in-line in
`Stage2_Master_Prompt.md`, which was partly justified by that report's
layer-wise decay claim — until independently re-derived from a real run
in this repository.

**Structural consequence for the earlier-flagged dataset-scaling bug:**
the old Experiment 2 run that reported `n_train_records_actual=380`
identically across all six requested sizes makes even less sense in
light of this finding than when it was first caught — it was never a
slicing-call bug reusing a fixed 380-record array, because **no code path
in this repository was ever capable of producing a 380-record training
set at all**, let alone six times consistently. Those ledger entries are
historical narrative, same as the rest of the pre-audit findings — not
evidence to reconcile with. If a genuine dataset-scaling experiment is
wanted later, it needs to be *built* (a real, documented, version-controlled
subsetting mechanism — e.g. a `max_samples_per_class` config field with a
stratified sampler, committed as its own reviewable change), not
retrofitted onto the old script.

**`--sanity-check` is authorized** — against the real, confirmed
17,418-record corpus. Do not artificially subsample to 380 to match the
old narrative; that would be fitting the data to a conclusion, backwards
from how this entire audit has been conducted.
