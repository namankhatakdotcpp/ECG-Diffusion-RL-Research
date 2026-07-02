# Stage 0 Pipeline Audit — Risk Register

Sorted by severity. Full evidence for every row in `Pipeline_Code_Audit.md`.

| # | Finding | Severity | File(s) | Status |
|---|---|---|---|---|
| 14 | The "~380 curated training sequences" described in every prior investigation report corresponds to NO code path anywhere in this repository — current tree, full history, all branches, all searched | **CRITICAL** | n/a — that's the finding | Confirmed absent; blocks `--sanity-check` and further GPU execution pending human decision |
| 5 | Tie-break on SCP-code confidence ties uses undocumented first-in-dict-order; affects 3,185/21,799 (14.6%) of multi-code records | **HIGH** | `step04_transformer_diffusion.py:486`, `step03_eda_and_class_mapping.py:144` | Confirmed with real data; fixed (commit `a9e6047`) |
| 6 | `config.yaml`'s class-taxonomy fallback (7 classes incl. AFIB) is already wrong vs. the real 6-class taxonomy on disk; silent trigger risk on GPU-server first run per Experiment 1 README's copy-instructions | **HIGH** (borders Critical) | `step04_transformer_diffusion.py:704-717`, `config.yaml:42-50`, `Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/README.md:25` | Confirmed — currently wrong, not just latent; fixed (commit `f8dba53`) |
| 8 | Checkpoints (135.1MB each, measured) accumulate indefinitely — never deleted within a run, and `snapshot_before_write` preserves the *entire* previous run's output on every re-run, including every `--sanity-check` call | MEDIUM | `step04_transformer_diffusion.py:912-925`, `utils/backup.py:22-45` | Confirmed, quantified; remote disk state not verifiable from here; fixed (commit `f78c6c2`) |
| 7 | `step04` silently drops records with empty `scp_codes`; `step03` assigns them to OTHER instead | LOW | `step04_transformer_diffusion.py:479-481`, `step03_eda_and_class_mapping.py:140-141` | Confirmed as code divergence; 0 records currently affected |
| 11 | "No CFG dropout in validation" is comment-enforced only, no runtime assertion | LOW | `step04_transformer_diffusion.py:884` | Confirmed; fix described not implemented |
| 12 | `n_leads=12` and `(1000,12)` shape hardcoded in several places instead of routed through `config.yaml` | LOW | 6+ files, see audit Finding 12 | Confirmed |
| 1 | Split leakage (train/val/test uses `strat_fold`, not random split) | N/A | `step02_preprocessing.py:379-385` | **Refuted** — clean |
| 2 | Normalization leakage (stats computed on train only) | N/A | `step02_preprocessing.py:480,485-487` | **Refuted** — clean |
| 3 | Sampling-rate inconsistency (100Hz vs. reported 500Hz) | N/A | grep across step01/02/mentor_eval | **Refuted for this repo** — single 100Hz pipeline throughout; the "500Hz" claim has no artifact here |
| 4 | step03/step04 label-mapping drift | N/A | `step03_eda_and_class_mapping.py:138-146`, `step04_transformer_diffusion.py:483-497` | **Refuted** — 0/21,799 disagreements, actually diffed |
| 9 | Seed placement / reproducibility | N/A | `step04_transformer_diffusion.py:1031-1037` | **Refuted** — correct order, verified |
| 10 | CFG dropout tensor aliasing | N/A | `step04_transformer_diffusion.py:843-845` | **Refuted** — `.clone()` present |
| 13 | Committed secrets (current tree + full git history, all branches) | N/A | repo-wide | **Refuted** — clean |

**Critical findings: 1** (added 2026-07-02, after the three fixes below
were already committed — see Finding 14. Discovered while sanity-checking
whether the tie-break fix's before/after diff was computed on the actual
training population or a different one. **Blocks `--sanity-check` and
further GPU execution pending human decision.**)
**High findings: 2** — both fixed and re-verified (commits `a9e6047`,
`f8dba53`), but their fix-verification diffs were computed on the full
17,418-record corpus, which Finding 14 established may not be the actual
~380-record population every prior conditioning-collapse finding was
built on. The High-finding fixes themselves are still correct and stand
independent of Finding 14 — the fixes just may not yet describe the
right population.
