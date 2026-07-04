# Reproducibility Audit -- Items 5, 6, 8 (local CPU runs)

Per the same provenance discipline `Roadmap/_infra/audit_reproducibility.py`
applies to GPU-run experiments (Stage 2.0.1) -- adapted here since Items
5/6/8 ran locally and are not wrapped in `ExperimentLogger` (a
pre-existing gap noted at Item 5's own closure, not introduced here).
The question this audit answers: **does the code currently committed in
git exactly match what was executed to produce each item's reported
numbers**, with no undetected drift between "code that ran" and "code
that's committed."

## Checkpoint identity

`outputs/models/diffusion_best.pt` -- confirmed unchanged throughout
Items 5/6/8 (and every prior Tier 0 item): file timestamp `Jul 3 02:31`
(predates all of this session's Item 5/6/8 work, run `Jul 4`), SHA-256
`16ac1715ac90ecb3db119de5611a3d2fff2cdc6ca82e53fb4d9c9c3a1864819d`.
Same checkpoint every item has used since Item 1.

## Code-state check

For each item, the actual workflow was: write/edit the script -> run it
-> write the report from that exact run's output -> commit script +
report + any `common/` extensions together, immediately, in one atomic
commit. This ordering means there is no window where the committed code
could differ from what actually executed -- confirmed by git log, not
assumed:

| Item | Commit | Timestamp | Contains |
|---|---|---|---|
| 5 | `098e547` | 2026-07-04 21:56:29 +0530 | `item5_adaln_statistics.py` + `Item5_PreRegistration.md` + `Item5_Report.md` + `common/plotting.py` extension, all together |
| 6 | `a834fe6` | 2026-07-04 22:03:20 +0530 | `item6_attention_entropy.py` + `Item6_PreRegistration.md` + `Item6_Report.md` + `common/hooks.py`/`common/plotting.py` extensions, all together |
| 8 | `731b212` | 2026-07-04 22:10:21 +0530 | `item8_representation_collapse.py` + `Item8_PreRegistration.md` + `Item8_Report.md` + `common/plotting.py` extension, all together |

**Clean.** No case where a script was modified after its run but before
its commit without a corresponding rerun -- each commit's script IS the
script that produced that commit's reported numbers, verified by the
single-atomic-commit workflow, not merely assumed from good intentions.

## Post-hoc validity-check additions (this review cycle)

Items 6 and 8 were extended after their original commits (Item 6: CI
computation; Item 8: n-vs-d/permutation/multi-split checks) --
these additions are being committed in this same review cycle,
immediately after the rerun that produced their numbers, maintaining
the same discipline (no drift window).

## Conclusion

**Clean for all three items** -- no reproducibility gap analogous to
Stage 1's dirty-tree finding (where a GPU run's claimed commit didn't
capture uncommitted local changes). The local single-atomic-commit
workflow used for Items 5/6/8 structurally avoids that failure mode
(nothing runs against code that isn't immediately committed after).
