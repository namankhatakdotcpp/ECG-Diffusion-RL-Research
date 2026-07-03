"""
build_master_log.py

Regenerates Reports/MASTER_LOG.md from Reports/results_ledger.jsonl.

Also runs a small set of automated sanity checks against the ledger and
surfaces any anomalies in a dedicated "Flags" section at the top of the
digest. This is a generic, experiment-agnostic mechanism — it does not know
anything about ECGs or diffusion models. It exists specifically to catch
the class of bug where a dataset-scaling (or any other) experiment silently
never varies the thing it claims to vary: e.g. a run logged with
params={"dataset_size_requested": 5000} but
metrics={"n_train_records_actual": 380} would previously only be caught by
a human noticing the CSV numbers looked suspiciously identical across runs.
Here it is caught automatically and blocks the affected results from being
treated as evidence until someone signs off.

Fixed 2026-07-02: check 2 (requested-vs-actual mismatch) originally paired
ANY param key ending in "size" or containing "requested" against ANY
metric key containing "actual" that shared a generic token. This produced
a real, confirmed false positive: run_experiment_1_for_real.py logs
params={"batch_size": 32, ...} and metrics={"n_train_records_actual": 17418,
...} in the SAME ledger record -- "batch_size" ends in "size", and
"n_train_records_actual" contains both "actual" and "n_train"/"record", so
the old logic flagged two semantically unrelated numbers as if one were
supposed to equal the other. Verified this would have fired on the very
first real run of that script (both keys are logged together there; see
Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py
lines ~133-161).

Fix: replaced the generic heuristic with REQUESTED_ACTUAL_PAIRS, an
explicit {requested_param_key: (actual_metric_key, tolerance)} registry --
only declared pairs are compared, and only beyond a documented tolerance
(see REQUESTED_ACTUAL_PAIRS' comment for how the dataset-scaling tolerance
was derived -- measured empirically against this repo's real class
distribution, not assumed). New experiment families that want this check
must add their own pair explicitly; that's intentional, not a gap --
implicit heuristic matching is what caused the false positive.

Check 3 (constant-metric-across-a-family) also produced a false positive:
run_dataset_scaling.py's "n_generated" metric is a fixed evaluation budget
(samples-per-class x class-count), constant BY DESIGN across every dataset
size in a scaling sweep -- not a sign the sweep parameter failed to reach
the code. Fixed via CONSTANT_BY_DESIGN_METRICS, an explicit allowlist
(not a blanket loosening of the check, which would also hide a metric
that's supposed to vary and doesn't).

Can be run standalone:

    python build_master_log.py Roadmap/Stage_1_Diagnosis

or imported and called as `regenerate(root_dir)`, which is what
ExperimentLogger.__exit__ does after every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# {requested_param_key: (actual_metric_key, tolerance)} -- only these
# explicit pairs are compared by check 2. Add new pairs here as new
# experiment families need this check; do not widen the matching logic
# back to a generic heuristic (see the module docstring for why).
#
# dataset_size_requested / n_train_records_actual tolerance: derived by
# actually running run_dataset_scaling.py's stratified_subset() against
# this repo's real per-class training counts (outputs/processed/
# class_counts.json) for target sizes 380/1000/2500/5000/10000 -- observed
# deviations were 1, 1, 1, 0, 2 (max 2). Each class's share is independently
# rounded (max_samples_per_class-style round()-per-class, 6 classes), so
# the theoretical worst case is bounded by ~n_classes/2 = 3. Tolerance set
# to 5 for headroom beyond both the measured and theoretical bound, not
# because 5 itself was derived from anything -- if this ever needs to be
# tighter, re-run the same empirical check rather than guess.
REQUESTED_ACTUAL_PAIRS: dict[str, tuple[str, int]] = {
    "dataset_size_requested": ("n_train_records_actual", 5),
}

# Metrics that are legitimately constant across every run in an experiment
# family, by design -- not a sign the varying parameter failed to reach
# the underlying code. Add an entry here only with a comment explaining
# why it's expected to be constant (see run_dataset_scaling.py's
# evaluate_with_fixed_classifier(), which returns n_generated =
# n_gen_per_class x n_generatable_classes -- a fixed evaluation budget,
# independent of training set size).
CONSTANT_BY_DESIGN_METRICS: set[str] = {"n_generated"}


def _load_ledger(root_dir: Path) -> list[dict[str, Any]]:
    ledger_path = root_dir / "Reports" / "results_ledger.jsonl"
    if not ledger_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _run_sanity_checks(records: list[dict[str, Any]]) -> list[str]:
    """
    Three generic anomaly detectors, each aimed at a specific class of
    silent pipeline bug rather than a genuine scientific finding.
    """
    flags: list[str] = []

    # 1. Any run that did not finish with status == "success".
    for r in records:
        if r["status"] != "success":
            flags.append(
                f"[{r['experiment_id']}] finished with status={r['status']!r} "
                f"— see {r.get('log_file', 'no log file recorded')}"
            )

    # 2. An explicitly-registered requested-vs-actual pair (see
    #    REQUESTED_ACTUAL_PAIRS) whose values disagree beyond the
    #    registered tolerance. This is the exact pattern of the original
    #    dataset-scaling bug: params.dataset_size_requested vs.
    #    metrics.n_train_records_actual. Only registered pairs are
    #    compared -- no generic key-matching (see module docstring for
    #    why: the old heuristic falsely paired unrelated keys like
    #    batch_size against n_train_records_actual).
    for r in records:
        params, metrics = r.get("params", {}), r.get("metrics", {})
        for p_key, (m_key, tolerance) in REQUESTED_ACTUAL_PAIRS.items():
            if p_key not in params or m_key not in metrics:
                continue
            p_val, m_val = params[p_key], metrics[m_key]
            if not isinstance(p_val, (int, float)) or not isinstance(m_val, (int, float)):
                continue
            if abs(p_val - m_val) > tolerance:
                flags.append(
                    f"[{r['experiment_id']}] requested {p_key}={p_val!r} "
                    f"but recorded {m_key}={m_val!r} (tolerance={tolerance}) "
                    f"— the parameter may not have reached the "
                    f"training/generation code"
                )

    # 3. Multiple runs in the same experiment family (same id with a
    #    trailing suffix stripped, e.g. exp2_dataset_scaling_5000 and
    #    exp2_dataset_scaling_10000 share the family exp2_dataset_scaling)
    #    whose metrics are byte-identical despite different params.
    #    Skips CONSTANT_BY_DESIGN_METRICS (e.g. a fixed evaluation budget
    #    that is SUPPOSED to be the same across every run in the family).
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        family = r["experiment_id"].rsplit("_", 1)[0]
        by_family.setdefault(family, []).append(r)
    for family, runs in by_family.items():
        if len(runs) < 3:
            continue
        metric_keys: set[str] = set()
        for r in runs:
            metric_keys.update(r.get("metrics", {}).keys())
        for mk in metric_keys:
            if mk in CONSTANT_BY_DESIGN_METRICS:
                continue
            values = [r["metrics"][mk] for r in runs if mk in r.get("metrics", {})]
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(numeric) >= 3 and len(set(numeric)) == 1:
                flags.append(
                    f"[{family}] metric {mk!r} is byte-identical across "
                    f"{len(numeric)} runs with different params — check "
                    f"whether the varying parameter actually reached the "
                    f"underlying training/generation code"
                )

    return flags


def _format_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No experiments logged yet._\n"

    all_metric_keys: list[str] = []
    for r in records:
        for k in r.get("metrics", {}):
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    headers = ["Experiment", "Stage", "Status", "Duration (s)", "Timestamp"] + all_metric_keys
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in sorted(records, key=lambda x: x["timestamp_start"]):
        row = [
            r["experiment_id"],
            r["stage"],
            r["status"],
            str(r["duration_seconds"]),
            r["timestamp_start"],
        ]
        for k in all_metric_keys:
            row.append(str(r.get("metrics", {}).get(k, "")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def regenerate(root_dir: Path) -> Path:
    root_dir = Path(root_dir)
    records = _load_ledger(root_dir)
    flags = _run_sanity_checks(records)

    out_path = root_dir / "Reports" / "MASTER_LOG.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = [f"# Master Experiment Log — {root_dir.name}\n"]
    parts.append(
        "_Auto-generated from `results_ledger.jsonl`. Do not hand-edit — "
        "edits are overwritten on the next run._\n"
    )

    parts.append("## Flags (automated sanity checks)\n")
    if flags:
        parts.append("**These require human review before the affected results can be trusted:**\n")
        for f in flags:
            parts.append(f"- ⚠️ {f}")
    else:
        parts.append("_No anomalies detected._")
    parts.append("")

    parts.append("## All Experiment Runs\n")
    parts.append(_format_table(records))

    parts.append("\n## Failures and Crashes\n")
    failures = [r for r in records if r["status"] != "success"]
    if not failures:
        parts.append("_None._")
    else:
        for r in failures:
            parts.append(f"### {r['experiment_id']} ({r['status']})")
            parts.append(f"- Log: `{r['log_file']}`")
            if r.get("exception"):
                parts.append("```")
                parts.append(r["exception"].strip())
                parts.append("```")
            parts.append("")

    out_path.write_text("\n".join(parts))
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    written = regenerate(target)
    print(f"Wrote {written}")
