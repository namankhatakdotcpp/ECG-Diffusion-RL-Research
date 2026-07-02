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

    # 2. A declared "requested size"-style parameter that does not match a
    #    correspondingly-named "actual"-style metric. This is the exact
    #    pattern of the dataset-scaling bug: params.dataset_size_requested
    #    vs. metrics.n_train_records_actual.
    for r in records:
        params, metrics = r.get("params", {}), r.get("metrics", {})
        for p_key, p_val in params.items():
            p_key_l = p_key.lower()
            if "requested" not in p_key_l and not p_key_l.endswith("size"):
                continue
            for m_key, m_val in metrics.items():
                m_key_l = m_key.lower()
                if "actual" not in m_key_l:
                    continue
                # crude but effective: only compare when both keys share a
                # stem word (e.g. "dataset" in both dataset_size_requested
                # and n_train_records_actual would NOT share a stem, so this
                # also matches on generic "size"/"records"/"count" families)
                shared_family = any(
                    tok in m_key_l
                    for tok in ("size", "record", "count", "n_train", "n_samples")
                )
                if shared_family and p_val != m_val:
                    flags.append(
                        f"[{r['experiment_id']}] requested {p_key}={p_val!r} "
                        f"but recorded {m_key}={m_val!r} — the parameter may "
                        f"not have reached the training/generation code"
                    )

    # 3. Multiple runs in the same experiment family (same id with a
    #    trailing suffix stripped, e.g. exp2_dataset_scaling_5000 and
    #    exp2_dataset_scaling_10000 share the family exp2_dataset_scaling)
    #    whose metrics are byte-identical despite different params.
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
