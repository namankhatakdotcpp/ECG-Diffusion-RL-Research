"""
Stage 3 -- cross-candidate comparison / reporting script.

Reads whichever of S3-001..S3-005 (and S3-006 once queued) currently have
completed mentor_eval output and produces one markdown table + one CSV,
both re-generated (not appended) on every run so they always reflect
current on-disk state. Safe to re-run at any point, including while other
candidates are still training -- a candidate with no metadata.json /
mentor_eval output yet is reported as "not yet evaluated", not skipped
silently and not fabricated.

Reuses run_stage3_queue.py's own VARIANT_BY_RUN_ID and RESULTS_ROOT
(single source of truth for which run_id maps to which variant) rather
than re-declaring the candidate list here.

Usage:
    python compare_candidates.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_stage3_queue import VARIANT_BY_RUN_ID, RESULTS_ROOT, BASELINE_METRICS_PATH  # noqa: E402

REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"
BASELINE_MANIFEST_PATH = REPO_ROOT / "outputs" / "mentor_review" / "baseline_manifest.json"
BASELINE_CKPT_PATH = REPO_ROOT / "outputs" / "models" / "diffusion_best.pt"

# The optimizer fix landed as two commits: 0294330 (name-based gamma/boost
# exclusion) then 432395c (tripwire guard). History is linear, so any
# training commit at-or-after 432395c is necessarily at-or-after 0294330
# too -- checking ancestry against the later commit is sufficient and
# avoids a two-commit special case.
OPTIMIZER_FIX_COMMIT = "432395c"


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit_exists(commit: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", commit], cwd=str(REPO_ROOT), capture_output=True,
    )
    return proc.returncode == 0


def classify_optimizer_config(commit: Optional[str]) -> str:
    """"pre-fix" / "post-fix" / "unknown", derived from actual git ancestry
    against OPTIMIZER_FIX_COMMIT -- never a hardcoded per-run_id literal, so
    this stays correct automatically (e.g. if S3-002 is ever retrained).
    """
    if not commit or commit == "unknown" or not _commit_exists(commit):
        return "unknown"
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", OPTIMIZER_FIX_COMMIT, commit],
        cwd=str(REPO_ROOT), capture_output=True,
    )
    return "post-fix" if proc.returncode == 0 else "pre-fix"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_candidate_row(run_id: str, variant: str) -> dict:
    run_dir = RESULTS_ROOT / run_id
    meta = _load_json(run_dir / "metadata.json")

    row = {
        "Candidate": run_id,
        "Variant": variant,
        "Generated Accuracy": None,
        "Generated Macro-F1": None,
        "Generated Macro-AUC": None,
        "Real-data Accuracy": None,
        "Delta vs. Baseline (accuracy)": None,
        "Optimizer Config": "unknown",
        "Status": "not yet evaluated -- no metadata.json found",
    }

    if meta is None:
        return row

    row["Status"] = meta.get("status", "unknown")
    row["Optimizer Config"] = classify_optimizer_config(meta.get("commit"))

    eval_dir = run_dir / "mentor_eval"
    real_metrics = _load_json(eval_dir / "classifier_real_eval.json")
    gen_metrics = _load_json(eval_dir / "classifier_generated_eval.json")

    if real_metrics is not None:
        row["Real-data Accuracy"] = round(real_metrics["accuracy"], 4)

    if gen_metrics is not None:
        row["Generated Accuracy"] = round(gen_metrics["accuracy"], 4)
        row["Generated Macro-F1"] = round(gen_metrics["macro_f1"], 4)
        row["Generated Macro-AUC"] = (
            round(gen_metrics["macro_auc"], 4) if gen_metrics.get("macro_auc") is not None else None
        )
    elif row["Status"] == "done":
        row["Status"] = "done -- generated-eval metrics missing (unexpected for status=done)"
    elif eval_dir.exists() and not gen_metrics:
        row["Status"] = row["Status"] + " -- generated-eval not yet produced"

    return row


def _resolve_baseline() -> tuple[Optional[float], dict]:
    """Returns (baseline_accuracy, provenance_info). provenance_info states
    plainly whether this came from a frozen, previously-written manifest
    or was computed live (i.e. no frozen manifest exists on this machine
    yet) -- never silently claim "checksum-verified" without one."""
    baseline_metrics = _load_json(BASELINE_METRICS_PATH)
    baseline_accuracy = baseline_metrics["accuracy"] if baseline_metrics else None

    manifest = _load_json(BASELINE_MANIFEST_PATH)
    if manifest is not None:
        provenance = {
            "source": "frozen baseline_manifest.json",
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "commit": manifest.get("git_commit"),
        }
    else:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        provenance = {
            "source": (
                "COMPUTED LIVE -- no frozen outputs/mentor_review/baseline_manifest.json "
                "found on this machine; run mentor_eval.run_all to produce one"
            ),
            "checkpoint_sha256": _sha256_file(BASELINE_CKPT_PATH),
            "commit": proc.stdout.strip() if proc.returncode == 0 else "unknown",
        }
    return baseline_accuracy, provenance


def build_rows() -> list[dict]:
    rows = []
    for run_id in sorted(VARIANT_BY_RUN_ID):
        variant = VARIANT_BY_RUN_ID[run_id]
        row = load_candidate_row(run_id, variant)
        rows.append(row)
    return rows


def _delta(row: dict, baseline_accuracy: Optional[float]) -> None:
    if row["Generated Accuracy"] is not None and baseline_accuracy is not None:
        row["Delta vs. Baseline (accuracy)"] = round(row["Generated Accuracy"] - baseline_accuracy, 4)


def write_markdown(rows: list[dict], baseline_accuracy: Optional[float], provenance: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Stage 3 -- Cross-Candidate Comparison\n")
    lines.append(
        "All candidates evaluated via the identical `mentor_eval.classification_validation` "
        "pipeline against the frozen baseline manifest "
        f"(checksum `{provenance['checkpoint_sha256']}`, commit `{provenance['commit']}`, "
        f"source: {provenance['source']}). Optimizer-config column reflects whether the run's "
        "training commit predates the gain-parameter weight-decay fix "
        f"(`0294330`/`{OPTIMIZER_FIX_COMMIT}`).\n"
    )
    if baseline_accuracy is None:
        lines.append(
            "**Baseline generated-data classifier metrics not found** "
            f"(expected at `{BASELINE_METRICS_PATH}`) -- Delta column cannot be computed.\n"
        )
    else:
        lines.append(f"Baseline (classifier trained on real data, evaluated on generated ECGs) accuracy: `{baseline_accuracy:.4f}`\n")

    header = [
        "Candidate", "Variant", "Generated Accuracy", "Generated Macro-F1",
        "Generated Macro-AUC", "Real-data Accuracy", "Delta vs. Baseline (accuracy)",
        "Optimizer Config", "Status",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        cells = [str(row[h]) if row[h] is not None else "--" for h in header]
        lines.append("| " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], out_path: Path) -> None:
    header = [
        "Candidate", "Variant", "Generated Accuracy", "Generated Macro-F1",
        "Generated Macro-AUC", "Real-data Accuracy", "Delta vs. Baseline (accuracy)",
        "Optimizer Config", "Status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_accuracy, provenance = _resolve_baseline()

    rows = build_rows()
    for row in rows:
        _delta(row, baseline_accuracy)
        if row["Status"].startswith("not yet evaluated"):
            print(f"[{row['Candidate']}] {row['Status']}")
        else:
            print(f"[{row['Candidate']}] status={row['Status']} optimizer_config={row['Optimizer Config']}")

    md_path = REPORTS_DIR / "Stage3_Comparison.md"
    csv_path = REPORTS_DIR / "Stage3_Comparison.csv"
    write_markdown(rows, baseline_accuracy, provenance, md_path)
    write_csv(rows, csv_path)
    print(f"\nWrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
