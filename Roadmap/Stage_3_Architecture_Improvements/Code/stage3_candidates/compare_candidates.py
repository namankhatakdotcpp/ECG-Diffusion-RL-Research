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
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

STAGE2_CODE_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from run_stage3_queue import VARIANT_BY_RUN_ID, RESULTS_ROOT, BASELINE_METRICS_PATH  # noqa: E402
from common.io import load_config  # noqa: E402
from model_variants import build_variant_model  # noqa: E402

REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"

# Single source of truth for column order/names -- write_markdown() and
# write_csv() both use this so the two outputs can never silently drift
# apart from each other.
TABLE_HEADER = [
    "Candidate", "Variant", "Params (M)", "Generated Accuracy", "Generated Macro-F1",
    "Generated Macro-AUC", "Real-data Accuracy", "Delta vs. Baseline (accuracy)",
    "Similarity Cosine (mean)", "Similarity Mahalanobis (mean)", "Sharma/Subband Metrics",
    "Training Time", "Optimizer Config", "Status",
]
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


_PARAM_COUNT_CACHE: dict[str, float] = {}


def param_count_millions(variant: str) -> Optional[float]:
    """Parameter count depends only on the variant (architecture), not on
    which run_id trained it -- computable right now, without GPU or any
    training having happened, by instantiating the same
    build_variant_model() used for training/smoke-testing. Cached per
    variant since 6 variants would otherwise be rebuilt on every row."""
    if variant in _PARAM_COUNT_CACHE:
        return _PARAM_COUNT_CACHE[variant]
    try:
        cfg = load_config()
        model = build_variant_model(cfg, n_classes=int(cfg.ptbxl.n_classes), variant=variant)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        value = round(n_params / 1e6, 2)
    except Exception:
        value = None
    _PARAM_COUNT_CACHE[variant] = value
    return value


def training_duration_str(meta: dict) -> Optional[str]:
    """Derived from metadata.json's own start_time/last_updated_utc (both
    already written by stage3_metadata.write_metadata() -- no new
    instrumentation needed). Only meaningful once status == "done"; for
    an in-progress or failed run this would understate/misrepresent
    actual training time, so it is intentionally left blank otherwise."""
    if meta.get("status") != "done":
        return None
    start = meta.get("start_time")
    end = meta.get("last_updated_utc")
    if not start or not end:
        return None
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except ValueError:
        return None
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def similarity_summary(run_dir: Path) -> dict:
    """Mean cosine/Mahalanobis/Bhattacharyya across classes, from
    similarity_metrics.csv -- already produced by run_stage3_queue.py's
    per-candidate eval loop (mentor_eval.similarity_metrics), just never
    read by this script before. Rows with a non-empty 'flag' (skipped,
    e.g. too few real samples) are excluded from the mean rather than
    treated as 0 -- averaging in a None/skipped row would silently bias
    the mean toward looking better or worse than the classes that
    actually got measured."""
    csv_path = run_dir / "mentor_eval" / "similarity_metrics.csv"
    empty = {"cosine": None, "mahalanobis": None, "bhattacharyya": None}
    if not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return empty
    if "flag" in df.columns:
        df = df[df["flag"].fillna("") == ""]
    if df.empty:
        return empty
    return {
        "cosine": round(df["cosine_similarity"].mean(), 4) if df["cosine_similarity"].notna().any() else None,
        "mahalanobis": round(df["mahalanobis"].mean(), 4) if df["mahalanobis"].notna().any() else None,
        "bhattacharyya": round(df["bhattacharyya"].mean(), 4) if df["bhattacharyya"].notna().any() else None,
    }


def subband_summary(run_dir: Path) -> Optional[dict]:
    """Sharma-inspired per-subband similarity (mentor_eval.subband_similarity_metrics),
    including the A3 subband ("P/T-wave, ST-segment, baseline" per
    subband_features.SUBBAND_CLINICAL_LABEL) -- the ST/T-wave-specific
    signal referenced in Stage 3 planning. Returns None (not a dict of
    Nones) when the file is absent, so callers can distinguish "not
    computed for this candidate" from "computed, all values null" --
    this file is NOT currently produced by run_stage3_queue.py's
    automated per-candidate eval loop (only classification_validation
    and similarity_metrics are), so it will be None for every candidate
    until someone runs mentor_eval.subband_similarity_metrics manually,
    or the queue is deliberately extended to include it."""
    csv_path = run_dir / "mentor_eval" / "subband_similarity_metrics.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    return {"n_rows": len(df), "path": str(csv_path)}


def load_candidate_row(run_id: str, variant: str) -> dict:
    run_dir = RESULTS_ROOT / run_id
    meta = _load_json(run_dir / "metadata.json")

    row = {
        "Candidate": run_id,
        "Variant": variant,
        "Params (M)": param_count_millions(variant),
        "Generated Accuracy": None,
        "Generated Macro-F1": None,
        "Generated Macro-AUC": None,
        "Real-data Accuracy": None,
        "Delta vs. Baseline (accuracy)": None,
        "Similarity Cosine (mean)": None,
        "Similarity Mahalanobis (mean)": None,
        "Sharma/Subband Metrics": "not computed (queue does not run subband_similarity_metrics)",
        "Training Time": None,
        "Optimizer Config": "unknown",
        "Status": "not yet evaluated -- no metadata.json found",
    }

    if meta is None:
        return row

    row["Status"] = meta.get("status", "unknown")
    row["Optimizer Config"] = classify_optimizer_config(meta.get("commit"))
    row["Training Time"] = training_duration_str(meta)

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

    sim = similarity_summary(run_dir)
    row["Similarity Cosine (mean)"] = sim["cosine"]
    row["Similarity Mahalanobis (mean)"] = sim["mahalanobis"]

    subband = subband_summary(run_dir)
    if subband is not None:
        row["Sharma/Subband Metrics"] = f"computed ({subband['n_rows']} rows, see {subband['path']})"

    return row


class MissingBaselineManifestError(RuntimeError):
    pass


def _resolve_baseline(require_manifest: bool = True) -> tuple[Optional[float], dict]:
    """Returns (baseline_accuracy, provenance_info).

    BUG FIX (2026-07-06): this used to silently fall back to a "COMPUTED
    LIVE" checksum (computed from whatever checkpoint currently sits at
    BASELINE_CKPT_PATH) when the frozen manifest was missing. That fallback
    produces a baseline number that looks exactly like a normal, verified
    one in the rendered table -- the checksum-manifest protocol exists
    specifically so a comparison can be trusted as "against the exact
    checkpoint that produced these numbers," and a live-computed
    substitute defeats that silently. Now raises MissingBaselineManifestError
    instead, unless the caller explicitly opts into the old best-effort
    behavior via require_manifest=False (kept only for callers that
    genuinely want a rough number and will label it themselves)."""
    baseline_metrics = _load_json(BASELINE_METRICS_PATH)
    baseline_accuracy = baseline_metrics["accuracy"] if baseline_metrics else None

    manifest = _load_json(BASELINE_MANIFEST_PATH)
    if manifest is not None:
        provenance = {
            "source": "frozen baseline_manifest.json",
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "commit": manifest.get("git_commit"),
        }
        return baseline_accuracy, provenance

    if require_manifest:
        raise MissingBaselineManifestError(
            f"No frozen baseline manifest at {BASELINE_MANIFEST_PATH}. "
            f"Refusing to silently substitute a live-computed checksum -- that "
            f"would produce a baseline number indistinguishable from a verified "
            f"one in the rendered table, defeating the checksum-manifest "
            f"protocol's entire purpose. Run mentor_eval.run_all against the "
            f"intended baseline checkpoint to produce {BASELINE_MANIFEST_PATH.name} "
            f"first, or call _resolve_baseline(require_manifest=False) explicitly "
            f"if a rough, unverified number is genuinely what's wanted."
        )

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
    checksum_str = provenance["checkpoint_sha256"] or "unavailable"
    commit_str = provenance["commit"] or "unavailable"
    lines.append(
        "All candidates evaluated via the identical `mentor_eval.classification_validation` "
        "pipeline against the frozen baseline manifest "
        f"(checksum `{checksum_str}`, commit `{commit_str}`, "
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

    lines.append("| " + " | ".join(TABLE_HEADER) + " |")
    lines.append("|" + "---|" * len(TABLE_HEADER))
    for row in rows:
        cells = [str(row[h]) if row[h] is not None else "--" for h in TABLE_HEADER]
        lines.append("| " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        baseline_accuracy, provenance = _resolve_baseline()
    except MissingBaselineManifestError as exc:
        print("=" * 70)
        print("ERROR: frozen baseline manifest missing -- Delta column NOT computed.")
        print(str(exc))
        print("Continuing to report per-candidate metrics (still valid on their")
        print("own), but this comparison is NOT verified against a checksummed")
        print("baseline until the manifest above is produced.")
        print("=" * 70)
        baseline_accuracy = None
        provenance = {
            "source": "ERROR -- baseline manifest missing, NOT computed live (see error banner above)",
            "checkpoint_sha256": None, "commit": None,
        }

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
