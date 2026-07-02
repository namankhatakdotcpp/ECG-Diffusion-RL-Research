"""
audit_repository.py

Stage 2.0.5 -- Repository Audit.

Checks artifact PROVENANCE, not numerical correctness (that's the
Verification Gate's job). Answers: is the checkpoint actually the one the
reports claim it is? Are any figures/CSVs stale, generated from a
checkpoint that has since been overwritten? Are there duplicate
checkpoints hiding a "best" that's actually an earlier, under-trained
snapshot?

This is not a hypothetical concern for this project -- the EMA bug in the
original investigation was exactly a "wrong weights silently used"
provenance failure: use_ema=True defaulted to loading severely
under-trained shadow weights without anyone noticing until someone
compared std(unproj.weight) between the EMA shadow and the live weights.
A repository audit run BEFORE trusting any Stage 2 checkpoint-only
analysis is cheap insurance against a repeat of that exact failure mode.

Checks performed
-----------------
1. Duplicate checkpoints: any two checkpoint files with identical SHA-256
   content hash despite different filenames/epochs (a save step that
   didn't actually update weights, or diffusion_best.pt silently pointing
   at an earlier epoch than its filename/metadata implies).
2. Staleness: any generated sample (.npy) or result figure (.png) whose
   mtime PREDATES the mtime of the checkpoint it's supposedly derived
   from -- a strong signal it was produced by a previous checkpoint and
   never regenerated.
3. Metadata cross-check: diffusion_architecture.json's best_val_loss
   against the matching row in diffusion_training_log.csv -- these are
   two independently-written artifacts that should agree; a mismatch
   means one of them is stale or the write path has a bug.
4. Orphan detection: any checkpoint file with no corresponding entry in
   the training log epoch sequence (e.g. a checkpoint from an abandoned
   run mixed into the same directory as the current one).

Usage
-----
    python audit_repository.py <outputs_dir> <report_output_path>

    python audit_repository.py outputs/ \
        Roadmap/Stage_2_Architecture_Investigation/Reports/Repository_Audit_Report.md

Stdlib only -- hashlib, csv, json, pathlib. No torch/numpy dependency, so
it can run even before the Python environment for the diffusion model
itself is set up.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class AuditFinding:
    severity: str  # "error" | "warning" | "info"
    category: str
    message: str


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)

    def add(self, severity: str, category: str, message: str) -> None:
        self.findings.append(AuditFinding(severity, category, message))

    def errors(self) -> list:
        return [f for f in self.findings if f.severity == "error"]

    def warnings(self) -> list:
        return [f for f in self.findings if f.severity == "warning"]


def _check_duplicate_checkpoints(models_dir: Path, report: AuditReport) -> None:
    if not models_dir.exists():
        report.add("warning", "checkpoints", f"models directory not found: {models_dir}")
        return

    checkpoints = sorted(models_dir.glob("*.pt"))
    if not checkpoints:
        report.add("warning", "checkpoints", f"no .pt checkpoint files found in {models_dir}")
        return

    hashes: dict[str, list[Path]] = {}
    for ckpt in checkpoints:
        digest = _sha256_of_file(ckpt)
        hashes.setdefault(digest, []).append(ckpt)

    for digest, paths in hashes.items():
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            report.add(
                "error",
                "duplicate_checkpoint",
                f"{len(paths)} checkpoint files are byte-identical (sha256={digest[:12]}...): "
                f"{names} -- one of these did not actually update during training, or "
                f"'best' is silently pointing at an earlier/wrong epoch.",
            )

    report.add("info", "checkpoints", f"scanned {len(checkpoints)} checkpoint files, "
                                       f"{len(hashes)} distinct content hashes")


def _check_staleness(
    models_dir: Path,
    generated_dir: Path,
    results_dir: Path,
    report: AuditReport,
) -> None:
    best_ckpt = models_dir / "diffusion_best.pt"
    if not best_ckpt.exists():
        report.add("warning", "staleness", f"{best_ckpt} not found -- cannot check staleness")
        return
    best_mtime = best_ckpt.stat().st_mtime

    for pattern, label in (
        ("*.npy", "generated sample"),
        ("*.png", "result figure"),
    ):
        search_dir = generated_dir if label == "generated sample" else results_dir
        if not search_dir.exists():
            continue
        for artifact in search_dir.rglob(pattern):
            if artifact.stat().st_mtime < best_mtime:
                report.add(
                    "warning",
                    "stale_artifact",
                    f"{label} {artifact} is OLDER than the current diffusion_best.pt "
                    f"(artifact mtime predates checkpoint mtime by "
                    f"{best_mtime - artifact.stat().st_mtime:.0f}s) -- this artifact was "
                    f"likely generated from a previous checkpoint and never regenerated. "
                    f"Do not cite it as representing the current model.",
                )


def _check_metadata_consistency(
    models_dir: Path, logs_dir: Path, report: AuditReport
) -> None:
    arch_path = models_dir / "diffusion_architecture.json"
    log_path = logs_dir / "diffusion_training_log.csv"

    if not arch_path.exists() or not log_path.exists():
        report.add(
            "warning",
            "metadata_consistency",
            f"cannot cross-check -- missing {'architecture.json' if not arch_path.exists() else 'training_log.csv'}",
        )
        return

    with open(arch_path) as f:
        arch = json.load(f)
    arch_best_val = arch.get("best_val_loss")

    log_best_val: Optional[float] = None
    with open(log_path) as f:
        for row in csv.DictReader(f):
            val = row.get("val_loss", "")
            if val:
                v = float(val)
                if log_best_val is None or v < log_best_val:
                    log_best_val = v

    if arch_best_val is None or log_best_val is None:
        report.add(
            "warning",
            "metadata_consistency",
            "best_val_loss missing from architecture.json or training_log.csv -- cannot cross-check",
        )
        return

    if abs(arch_best_val - log_best_val) > 1e-4:
        report.add(
            "error",
            "metadata_consistency",
            f"diffusion_architecture.json reports best_val_loss={arch_best_val:.6f} but "
            f"diffusion_training_log.csv's minimum val_loss is {log_best_val:.6f} -- these "
            f"two independently-written files disagree. One of them was written by a stale "
            f"run, or architecture.json was not regenerated after the actual best checkpoint.",
        )
    else:
        report.add("info", "metadata_consistency", "architecture.json and training_log.csv agree on best_val_loss")


def _check_orphan_checkpoints(models_dir: Path, logs_dir: Path, report: AuditReport) -> None:
    log_path = logs_dir / "diffusion_training_log.csv"
    if not log_path.exists() or not models_dir.exists():
        return

    logged_epochs: set[int] = set()
    with open(log_path) as f:
        for row in csv.DictReader(f):
            ep = row.get("epoch", "")
            if ep:
                logged_epochs.add(int(ep))

    for ckpt in models_dir.glob("diffusion_ckpt_ep*.pt"):
        try:
            epoch_str = ckpt.stem.split("_ep")[-1]
            epoch = int(epoch_str)
        except ValueError:
            report.add("warning", "orphan_checkpoint", f"could not parse epoch from filename: {ckpt.name}")
            continue
        if epoch not in logged_epochs:
            report.add(
                "warning",
                "orphan_checkpoint",
                f"{ckpt.name} has no matching epoch={epoch} entry in diffusion_training_log.csv "
                f"-- possibly a checkpoint from an abandoned or overwritten run.",
            )


def run_audit(outputs_dir: Path) -> AuditReport:
    outputs_dir = Path(outputs_dir)
    models_dir = outputs_dir / "models"
    generated_dir = outputs_dir / "generated"
    results_dir = outputs_dir / "results"
    logs_dir = outputs_dir / "logs"

    report = AuditReport()
    _check_duplicate_checkpoints(models_dir, report)
    _check_staleness(models_dir, generated_dir, results_dir, report)
    _check_metadata_consistency(models_dir, logs_dir, report)
    _check_orphan_checkpoints(models_dir, logs_dir, report)
    return report


def write_report(report: AuditReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Repository Audit Report\n"]
    lines.append(
        "_Checks artifact provenance (right checkpoint, not stale, not duplicated) -- "
        "NOT numerical correctness of reported results, which is the Verification Gate's job._\n"
    )

    errors = report.errors()
    warnings = report.warnings()

    lines.append(f"## Summary: {len(errors)} error(s), {len(warnings)} warning(s)\n")

    if errors:
        lines.append("## Errors (block Stage 2 until resolved)\n")
        for e in errors:
            lines.append(f"- ❌ **[{e.category}]** {e.message}")
        lines.append("")

    if warnings:
        lines.append("## Warnings (review before citing affected artifacts)\n")
        for w in warnings:
            lines.append(f"- ⚠️ **[{w.category}]** {w.message}")
        lines.append("")

    info = [f for f in report.findings if f.severity == "info"]
    if info:
        lines.append("## Info\n")
        for i in info:
            lines.append(f"- ℹ️ {i.message}")
        lines.append("")

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audit_repository.py <outputs_dir> <report_output_path>")
        sys.exit(1)

    outputs_dir = Path(sys.argv[1])
    report_path = Path(sys.argv[2])

    audit = run_audit(outputs_dir)
    write_report(audit, report_path)

    n_errors = len(audit.errors())
    print(f"Audit complete: {n_errors} error(s), {len(audit.warnings())} warning(s)")
    print(f"Report written to {report_path}")

    if n_errors > 0:
        sys.exit(1)  # non-zero exit so a CI/master-runner script can block on this
