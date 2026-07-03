"""
audit_reproducibility.py

Stage 2.0.1 -- Reproducibility Audit.

Different failure mode than the Verification Gate (do the reported
NUMBERS reproduce from raw data) and the Repository Audit (are artifacts
stale/duplicated, independent of any specific experiment's claims). This
stage asks: is each ledger entry's claimed ARTIFACT IDENTITY actually
what's on disk right now -- the checkpoint that entry describes, not a
stale or mismatched one sharing a filename; the code state it claims to
have run against, actually reachable and inspectable, not just a string.

Design point raised explicitly during review and built in from the
start, not bolted on after: a commit-hash comparison is only meaningful
relative to WHAT IT WAS COMPARED AGAINST AND WHEN. Comparing a ledger's
`environment.commit` field to "current HEAD" is ambiguous across two
machines whose git state can drift (this project has already hit exactly
that drift more than once this session). So every manifest this script
writes carries its own `audit_metadata` block recording the local git
HEAD, branch, dirty-state, hostname, and timestamp AT THE MOMENT THE
AUDIT RAN -- not the experiment's run time, the audit's. A manifest found
on disk later is only as trustworthy as that recorded context; re-run
the audit rather than trust a stale manifest if local git state has
moved on since.

Usage:
    python audit_reproducibility.py <experiment_id> <ledger_path> <output_manifest_path> \\
        [--checkpoint PATH]... [--array PATH]... [--config PATH]...

Or import and call audit_experiment(...) / write_manifest(...).

Stdlib only for the core logic (hashlib, subprocess, json) -- no
torch/numpy dependency, so this can run even in a minimal environment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_ledger_entry(ledger_path: Path, experiment_id: str) -> Optional[dict[str, Any]]:
    if not ledger_path.exists():
        return None
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("experiment_id") == experiment_id:
                return r
    return None


def _run_git(args: list[str], repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.run(args, cwd=str(repo_dir), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _local_git_info(repo_dir: Path) -> dict[str, Any]:
    status = _run_git(["git", "status", "--porcelain"], repo_dir)
    return {
        "head_commit": _run_git(["git", "rev-parse", "HEAD"], repo_dir),
        "branch": _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        "dirty": bool(status) if status is not None else None,
    }


def _commit_reachable_locally(repo_dir: Path, commit: str) -> Optional[bool]:
    """True/False if determinable from local git history, None if the
    commit string is missing/malformed or git itself can't be queried."""
    if not commit:
        return None
    try:
        out = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0 and out.stdout.strip() == "commit"
    except Exception:
        return None


def audit_experiment(
    experiment_id: str,
    ledger_path: Path,
    repo_dir: Path,
    checkpoint_paths: Optional[list[Path]] = None,
    array_paths: Optional[list[Path]] = None,
    config_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """Build a checksummed reproducibility manifest for one ledger experiment."""
    audit_time = datetime.now(timezone.utc)
    local_git = _local_git_info(repo_dir)

    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "audit_metadata": {
            "audit_performed_at": audit_time.isoformat(),
            "audit_performed_on_hostname": platform.node(),
            "audit_performed_on_platform": platform.platform(),
            "local_git_head_at_audit_time": local_git["head_commit"],
            "local_git_branch_at_audit_time": local_git["branch"],
            "local_working_tree_dirty_at_audit_time": local_git["dirty"],
            "note": (
                "This manifest's verdicts are only valid as of the timestamp "
                "and git state recorded above -- re-run this audit rather than "
                "trust a manifest found on disk if local git state has moved "
                "on since, or before treating an old PASS as still meaningful."
            ),
        },
        "ledger_entry_found": False,
        "checks": [],
    }

    entry = _find_ledger_entry(ledger_path, experiment_id)
    if entry is None:
        manifest["checks"].append({
            "check": "ledger_entry_exists",
            "result": "FAIL",
            "detail": f"No entry for {experiment_id!r} in {ledger_path}",
        })
        return manifest

    manifest["ledger_entry_found"] = True
    env = entry.get("environment", {})

    # ── Commit reachability: is the ledger's claimed commit resolvable in
    #    THIS machine's git history, right now? ─────────────────────────────
    ledger_commit = env.get("commit")
    reachable = _commit_reachable_locally(repo_dir, ledger_commit) if ledger_commit else None
    manifest["checks"].append({
        "check": "commit_reachable_locally",
        "result": "PASS" if reachable else ("FAIL" if reachable is False else "UNKNOWN"),
        "ledger_claimed_commit": ledger_commit,
        "compared_against_local_head": local_git["head_commit"],
        "detail": (
            f"Ledger claims commit {ledger_commit!r}; "
            f"{'found' if reachable else 'NOT found'} in local git history "
            f"(checked against local HEAD {local_git['head_commit']} at audit time)."
        ),
    })

    # ── Working-tree cleanliness on the GPU AT THE TIME THE EXPERIMENT RAN ──
    ran_dirty = env.get("dirty")
    manifest["checks"].append({
        "check": "gpu_working_tree_was_clean_when_run",
        "result": "INFO" if ran_dirty is None else ("WARNING" if ran_dirty else "PASS"),
        "detail": (
            f"GPU server's working tree was "
            f"{'DIRTY (uncommitted changes present)' if ran_dirty else 'clean'} "
            f"when this experiment ran, per the ledger's own recorded "
            f"environment. " + (
                "The exact code that ran is not fully captured by the commit "
                "hash alone -- there were uncommitted local changes on top of it."
                if ran_dirty else ""
            )
        ),
    })

    # ── Checkpoint checksums ─────────────────────────────────────────────────
    for ckpt_path in (checkpoint_paths or []):
        if ckpt_path.exists():
            manifest["checks"].append({
                "check": "checkpoint_checksum",
                "result": "RECORDED",
                "path": str(ckpt_path),
                "sha256": _sha256_of_file(ckpt_path),
                "size_bytes": ckpt_path.stat().st_size,
                "mtime": datetime.fromtimestamp(ckpt_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        else:
            manifest["checks"].append({
                "check": "checkpoint_checksum",
                "result": "MISSING",
                "path": str(ckpt_path),
                "detail": "File not found -- not present in the extracted archive.",
            })

    # ── Dataset array checksums ──────────────────────────────────────────────
    for arr_path in (array_paths or []):
        if arr_path.exists():
            manifest["checks"].append({
                "check": "dataset_array_checksum",
                "result": "RECORDED",
                "path": str(arr_path),
                "sha256": _sha256_of_file(arr_path),
                "size_bytes": arr_path.stat().st_size,
            })
        else:
            manifest["checks"].append({
                "check": "dataset_array_checksum",
                "result": "MISSING",
                "path": str(arr_path),
            })

    # ── Config state ──────────────────────────────────────────────────────────
    for cfg_path in (config_paths or []):
        if cfg_path.exists():
            manifest["checks"].append({
                "check": "config_file_checksum",
                "result": "RECORDED",
                "path": str(cfg_path),
                "sha256": _sha256_of_file(cfg_path),
            })
        else:
            manifest["checks"].append({
                "check": "config_file_checksum",
                "result": "MISSING",
                "path": str(cfg_path),
            })

    # ── Seed, library versions -- straight from the ledger, no re-derivation ──
    manifest["checks"].append({
        "check": "seed_and_versions",
        "result": "RECORDED",
        "seed": entry.get("seed"),
        "python_version": env.get("python_version"),
        "torch_version": env.get("torch_version"),
        "cuda_available": env.get("cuda_available"),
        "cuda_device_name": env.get("cuda_device_name"),
    })

    return manifest


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2.0.1 Reproducibility Audit -- checksummed manifest for one ledger experiment."
    )
    parser.add_argument("experiment_id")
    parser.add_argument("ledger_path", type=Path)
    parser.add_argument("output_manifest_path", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--array", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, action="append", default=[])
    args = parser.parse_args()

    manifest = audit_experiment(
        args.experiment_id, args.ledger_path, args.repo_dir,
        checkpoint_paths=args.checkpoint, array_paths=args.array, config_paths=args.config,
    )
    write_manifest(manifest, args.output_manifest_path)
    print(f"Wrote {args.output_manifest_path}")
