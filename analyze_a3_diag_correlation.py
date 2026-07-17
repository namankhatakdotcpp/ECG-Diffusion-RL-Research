"""
analyze_a3_diag_correlation.py -- per-iteration relationship between
reward_a3 (r_a3) and diagnostic reward (r_diag) for the HYP and OTHER
classes, to test the reward-reallocation/"reward hacking" hypothesis
(Roadmap/Stage_4_Optimization/Decisions.md, Finding 7): does the RL policy
trade diagnostic identity for A3/morphology score specifically for these
two classes, which show the severe r_diag collapse documented in that file?

WHY RAW CORRELATION IS NOT ENOUGH
------------------------------------
Both r_a3 (rising over the run) and r_diag for HYP/OTHER (falling over the
run) trend monotonically over the same ~1000 iterations. A raw Pearson/
Spearman correlation on the untransformed series will show strong apparent
anticorrelation almost by construction, driven by two series sharing a
training-time trend, not necessarily by iteration-to-iteration coupling.
This script reports the raw correlation for context but treats it as
NON-decision-bearing, and computes a first-differenced ("detrended")
correlation as the actual test of the coupling hypothesis, plus a rolling
correlation to check whether any (anti)coupling is stable across the run
or concentrated in one phase.

DECISION RULE (mechanical, not eyeballed)
--------------------------------------------
The reward-reallocation hypothesis is upgraded from HYPOTHESIS to
CONFIRMED for a given class ONLY IF the first-differenced (detrended)
Pearson correlation is more negative than -0.3 AND p < 0.05 for that
class. Otherwise it stays HYPOTHESIS, with the raw and detrended numbers
both reported explicitly either way -- this script never reports a class
as CONFIRMED based on the raw correlation alone.

"ITERATION" MEANS "OCCURRENCE OF THAT CLASS", NOT GLOBAL ROW INDEX
-----------------------------------------------------------------------
`class` is sampled per-iteration in step07_rl_finetuning.py's training
loop (rotates across all classes), so r_diag/r_a3 for HYP only exist on
the subset of rows where class == "HYP". "Iteration-to-iteration" here
means successive occurrences of that class in `iter` order (same
convention already used by diagnose_smoke_test_reward_trend.py's
per-class trend section and Decisions.md's HYP/OTHER first-half/
second-half analysis) -- not a diff against the immediately-preceding CSV
row, which would usually belong to a different class entirely.

Usage:
    python analyze_a3_diag_correlation.py [--run-tag stage4_finetune_v1]
        [--csv PATH_OVERRIDE] [--out-dir outputs/results/reward_correlation_analysis]
        [--rolling-window 100]

    --run-tag resolves to logs/<run-tag>/rl_training_log.csv (matching
    step07_rl_finetuning.py's --run-tag convention, logs_dir/run_tag,
    step07_rl_finetuning.py:1130-1133) -- NOT a flat logs/rl_training_log.csv,
    which is only where an UNTAGGED run would write. --csv overrides this
    resolution entirely if passed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

TARGET_CLASSES = ["HYP", "OTHER"]
CONFIRM_R_THRESHOLD = -0.3
CONFIRM_P_THRESHOLD = 0.05


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (pearson_r, pearson_p, spearman_r, spearman_p). None values if
    scipy is unavailable (falls back to np.corrcoef for r only, no p-value)
    or if there are fewer than 3 paired points (correlation undefined)."""
    n = len(x)
    if n < 3:
        return None, None, None, None
    if _HAVE_SCIPY:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return float(pr), float(pp), float(sr), float(sp)
    # Fallback: Pearson via np.corrcoef, Spearman via Pearson-on-ranks. No
    # p-values without scipy -- reported as None, not fabricated.
    pr = float(np.corrcoef(x, y)[0, 1])
    xr, yr = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    sr = float(np.corrcoef(xr, yr)[0, 1])
    return pr, None, sr, None


def _rolling_corr(iters: np.ndarray, x: np.ndarray, y: np.ndarray, window: int) -> list[tuple[float, Optional[float]]]:
    """Rolling Pearson r of x vs y over a sliding window of `window`
    consecutive occurrences (indexed by position in the already-filtered,
    already-sorted per-class arrays -- see module docstring on what
    'iteration' means here). Returns (center_iter, r) pairs."""
    n = len(x)
    out = []
    if n < window:
        return out
    for start in range(0, n - window + 1):
        end = start + window
        xw, yw = x[start:end], y[start:end]
        if np.std(xw) == 0 or np.std(yw) == 0:
            r = None
        else:
            r = float(np.corrcoef(xw, yw)[0, 1])
        center_iter = float(iters[start:end].mean())
        out.append((center_iter, r))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="stage4_finetune_v1",
                         help="Matches step07_rl_finetuning.py's --run-tag convention: logs "
                              "for a tagged run are written to logs/<run-tag>/rl_training_log.csv "
                              "(logs_dir = logs_dir / run_tag when run_tag is set, "
                              "step07_rl_finetuning.py:1130-1133), not a flat logs/ path. "
                              "Ignored if --csv is passed explicitly.")
    parser.add_argument("--csv", type=str, default=None,
                         help="Explicit path override. If omitted, resolved from --run-tag as "
                              "logs/<run-tag>/rl_training_log.csv, matching the same tagged-run "
                              "convention analyze_stage4_hyp_other.py uses for its own outputs.")
    parser.add_argument("--out-dir", type=str, default="outputs/results/reward_correlation_analysis")
    parser.add_argument("--rolling-window", type=int, default=100)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else Path("logs") / args.run_tag / "rl_training_log.csv"
    if not csv_path.exists():
        print(f"[BLOCKED] {csv_path} not found. This script re-analyses an "
              "existing rl_training_log.csv from a real Stage 4 run -- it "
              "does not generate new data. Run on the GPU server where that "
              "run's log actually lives, e.g.:\n\n"
              f"    python analyze_a3_diag_correlation.py --run-tag {args.run_tag}\n\n"
              f"  (resolves to logs/{args.run_tag}/rl_training_log.csv; pass --csv to "
              "override directly if the run used a different --run-tag or no tag at all.)\n")
        return

    if not _HAVE_SCIPY:
        print("[WARNING] scipy not importable in this environment -- p-values "
              "will be reported as None and the CONFIRMED/HYPOTHESIS decision "
              "rule (which requires p < 0.05) cannot be mechanically applied. "
              "scipy is already a project dependency (requirements.txt) -- "
              "this warning should only fire in a stripped-down local check, "
              "not on the GPU server.\n")

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "iter": int(row["iter"]),
                    "class": row["class"],
                    "r_diag": float(row["r_diag"]),
                    "r_a3": float(row["r_a3"]),
                })
            except (KeyError, ValueError) as e:
                print(f"[WARNING] skipping malformed row {row!r}: {e}")
    rows.sort(key=lambda r: r["iter"])
    n_total = len(rows)

    print("=" * 70)
    print(f"A3 / DIAGNOSTIC-REWARD CORRELATION ANALYSIS  ({csv_path}, "
          f"n={n_total} total rows)")
    print("=" * 70)

    out_dir = Path(args.out_dir)
    plots_made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _HAVE_MPL = True
    except ImportError:
        _HAVE_MPL = False
        print("[WARNING] matplotlib not importable -- skipping plot generation, "
              "numeric analysis below is unaffected.")

    verdicts: dict[str, str] = {}

    for target in TARGET_CLASSES:
        class_rows = [r for r in rows if r["class"] == target]
        n = len(class_rows)
        print(f"\n{'-' * 70}")
        print(f"CLASS: {target}  (n_occurrences={n})")
        print("-" * 70)

        if n < 5:
            print(f"  [INCONCLUSIVE] Only {n} occurrences of {target} in this log -- "
                  "too few for any correlation to be meaningful. Skipping.")
            verdicts[target] = "INCONCLUSIVE (insufficient occurrences)"
            continue

        iters = np.array([r["iter"] for r in class_rows], dtype=float)
        r_diag = np.array([r["r_diag"] for r in class_rows], dtype=float)
        r_a3 = np.array([r["r_a3"] for r in class_rows], dtype=float)

        # 1. RAW correlation -- reported, NOT decision-bearing.
        raw_pr, raw_pp, raw_sr, raw_sp = _corr(r_diag, r_a3)
        print("\n  [1. RAW correlation -- trend-confounded, NOT decision-bearing]")
        print(f"    Pearson  r={raw_pr}  p={raw_pp}")
        print(f"    Spearman r={raw_sr}  p={raw_sp}")
        print("    Both r_a3 and r_diag trend monotonically over the run -- a strong raw "
              "anticorrelation is expected even with no direct iteration-to-iteration "
              "coupling. Do not cite this number alone as evidence of reward reallocation.")

        # 2. FIRST-DIFFERENCED (detrended) correlation -- decision-bearing.
        d_diag = np.diff(r_diag)
        d_a3 = np.diff(r_a3)
        det_pr, det_pp, det_sr, det_sp = _corr(d_diag, d_a3)
        print("\n  [2. FIRST-DIFFERENCED (detrended) correlation -- DECISION-BEARING]")
        print(f"    Pearson  r={det_pr}  p={det_pp}")
        print(f"    Spearman r={det_sr}  p={det_sp}")

        # 3. Rolling-window correlation.
        window = args.rolling_window
        if n < window:
            print(f"\n  [3. Rolling correlation] n={n} < window={window} -- reducing "
                  f"window to {max(5, n // 3)} for this class.")
            window = max(5, n // 3)
        rolling = _rolling_corr(iters, r_diag, r_a3, window)
        if rolling:
            rs = [r for _, r in rolling if r is not None]
            if rs:
                print(f"  [3. Rolling correlation, window={window} occurrences] "
                      f"min={min(rs):.3f}  max={max(rs):.3f}  "
                      f"{'stable sign' if (min(rs) < 0) == (max(rs) < 0) else 'SIGN FLIPS across the run -- not stable'}")
            else:
                print(f"  [3. Rolling correlation, window={window}] all windows had zero variance -- no usable value.")
        else:
            print(f"  [3. Rolling correlation] not enough occurrences for window={window}.")

        # 4. Descriptive first-half vs second-half (context only, not decision-bearing).
        half = max(1, n // 2)
        fh_diag, sh_diag = r_diag[:half], r_diag[half:]
        fh_a3, sh_a3 = r_a3[:half], r_a3[half:]
        print("\n  [4. First-half vs second-half means -- descriptive context only, "
              "same trend-confound caveat as (1)]")
        print(f"    r_diag: first_half_mean={fh_diag.mean():.4f}  second_half_mean={sh_diag.mean():.4f}")
        print(f"    r_a3:   first_half_mean={fh_a3.mean():.4f}  second_half_mean={sh_a3.mean():.4f}")

        # ── Decision rule ────────────────────────────────────────────────────
        print("\n  [DECISION]")
        if det_pr is None or det_pp is None:
            verdict = "INCONCLUSIVE (no p-value available -- scipy missing)"
        elif det_pr < CONFIRM_R_THRESHOLD and det_pp < CONFIRM_P_THRESHOLD:
            verdict = f"CONFIRMED (detrended Pearson r={det_pr:.4f}, p={det_pp:.4g})"
            print(f"    {target}: detrended r={det_pr:.4f} < {CONFIRM_R_THRESHOLD} and "
                  f"p={det_pp:.4g} < {CONFIRM_P_THRESHOLD} -- reward-reallocation "
                  f"hypothesis UPGRADED to CONFIRMED for {target}.")
        else:
            reason = []
            if det_pr is not None and det_pr >= CONFIRM_R_THRESHOLD:
                reason.append(f"r={det_pr:.4f} not below {CONFIRM_R_THRESHOLD}")
            if det_pp is not None and det_pp >= CONFIRM_P_THRESHOLD:
                reason.append(f"p={det_pp:.4g} not below {CONFIRM_P_THRESHOLD}")
            verdict = f"HYPOTHESIS (unchanged) -- {', '.join(reason)}"
            print(f"    {target}: {verdict}. Raw correlation confounded by shared trend; "
                  "detrended analysis does not clear the confirmation threshold.")
        verdicts[target] = verdict

        # ── Plots ────────────────────────────────────────────────────────────
        if _HAVE_MPL:
            out_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(iters, r_diag, marker=".", lw=1)
            ax.set_xlabel("iteration"); ax.set_ylabel("r_diag")
            ax.set_title(f"{target}: r_diag vs iteration")
            p = out_dir / f"{target.lower()}_r_diag_vs_iter.png"
            fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
            plots_made.append(p)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(iters, r_a3, marker=".", lw=1, color="tab:orange")
            ax.set_xlabel("iteration"); ax.set_ylabel("r_a3")
            ax.set_title(f"{target}: r_a3 vs iteration")
            p = out_dir / f"{target.lower()}_r_a3_vs_iter.png"
            fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
            plots_made.append(p)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(d_a3, d_diag, s=12, alpha=0.6)
            ax.set_xlabel("delta r_a3 (occurrence-to-occurrence)")
            ax.set_ylabel("delta r_diag (occurrence-to-occurrence)")
            title_suffix = f" (r={det_pr:.3f})" if det_pr is not None else ""
            ax.set_title(f"{target}: detrended relationship{title_suffix}")
            p = out_dir / f"{target.lower()}_delta_scatter.png"
            fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
            plots_made.append(p)

            if rolling:
                rc_iters = [it for it, r in rolling if r is not None]
                rc_vals = [r for it, r in rolling if r is not None]
                if rc_vals:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(rc_iters, rc_vals, lw=1.5)
                    ax.axhline(0, color="gray", lw=0.8, ls="--")
                    ax.set_xlabel("iteration (window center)"); ax.set_ylabel(f"rolling Pearson r (window={window})")
                    ax.set_title(f"{target}: rolling r_diag/r_a3 correlation")
                    p = out_dir / f"{target.lower()}_rolling_corr.png"
                    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
                    plots_made.append(p)

    # Normalized overlay (context/visual only, both classes together) --
    # explicitly NOT evidence of coupling, per instruction.
    if _HAVE_MPL and n_total > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for target in TARGET_CLASSES:
            class_rows = [r for r in rows if r["class"] == target]
            if len(class_rows) < 2:
                continue
            iters = np.array([r["iter"] for r in class_rows], dtype=float)
            r_diag = np.array([r["r_diag"] for r in class_rows], dtype=float)
            r_a3 = np.array([r["r_a3"] for r in class_rows], dtype=float)
            def _norm(v):
                lo, hi = v.min(), v.max()
                return (v - lo) / (hi - lo) if hi > lo else v * 0
            ax.plot(iters, _norm(r_diag), label=f"{target} r_diag (norm)", lw=1)
            ax.plot(iters, _norm(r_a3), label=f"{target} r_a3 (norm)", lw=1, ls="--")
        ax.set_xlabel("iteration"); ax.set_ylabel("normalized [0,1]")
        ax.set_title("Trend visualization ONLY -- not evidence of coupling")
        ax.legend(fontsize=8)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "normalized_trend_overlay.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        plots_made.append(p)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for target, v in verdicts.items():
        print(f"  {target}: {v}")
    if plots_made:
        print(f"\n  {len(plots_made)} plot(s) written to {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
