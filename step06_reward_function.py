"""
step06_reward_function.py — Clinical reward function for RL fine-tuning.

This is the intellectual core of the paper: a physiology-grounded reward that
penalises generated ECGs lacking clinical validity, preventing reward hacking.

Five components (all return float in [0, 1]):
  MorphologyReward        — PQRST interval matching vs reference stats (neurokit2)
  HRVReward               — SDNN/RMSSD plausibility vs NORM reference
  RealismReward           — PCA-based proximity to real training manifold
  DiagnosticUtilityReward — CNN classifier confidence for target disease class
  A3Reward                — A3-subband (slow-wave) energy match vs reference
                             distribution; directly motivated by Stage 3's
                             dominant finding (largest remaining real-vs-
                             generated divergence, every architecture tested)

Composite: ClinicalReward = weighted sum, weights from cfg.reward.weights
(config.yaml) when config_name="full" — NOT from ABLATION_CONFIGS in that
case (a prior bug had get_reward() silently ignore cfg.reward.weights even
for "full"; fixed, see Decisions.md). Named ablation variants below still
use the fixed ABLATION_CONFIGS table regardless of cfg, by design.

Ablation variants (for step09): 'full', 'diag_only', 'no_diag', 'no_morph',
'no_hrv', 'no_a3', 'a3_only'

Reads at startup:
  outputs/processed/morphology_stats.json
  outputs/processed/hrv_stats.json
  outputs/processed/a3_subband_stats.json
  outputs/processed/class_names.json
  outputs/processed/X_train.npy          (for PCA fitting)
  outputs/models/trtr_classifier.pt      (DiagnosticUtilityReward, optional)

Self-test writes:
  outputs/results/fig04_reward_components.pdf
  outputs/processed/reward_function_validated.pkl
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

warnings.filterwarnings("ignore")

LEAD_II = 1   # index of Lead II in the 12-lead array

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# ──────────────────────────────────────────────────────────────────────────────
# Ablation weight configurations (step09)
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict[str, float]] = {
    "full":      {"morph": 0.3, "hrv": 0.3, "real": 0.2, "diag": 0.2, "a3": 0.0},
    "diag_only": {"morph": 0.0, "hrv": 0.0, "real": 0.0, "diag": 1.0, "a3": 0.0},
    "no_diag":   {"morph": 0.4, "hrv": 0.4, "real": 0.2, "diag": 0.0, "a3": 0.0},
    "no_morph":  {"morph": 0.0, "hrv": 0.4, "real": 0.3, "diag": 0.3, "a3": 0.0},
    "no_hrv":    {"morph": 0.4, "hrv": 0.0, "real": 0.3, "diag": 0.3, "a3": 0.0},
    "no_a3":     {"morph": 0.3, "hrv": 0.3, "real": 0.2, "diag": 0.2, "a3": 0.0},
    "a3_only":   {"morph": 0.0, "hrv": 0.0, "real": 0.0, "diag": 0.0, "a3": 1.0},
}
# NOTE: "full"/"no_a3" here keep a3=0.0 for backward compatibility with the
# step09 ablation harness's existing 4-term configs. The actual training
# weight for a3 (nonzero, per the Stage 4 decision that A3 must be in the
# reward from the first RL experiment) is set in config.yaml's
# reward.weights and used by get_reward() when config_name="full" is
# called WITHOUT overriding weights from cfg — see get_reward() below.

# ──────────────────────────────────────────────────────────────────────────────
# Shared scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_score(value: float, ref_mean: float, ref_std: float) -> float:
    """
    Gaussian kernel score centred at ref_mean.

      score = exp( -((value - ref_mean) / ref_std)² )

    Returns 1.0 at the reference mean and decays toward 0 as deviation grows.
    At ±1σ → e⁻¹ ≈ 0.37; at ±2σ → e⁻⁴ ≈ 0.018.
    """
    if ref_std < 1e-8:
        return 1.0 if abs(value - ref_mean) < 1e-6 else 0.0
    z = (value - ref_mean) / ref_std
    return float(np.clip(np.exp(-(z ** 2)), 0.0, 1.0))


def _extract_pqrst(ecg_lead2: np.ndarray, fs: float) -> Optional[dict[str, float]]:
    """
    Run neurokit2 on a single Lead-II trace and return median beat intervals (ms).

    Returns dict with any subset of {'pr_ms', 'qrs_ms', 'qt_ms'}, or None on failure.
    Physiological plausibility windows filter delineation artefacts.
    """
    try:
        import neurokit2 as nk
        import pandas as pd

        signals, info = nk.ecg_process(ecg_lead2.astype(np.float64), sampling_rate=fs)
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if len(r_peaks) < 3:
            return None

        ms    = 1000.0 / fs
        q_win = int(0.06 * fs)
        t_win = int(0.25 * fs)

        def _locs(col: str) -> np.ndarray:
            s = signals.get(col)
            if s is None:
                return np.empty(0, dtype=int)
            return np.where(s.fillna(0).astype(int) == 1)[0]

        p_locs = _locs("ECG_P_Peaks")
        q_locs = _locs("ECG_Q_Peaks")
        s_locs = _locs("ECG_S_Peaks")
        t_locs = _locs("ECG_T_Peaks")

        pr_list: list[float] = []
        qrs_list: list[float] = []
        qt_list: list[float] = []

        for r in r_peaks:
            pre_p  = p_locs[p_locs < r]
            q_near = q_locs[(q_locs < r) & (q_locs > r - q_win)]
            s_near = s_locs[(s_locs > r) & (s_locs < r + q_win)]
            t_near = t_locs[(t_locs > r) & (t_locs < r + t_win)]

            if len(pre_p) and len(q_near):
                pr = (r - pre_p[-1]) * ms
                if 60 < pr < 400:
                    pr_list.append(pr)
            if len(q_near) and len(s_near):
                qrs = (s_near[0] - q_near[-1]) * ms
                if 40 < qrs < 200:
                    qrs_list.append(qrs)
            if len(q_near) and len(t_near):
                qt = (t_near[0] - q_near[-1]) * ms
                if 200 < qt < 650:
                    qt_list.append(qt)

        result: dict[str, float] = {}
        if pr_list:  result["pr_ms"]  = float(np.median(pr_list))
        if qrs_list: result["qrs_ms"] = float(np.median(qrs_list))
        if qt_list:  result["qt_ms"]  = float(np.median(qt_list))
        return result if result else None
    except Exception:
        return None


def _extract_hrv(ecg_lead2: np.ndarray, fs: float) -> Optional[dict[str, float]]:
    """
    Compute SDNN and RMSSD from R-peak intervals on a Lead-II trace.
    Returns None if fewer than 4 R-peaks detected or HRV computation fails.
    """
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(ecg_lead2.astype(np.float64), sampling_rate=fs)
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if len(r_peaks) < 4:
            return None

        rr_ms = np.diff(r_peaks) * (1000.0 / fs)
        valid = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
        if len(valid) < 3:
            return None

        sdnn  = float(np.std(valid, ddof=1))
        rmssd = float(np.sqrt(np.mean(np.diff(valid) ** 2)))
        return {"sdnn_ms": sdnn, "rmssd_ms": rmssd}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Reward components
# ──────────────────────────────────────────────────────────────────────────────

class MorphologyReward:
    """
    Scores how well the generated ECG's PQRST morphology matches the reference
    for its target disease class.

    Clinical justification: an MI ECG must show MI-typical features (broad Q waves,
    ST elevation) to be clinically valid — not merely look like plausible noise.
    Without this component, reward hacking produces "noise with the right frequency
    content" rather than disease-typical waveforms (demonstrated in step09 ablation).

    Gaussian scoring: score_i = exp(-((value − μ_ref) / σ_ref)²)
    Returns mean score over detected {PR, QRS, QT} intervals.
    """

    def __init__(self, morph_stats: dict, fs: float = 100.0):
        self.stats = morph_stats
        self.fs    = fs

    def compute(self, ecg: np.ndarray, target_class: str) -> float:
        """
        Args:
            ecg:          (1000, 12) z-score normalised ECG
            target_class: class name string (e.g. 'MI', 'NORM')
        Returns:
            float in [0, 1]
        """
        ref    = self.stats.get(target_class, {})
        result = _extract_pqrst(ecg[:, LEAD_II], self.fs)

        if result is None:
            return 0.0
        if not ref:
            return 0.5   # no reference — neutral

        scores: list[float] = []
        for key in ("pr_ms", "qrs_ms", "qt_ms"):
            if key in ref and key in result:
                scores.append(_gaussian_score(
                    result[key], ref[key]["mean"], ref[key]["std"]
                ))

        return float(np.mean(scores)) if scores else 0.0


class HRVReward:
    """
    Scores whether heart rate variability is physiologically plausible.

    An ECG with perfectly regular RR intervals (SDNN = 0) is not realistic —
    even pathological rhythms show some variability. This component penalises
    both artificially regular and impossibly erratic generated signals.
    Uses NORM as the HRV reference since basic autonomic variability is present
    in all healthy and most diseased rhythms.
    """

    def __init__(self, hrv_stats: dict, fs: float = 100.0):
        self.stats = hrv_stats
        self.fs    = fs
        self.ref   = hrv_stats.get("NORM", {})

    def compute(self, ecg: np.ndarray) -> float:
        """
        Args:
            ecg: (1000, 12)
        Returns:
            float in [0, 1]
        """
        if not self.ref:
            return 0.5   # no reference — neutral

        result = _extract_hrv(ecg[:, LEAD_II], self.fs)
        if result is None:
            return 0.0

        scores: list[float] = []
        for key in ("sdnn_ms", "rmssd_ms"):
            if key in self.ref and key in result:
                scores.append(_gaussian_score(
                    result[key], self.ref[key]["mean"], self.ref[key]["std"]
                ))

        return float(np.mean(scores)) if scores else 0.0


class RealismReward:
    """
    Penalises ECGs that fall outside the statistical manifold of real ECGs.

    Fits a PCA on Lead-II traces from the training set. Evaluates each generated
    ECG by computing its normalised Mahalanobis-like distance from the training
    distribution centre in PC space:

      distance = √( Σᵢ (pc_score_i / σ_i)² )
      score    = exp( −distance / 5.0 )

    where σ_i is the std of the i-th principal component score on the training set.
    """

    def __init__(
        self,
        X_train:      np.ndarray,    # (N, 1000, 12)
        n_components: int   = 50,
        n_samples:    int   = 2000,
        fs:           float = 100.0,
    ):
        self.fs    = fs
        self._scale = 5.0

        rng = np.random.default_rng(42)
        idx  = rng.choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
        lead2 = X_train[idx, :, LEAD_II]   # (n, 1000)

        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(lead2)

        # Std of PC scores on training set — used as per-component scale
        scores       = self.pca.transform(lead2)
        self.pca_std = scores.std(axis=0) + 1e-8   # (n_components,)

    def compute(self, ecg: np.ndarray) -> float:
        """
        Args:
            ecg: (1000, 12)
        Returns:
            float in [0, 1]; 1.0 for ECGs at the training distribution centre
        """
        lead2    = ecg[:, LEAD_II].reshape(1, -1)          # (1, 1000)
        scores   = self.pca.transform(lead2)[0]             # (n_components,)
        norm_sc  = scores / self.pca_std                    # (n_components,)
        distance = float(np.sqrt(np.sum(norm_sc ** 2)))
        return float(np.exp(-distance / self._scale))


class DiagnosticUtilityReward:
    """
    Rewards ECGs that would be useful for training a downstream disease classifier.
    Returns the softmax probability assigned to the target class by a CNN trained
    on real ECGs (loaded from outputs/models/trtr_classifier.pt), scaled by that
    class's held-out reliability (per-class F1, from trtr_classifier_eval.json).

    Reliability scaling: reward = confidence(target_class) * reliability[target_class]
    Classes the TRTR classifier is unreliable on (e.g. HYP, macro-F1 ~0.39 vs
    NORM's ~0.82) get discounted rather than trusted at face value — a policy
    shouldn't be able to farm free reward from a class the classifier itself is
    weak at distinguishing. Falls back to reliability=1.0 (no scaling) if
    trtr_classifier_eval.json isn't present, since the weight-vs-uniform
    question is unvalidated until real per-class F1 numbers exist.

    WARNING: This component alone causes reward hacking if used without the others.
    The diffusion model quickly learns to produce ECGs that fool the CNN while
    ignoring physiological constraints (demonstrated in the ablation study, step09).
    The morphology and HRV components act as physiological anchors.

    Falls back to 0.5 (neutral) if the classifier file is not found — run step05
    first or call build_diagnostic_classifier() to create it.
    """

    def __init__(
        self,
        classifier_path: str,
        n_classes:       int,
        device:          str = "cpu",
        eval_path:       Optional[str] = None,
        use_reliability: bool = True,
    ):
        from step05_baseline_eval import Simple1DCNN

        self.device      = device
        self.n_classes   = n_classes
        self.available   = False
        self._log        = logging.getLogger(__name__)
        self.model: Optional[nn.Module] = None
        self.use_reliability = use_reliability
        self.reliability  = np.ones(n_classes, dtype=float)

        path = Path(classifier_path)
        if path.exists():
            try:
                ckpt  = torch.load(str(path), map_location=device)
                nc    = ckpt.get("n_classes", n_classes)
                m     = Simple1DCNN(n_classes=nc).to(device)
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                self.model     = m
                self.available = True
                self._log.info(f"DiagnosticUtilityReward: loaded classifier from {path}")
            except Exception as exc:
                self._log.warning(
                    f"DiagnosticUtilityReward: failed to load {path} ({exc}). "
                    "Returning 0.5 (neutral) until a valid classifier is provided."
                )
        else:
            self._log.warning(
                f"DiagnosticUtilityReward: {path} not found. "
                "Run step05 first. Returning 0.5 (neutral)."
            )

        if not use_reliability:
            self._log.info(
                "DiagnosticUtilityReward: use_reliability_scaling=False — "
                "reliability fixed at 1.0 (ablation mode)."
            )
            return

        eval_p = Path(eval_path) if eval_path else path.parent / "trtr_classifier_eval.json"
        if eval_p.exists():
            try:
                per_class_f1 = json.load(open(eval_p)).get("per_class_f1")
                if per_class_f1 and len(per_class_f1) == n_classes:
                    self.reliability = np.asarray(per_class_f1, dtype=float)
                    self._log.info(
                        f"DiagnosticUtilityReward: loaded per-class reliability from {eval_p}"
                    )
                else:
                    self._log.warning(
                        f"DiagnosticUtilityReward: {eval_p} missing/mismatched per_class_f1 "
                        f"(expected {n_classes} entries). Using uniform reliability=1.0."
                    )
            except Exception as exc:
                self._log.warning(
                    f"DiagnosticUtilityReward: failed to load {eval_p} ({exc}). "
                    "Using uniform reliability=1.0."
                )
        else:
            self._log.warning(
                f"DiagnosticUtilityReward: {eval_p} not found. "
                "Using uniform reliability=1.0 until a real TRTR eval is produced."
            )

    def compute(self, ecg: np.ndarray, target_class_idx: int) -> float:
        """
        Args:
            ecg:              (1000, 12)
            target_class_idx: integer class index
        Returns:
            float in [0, 1]
        """
        if not self.available or self.model is None:
            return 0.5

        # (1, 12, 1000)  ← model expects channel-first
        x = torch.from_numpy(ecg.T[np.newaxis]).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x)                      # (1, n_classes)
            prob   = F.softmax(logits, dim=-1)          # (1, n_classes)
        idx        = min(target_class_idx, prob.shape[-1] - 1)
        confidence = float(prob[0, idx].item())
        reliability = float(self.reliability[idx]) if idx < len(self.reliability) else 1.0
        return confidence * reliability


class A3Reward:
    """
    Scores how close a generated ECG's A3-subband (slow-wave: P-wave,
    T-wave, ST-segment, baseline; 0-6.25 Hz at fs=100 Hz) energy profile is
    to the real per-class reference distribution.

    Directly motivated by Stage 3's dominant finding, replicated across
    every architecture evaluated there: A3-subband divergence is the
    largest remaining gap between generated and real ECGs. See
    Roadmap/Stage_3_Architecture_Improvements/Reports/
    Stage3_Subband_Master_Comparison.md for the table this is meant to
    close the gap against — NOTE that file currently reads "0/72 rows
    evaluated" (subband_similarity_metrics.py has never been run to
    completion on this project, no diffusion checkpoint was available
    locally). There is no real generated-vs-real A3 number recorded
    anywhere in this repo yet to numerically validate against; that has
    to happen on the GPU (run subband_similarity_metrics.py once a
    checkpoint exists), not be assumed. See Decisions.md.

    Feature extraction reuses `mentor_eval.subband_features.
    extract_subband_energy_features` VERBATIM — the exact function Stage 3's
    evaluation script (mentor_eval/subband_similarity_metrics.py) already
    uses (bior4.4 wavelet, J=3, mean-squared-coefficient energy per lead).
    Imported, not reimplemented, so evaluation and this reward cannot
    silently drift apart on what "A3 energy" means.

    Scoring reuses the exact mean/covariance/regularisation formula from
    `mentor_eval.similarity_metrics.mahalanobis_distance` (ridge-regularised
    inverse covariance, d² = diff @ inv_cov @ diff), factored so the
    per-class (mean, inv_cov) is computed ONCE at init from
    outputs/processed/a3_subband_stats.json rather than recomputed from raw
    real samples on every reward call — mahalanobis_distance as written
    recomputes the full covariance every invocation, appropriate for a
    one-shot offline evaluation report but far too expensive inside a PPO
    rollout loop called thousands of times. This is a documented, necessary
    divergence from the evaluation-side call pattern, not an independent
    reimplementation: the feature extraction and distance formula are
    identical, only the caching differs. What could cause the two to
    disagree: if a3_subband_stats.json's reference (built from X_train in
    step03) and subband_similarity_metrics.py's real-sample comparison set
    (drawn fresh from PTB-XL at eval time, different sample/seed) diverge —
    e.g. different real records sampled — the two distance numbers won't be
    bit-identical even on the same generated ECG. They should be close, not
    exact; see the validation script for how "close" is checked.

    score = exp(-mahalanobis_distance / scale), the same Gaussian-decay
    mapping RealismReward already uses for its own PCA-space distance, for
    consistency within this reward function.
    """

    def __init__(self, a3_stats_file: dict, scale: float = 8.0):
        """
        Args:
            a3_stats_file: the full JSON loaded from a3_subband_stats.json,
                {"_metadata": {...}, "classes": {cls: {mean, cov, n}}}.
                Older files without "_metadata" (flat {cls: {...}}) are
                accepted for backward compatibility but log a warning,
                since they predate the drift-detection sanity check below.
        """
        from mentor_eval.subband_features import (
            extract_subband_energy_features, SUBBAND_NAMES, WAVELET, LEVELS,
        )
        self._extract  = extract_subband_energy_features
        self._a3_idx   = SUBBAND_NAMES.index("A3")
        self._n_leads  = 12
        self._scale    = scale
        self._log      = logging.getLogger(__name__)

        metadata = a3_stats_file.get("_metadata")
        a3_stats = a3_stats_file.get("classes", a3_stats_file)  # flat-dict fallback

        if metadata is None:
            if a3_stats_file:  # non-empty but no metadata -> genuinely old-format file
                self._log.warning(
                    "A3Reward: a3_subband_stats.json has no '_metadata' block "
                    "(old-format file, predates drift-detection). Cannot verify "
                    "wavelet/decomposition-level match — re-run step03 Stage 7 "
                    "to regenerate with metadata."
                )
        else:
            # Sanity-check against THIS code's actual extraction parameters,
            # not just against what step03 wrote — catches the file being
            # regenerated later with different wavelet/level choices without
            # this reward code being updated to match.
            mismatches = []
            if metadata.get("wavelet") != WAVELET:
                mismatches.append(f"wavelet: file={metadata.get('wavelet')!r} vs code={WAVELET!r}")
            if metadata.get("decomposition_level") != LEVELS:
                mismatches.append(
                    f"decomposition_level: file={metadata.get('decomposition_level')!r} vs code={LEVELS!r}"
                )
            if metadata.get("dataset", "").split(" ")[0] != "PTB-XL":
                mismatches.append(f"dataset: file={metadata.get('dataset')!r} (expected PTB-XL)")
            if mismatches:
                raise RuntimeError(
                    "A3Reward: a3_subband_stats.json metadata does not match this "
                    "code's extraction parameters — " + "; ".join(mismatches) + ". "
                    "The reference distribution and this reward's feature "
                    "extraction must use identical wavelet/level or the "
                    "Mahalanobis distance is meaningless. Regenerate the stats "
                    "file (step03 Stage 7) or fix the mismatch before proceeding."
                )

        self.stats = a3_stats
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for cls, s in a3_stats.items():
            try:
                mean = np.asarray(s["mean"], dtype=np.float64)
                cov  = np.asarray(s["cov"], dtype=np.float64)
                # Same ridge regularisation constant as mahalanobis_distance.
                cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
                inv_cov = np.linalg.pinv(cov_reg)
                self._cache[cls] = (mean, inv_cov)
            except Exception as exc:
                self._log.warning(f"A3Reward: failed to load stats for class {cls!r}: {exc}")

        if not self._cache:
            # Fail loudly, not a silent 0.5-neutral fallback: a3_subband_
            # stats.json missing/empty means there is no frozen real-data
            # reference distribution for A3Reward to score against at all.
            # Falling back to neutral would let a run silently train with
            # a3's weight nonzero but its score constant — indistinguishable
            # from a real (near-)neutral A3 signal in the logs. This must
            # be caught before RL starts, not discovered from a flat reward
            # curve after the fact.
            raise RuntimeError(
                "A3Reward: no per-class stats loaded — a3_subband_stats.json "
                "is missing, empty, or every class's stats failed to parse. "
                "Run step03_eda_and_class_mapping.py (Stage 7) first to build "
                "the frozen real-data A3 reference distribution. Refusing to "
                "silently proceed with a neutral/rebuilt fallback."
            )

    def compute(self, ecg: np.ndarray, target_class: str) -> float:
        """
        Args:
            ecg:          (1000, 12) z-score normalised ECG
            target_class: class name string (e.g. 'MI', 'NORM')
        Returns:
            float in [0, 1]
        """
        if target_class not in self._cache:
            return 0.5   # no reference — neutral, same convention as MorphologyReward/HRVReward

        mean, inv_cov = self._cache[target_class]
        full_feats = self._extract(ecg)   # (len(SUBBAND_NAMES) * 12,)
        feat = full_feats[self._a3_idx * self._n_leads:(self._a3_idx + 1) * self._n_leads]

        diff = feat - mean
        d2   = float(diff @ inv_cov @ diff)
        d2   = max(d2, 0.0)
        distance = float(np.sqrt(d2))
        return float(np.clip(np.exp(-distance / self._scale), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Composite reward
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalReward:
    """
    Composite clinical reward — the full reward signal used in RL fine-tuning.

      r(x) = w_morph·r_morph + w_hrv·r_hrv + w_real·r_real + w_diag·r_diag
             + w_a3·r_a3

    All components return float in [0, 1]; total is also in [0, 1] when weights
    sum to 1.0.

    Use get_reward(config_name) to obtain ablation variants for step09.

    Performance note: neurokit2 dominates latency (~20–50 ms/ECG on CPU).
    PCA and CNN together add < 5 ms. A3 (pywt DWT, J=3, 12 leads) is
    benchmarked by the smoke test (step07_rl_finetuning.py --smoke-test) —
    do not assume it's cheap, measure it. Target: < 100 ms/ECG.
    """

    def __init__(
        self,
        morph_reward: MorphologyReward,
        hrv_reward:   HRVReward,
        real_reward:  RealismReward,
        diag_reward:  DiagnosticUtilityReward,
        a3_reward:    A3Reward,
        weights:      dict[str, float],
        class_names:  list[str],
    ):
        self.morph       = morph_reward
        self.hrv         = hrv_reward
        self.real        = real_reward
        self.diag        = diag_reward
        self.a3          = a3_reward
        self.weights     = weights
        self.class_names = class_names

        # Per-component wall-clock timing (ms), accumulated across compute()
        # calls. Overhead is a handful of perf_counter() calls (~tens of ns)
        # so this stays on unconditionally — needed to catch a component that
        # turns out to dominate the PPO rollout loop before a real GPU run
        # commits hours to it.
        self._timing: dict[str, list[float]] = {
            "morph": [], "hrv": [], "real": [], "diag": [], "a3": [], "total": [],
        }

    # ── Single ECG ────────────────────────────────────────────────────────────

    def compute(
        self,
        ecg:              np.ndarray,    # (1000, 12)
        target_class:     str,
        target_class_idx: int,
    ) -> dict[str, float]:
        """
        Evaluate one ECG.

        Returns:
            dict with keys 'total', 'r_morph', 'r_hrv', 'r_real', 'r_diag', 'r_a3'
            All values float in [0, 1].
        """
        import time

        t0 = time.perf_counter()
        r_morph = self.morph.compute(ecg, target_class)
        t1 = time.perf_counter()
        r_hrv   = self.hrv.compute(ecg)
        t2 = time.perf_counter()
        r_real  = self.real.compute(ecg)
        t3 = time.perf_counter()
        r_diag  = self.diag.compute(ecg, target_class_idx)
        t4 = time.perf_counter()
        r_a3    = self.a3.compute(ecg, target_class)
        t5 = time.perf_counter()

        self._timing["morph"].append((t1 - t0) * 1000)
        self._timing["hrv"].append((t2 - t1) * 1000)
        self._timing["real"].append((t3 - t2) * 1000)
        self._timing["diag"].append((t4 - t3) * 1000)
        self._timing["a3"].append((t5 - t4) * 1000)
        self._timing["total"].append((t5 - t0) * 1000)

        w     = self.weights
        total = (
            w.get("morph", 0.3) * r_morph
            + w.get("hrv",   0.3) * r_hrv
            + w.get("real",  0.2) * r_real
            + w.get("diag",  0.2) * r_diag
            + w.get("a3",    0.0) * r_a3
        )

        return {
            "total":   float(np.clip(total,   0.0, 1.0)),
            "r_morph": float(np.clip(r_morph, 0.0, 1.0)),
            "r_hrv":   float(np.clip(r_hrv,   0.0, 1.0)),
            "r_real":  float(np.clip(r_real,  0.0, 1.0)),
            "r_a3":    float(np.clip(r_a3,    0.0, 1.0)),
            "r_diag":  float(np.clip(r_diag,  0.0, 1.0)),
        }

    def get_timing_summary(self, reset: bool = False) -> dict[str, dict[str, float]]:
        """
        Mean/max/n wall-clock ms per component, accumulated since init (or
        since the last reset). Use this to find out which component actually
        dominates the reward path before assuming any one of them is cheap.
        """
        summary = {}
        for k, vals in self._timing.items():
            if vals:
                summary[k] = {
                    "mean_ms": float(np.mean(vals)),
                    "max_ms":  float(np.max(vals)),
                    "n":       len(vals),
                }
            else:
                summary[k] = {"mean_ms": None, "max_ms": None, "n": 0}
        if reset:
            for k in self._timing:
                self._timing[k] = []
        return summary

    # ── Batch convenience ─────────────────────────────────────────────────────

    def compute_batch(
        self,
        ecgs:             np.ndarray,   # (B, 1000, 12)
        target_class:     str,
        target_class_idx: int,
    ) -> list[dict[str, float]]:
        """Evaluate B ECGs. Returns list of reward dicts (one per ECG)."""
        return [self.compute(ecg, target_class, target_class_idx) for ecg in ecgs]

    def total_batch(
        self,
        ecgs:             np.ndarray,
        target_class:     str,
        target_class_idx: int,
    ) -> np.ndarray:
        """Return just the total reward as a 1-D numpy array of shape (B,)."""
        return np.array([
            self.compute(ecg, target_class, target_class_idx)["total"]
            for ecg in ecgs
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────────────────────

def get_reward(
    config_name:  str                    = "full",
    cfg=None,
    X_train:      Optional[np.ndarray]   = None,
    class_names:  Optional[list[str]]    = None,
    device:       str                    = "cpu",
) -> ClinicalReward:
    """
    Build a ClinicalReward instance with the specified ablation configuration.

    Args:
        config_name: one of 'full', 'diag_only', 'no_diag', 'no_morph', 'no_hrv'
        cfg:         OmegaConf config; loaded from disk if None
        X_train:     (N, 1000, 12) training signals for PCA; loaded if None
        class_names: ordered class list; loaded from class_names.json if None
        device:      torch device for the DiagnosticUtilityReward CNN

    Returns:
        ClinicalReward ready for .compute() calls
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown config_name {config_name!r}. "
            f"Choose from {list(ABLATION_CONFIGS)}"
        )

    if cfg is None:
        cfg = load_config()

    processed_dir = Path(cfg.paths.outputs.processed)

    if class_names is None:
        cn_path = processed_dir / "class_names.json"
        class_names = json.load(open(cn_path)) if cn_path.exists() else list(cfg.ptbxl.classes)

    if X_train is None:
        X_train = np.load(str(processed_dir / "X_train.npy"))

    fs = float(cfg.ptbxl.sampling_rate)
    rc = cfg.reward

    # Load reference stats (graceful: empty dict if not found)
    def _load_json(path: Path) -> dict:
        return json.load(open(path)) if path.exists() else {}

    morph_stats = _load_json(processed_dir / "morphology_stats.json")
    hrv_stats   = _load_json(processed_dir / "hrv_stats.json")

    # Build components
    morph = MorphologyReward(morph_stats, fs=fs)
    hrv   = HRVReward(hrv_stats, fs=fs)
    real  = RealismReward(
        X_train,
        n_components=int(rc.pca_components),
        n_samples=int(rc.pca_n_train_samples),
        fs=fs,
    )
    # trtr_classifier.pt (real-data-trained), NOT tstr_classifier.pt -- the
    # latter is trained entirely on synthetic samples from the baseline
    # diffusion model, which is a reward-hacking risk when used to fine-tune
    # that same model. See Roadmap/Stage_4_Optimization/Decisions.md.
    classifier_path = str(Path(cfg.paths.outputs.models) / "trtr_classifier.pt")
    use_reliability = bool(rc.get("use_reliability_scaling", True))
    diag = DiagnosticUtilityReward(
        classifier_path, n_classes=len(class_names), device=device,
        use_reliability=use_reliability,
    )

    # ── Weights ──────────────────────────────────────────────────────────────
    # BUG FOUND AND FIXED HERE: this used to always take weights from the
    # hardcoded ABLATION_CONFIGS table, even for config_name="full" — meaning
    # cfg.reward.weights (config.yaml) was NEVER actually read. Every weight
    # number discussed for config.yaml up to this point (0.4/0.4/0.15/0.05,
    # the reliability-scaling defaults, etc.) was dead config, not what
    # training actually used (ABLATION_CONFIGS["full"]'s hardcoded
    # 0.3/0.3/0.2/0.2 was). See Decisions.md.
    #
    # Named ablation variants (diag_only, no_morph, ...) intentionally still
    # use the fixed ABLATION_CONFIGS table regardless of cfg — that's the
    # point of an ablation study: known, fixed configurations for
    # comparison, not whatever happens to be in config.yaml. Only "full"
    # (the actual training configuration) now defers to cfg.reward.weights.
    if config_name == "full" and rc.get("weights") is not None:
        weights = {k: float(v) for k, v in rc.weights.items()}
        weights.setdefault("a3", 0.0)
    else:
        weights = ABLATION_CONFIGS[config_name].copy()

    a3_stats = _load_json(processed_dir / "a3_subband_stats.json")
    a3 = A3Reward(a3_stats)

    return ClinicalReward(morph, hrv, real, diag, a3, weights, class_names)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic classifier builder (called when tstr_classifier.pt doesn't exist)
# ──────────────────────────────────────────────────────────────────────────────

def build_diagnostic_classifier(
    X_train:   np.ndarray,   # (N, 1000, 12)
    y_train:   np.ndarray,   # (N,)
    n_classes: int,
    save_path: Path,
    epochs:    int = 20,
    device:    str = "cpu",
    log=None,
) -> None:
    """
    Train a Simple1DCNN on real data and save to save_path.
    Called automatically in the self-test when tstr_classifier.pt is absent.
    """
    from step05_baseline_eval import Simple1DCNN

    _info = (log.info if log else print)
    _info(f"Training diagnostic classifier ({epochs} epochs) …")

    Xtr = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    ytr = torch.from_numpy(y_train).long()

    counts  = Counter(y_train.tolist())
    sampler = WeightedRandomSampler(
        weights=[1.0 / counts[int(l)] for l in y_train.tolist()],
        num_samples=len(y_train), replacement=True,
    )
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=64, sampler=sampler, drop_last=True
    )

    model = Simple1DCNN(n_classes=n_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss   = crit(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if ep % 5 == 0:
            _info(f"  epoch {ep:02d}/{epochs}  loss={ep_loss/len(loader):.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_classes": n_classes}, str(save_path))
    _info(f"Saved diagnostic classifier → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _load_labels_simple(
    record_ids:    np.ndarray,
    ptbxl_csv:    Path,
    class_mapping: dict,
    class_names:  list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal label loading for self-test (mirrors step04/05 logic)."""
    import ast, pandas as pd

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    db = pd.read_csv(str(ptbxl_csv), index_col="ecg_id")
    valid, labels = [], []
    for i, eid in enumerate(record_ids):
        eid = int(eid)
        if eid not in db.index:
            continue
        raw = str(db.at[eid, "scp_codes"])
        try:
            scp = ast.literal_eval(raw)
        except Exception:
            continue
        best_cls, best_conf = None, -1.0
        for code, conf in scp.items():
            m = class_mapping.get(code.upper())
            if m and m in name_to_idx and conf > best_conf:
                best_cls, best_conf = m, conf
        if best_cls is None:
            if "OTHER" in name_to_idx:
                best_cls = "OTHER"
            else:
                continue
        valid.append(i)
        labels.append(name_to_idx[best_cls])
    return np.array(valid, dtype=np.int64), np.array(labels, dtype=np.int64)


def _selftest(cfg, log) -> float:
    """
    Validate the reward function by comparing real vs generated ECGs.

    Expected: real ECGs score higher on average than generated ones
    (before RL fine-tuning the diffusion model doesn't match real morphology).

    Saves: fig04_reward_components.pdf + reward_function_validated.pkl
    """
    import time

    processed_dir = Path(cfg.paths.outputs.processed)
    results_dir   = Path(cfg.paths.outputs.results)
    models_dir    = Path(cfg.paths.outputs.models)
    gen_dir       = Path(cfg.paths.outputs.generated) / "baseline_samples"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load class info ───────────────────────────────────────────────────────
    with open(processed_dir / "class_names.json")   as f: class_names   = json.load(f)
    with open(processed_dir / "class_mapping.json") as f: class_mapping = json.load(f)
    n_classes = len(class_names)

    # ── Load X_train (needed for PCA) ─────────────────────────────────────────
    log.info("Loading X_train for PCA fitting …")
    X_train    = np.load(str(processed_dir / "X_train.npy"))
    rec_ids_tr = np.load(str(processed_dir / "record_ids_train.npy"))
    db_path    = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"

    vi_tr, y_train = _load_labels_simple(rec_ids_tr, db_path, class_mapping, class_names)
    X_train        = X_train[vi_tr]

    # ── Build / ensure diagnostic classifier ──────────────────────────────────
    clf_path = models_dir / "tstr_classifier.pt"
    if not clf_path.exists():
        log.warning("tstr_classifier.pt not found — training from scratch for self-test …")
        build_diagnostic_classifier(X_train, y_train, n_classes, clf_path, epochs=15, log=log)

    # ── Build ClinicalReward ('full' config) ──────────────────────────────────
    reward_fn = get_reward(
        config_name="full", cfg=cfg,
        X_train=X_train, class_names=class_names,
        device="cpu",
    )
    log.info("ClinicalReward built.")

    # ── Load 10 real ECGs from test set ──────────────────────────────────────
    X_test     = np.load(str(processed_dir / "X_test.npy"))
    rec_ids_te = np.load(str(processed_dir / "record_ids_test.npy"))
    vi_te, y_test = _load_labels_simple(rec_ids_te, db_path, class_mapping, class_names)
    X_test, y_test = X_test[vi_te], y_test[vi_te]

    # Pick the most represented class for a fair comparison
    counts     = Counter(y_test.tolist())
    eval_class_idx  = max(counts, key=counts.get)
    eval_class_name = class_names[eval_class_idx]
    log.info(f"Self-test class: {eval_class_name} (index {eval_class_idx})")

    real_mask  = (y_test == eval_class_idx)
    real_ecgs  = X_test[real_mask][:10]   # (≤10, 1000, 12)

    # ── Load 10 generated ECGs ────────────────────────────────────────────────
    gen_ecgs: list[np.ndarray] = []
    if gen_dir.exists():
        for i in range(10):
            p = gen_dir / f"class_{eval_class_name}_sample_{i:04d}.npy"
            if p.exists():
                gen_ecgs.append(np.load(str(p)))
    if not gen_ecgs:
        log.warning(
            f"No generated samples found in {gen_dir}. "
            "Using white noise as placeholder — scores will be low."
        )
        gen_ecgs = [np.random.randn(1000, 12).astype(np.float32) for _ in range(10)]
    gen_ecgs = np.array(gen_ecgs[:10])   # (≤10, 1000, 12)

    # ── Compute rewards ───────────────────────────────────────────────────────
    log.info(f"Computing rewards for {len(real_ecgs)} real + {len(gen_ecgs)} generated ECGs …")

    t0 = time.perf_counter()
    real_rewards = [reward_fn.compute(ecg, eval_class_name, eval_class_idx) for ecg in real_ecgs]
    elapsed = (time.perf_counter() - t0) / len(real_ecgs)
    log.info(f"  Average reward computation time: {elapsed * 1000:.1f} ms/ECG")
    if elapsed > 0.1:
        log.warning(f"  ⚠ Exceeds 100 ms target ({elapsed*1000:.1f} ms). "
                    "Consider disabling neurokit2 in batch settings.")

    gen_rewards = [reward_fn.compute(ecg, eval_class_name, eval_class_idx) for ecg in gen_ecgs]

    mean_real = float(np.mean([r["total"] for r in real_rewards]))
    mean_gen  = float(np.mean([r["total"] for r in gen_rewards]))
    log.info(f"  Mean real reward:      {mean_real:.4f}")
    log.info(f"  Mean generated reward: {mean_gen:.4f}")
    if mean_real <= mean_gen:
        log.warning(
            "Real ECGs did not score higher than generated. "
            "This can happen when morphology_stats.json or hrv_stats.json are absent, "
            "or when the generated samples are already high-quality (post-RL)."
        )

    # ── Figure 4: bar chart of component scores ────────────────────────────────
    _make_figure4(real_rewards, gen_rewards, eval_class_name, results_dir, log)

    # ── Save validation pickle ────────────────────────────────────────────────
    pkl_path = processed_dir / "reward_function_validated.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "config_name":      "full",
                "eval_class":       eval_class_name,
                "mean_real_reward": mean_real,
                "mean_gen_reward":  mean_gen,
                "real_rewards":     real_rewards,
                "gen_rewards":      gen_rewards,
                "weights":          reward_fn.weights,
            },
            f,
        )
    log.info(f"Saved reward_function_validated.pkl → {pkl_path}")

    return mean_real, mean_gen


def _make_figure4(
    real_rewards: list[dict],
    gen_rewards:  list[dict],
    class_name:   str,
    results_dir:  Path,
    log,
) -> None:
    """
    Paper Figure 4: grouped bar chart of all 5 reward components.
    Two bars per group: real ECG (green) vs generated ECG (orange).
    """
    components  = ["r_morph", "r_hrv", "r_real", "r_diag", "r_a3"]
    labels      = ["Morphology", "HRV", "Realism", "Diagnostic", "A3-subband"]
    colors_real = "#2ca02c"
    colors_gen  = "#ff7f0e"

    real_means = np.array([np.mean([r[c] for r in real_rewards]) for c in components])
    real_stds  = np.array([np.std( [r[c] for r in real_rewards]) for c in components])
    gen_means  = np.array([np.mean([r[c] for r in gen_rewards])  for c in components])
    gen_stds   = np.array([np.std( [r[c] for r in gen_rewards])  for c in components])

    x     = np.arange(len(components))
    width = 0.35

    with plt.rc_context(PUBSTYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        bars_r = ax.bar(
            x - width / 2, real_means, width,
            yerr=real_stds, capsize=5, color=colors_real,
            label=f"Real ECG ({class_name})", alpha=0.9, ecolor="black",
        )
        bars_g = ax.bar(
            x + width / 2, gen_means, width,
            yerr=gen_stds,  capsize=5, color=colors_gen,
            label="Baseline Diffusion", alpha=0.9, ecolor="black",
        )

        for bars, means in ((bars_r, real_means), (bars_g, gen_means)):
            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Component score [0, 1]", fontsize=11)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(
            f"Clinical Reward Components — Real vs Baseline Generated ({class_name})\n"
            "Higher = more clinically valid. Real ECGs should score higher before RL fine-tuning.",
            fontsize=10,
        )
        ax.legend(fontsize=10)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    for ext in ("pdf", "png"):
        fig.savefig(str(results_dir / f"fig04_reward_components.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig04_reward_components.{pdf,png}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step06_reward_function", cfg=cfg)
    set_seed(cfg.seeds[0])

    log.info("=" * 60)
    log.info("Running reward function self-test …")
    log.info("=" * 60)

    mean_real, mean_gen = _selftest(cfg, log)

    log.info("=" * 60)
    print(
        f"✓ Reward function validated. "
        f"Mean real reward: {mean_real:.2f} vs generated: {mean_gen:.2f}"
    )


if __name__ == "__main__":
    main()
