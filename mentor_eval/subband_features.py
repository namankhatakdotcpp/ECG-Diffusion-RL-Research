"""
mentor_eval/subband_features.py — wavelet subband decomposition + energy
features, following the multiscale-energy approach of Sharma, Tripathy &
Dandapat, "Multiscale Energy and Eigenspace Approach to Detection and
Localization of Myocardial Infarction," IEEE TBME 62(7), 2015 ("MEES").

WAVELET CHOICE
---------------
Library: PyWavelets (pywt).
Family:  bior4.4 — PyWavelets' name for the Daubechies/CDF 9/7 biorthogonal
         filter pair, matching the paper's "Daubechies 9/7 biorthogonal
         wavelet filters" (Sec. II.A.3, Fig. 5 caption) exactly.

DECOMPOSITION LEVEL — re-derived, not copied
----------------------------------------------
The paper uses 6-level decomposition and names subbands A6/D6/D5/D4 as the
diagnostically relevant ones, but that mapping was derived AT THEIR SAMPLING
RATE of 1000 Hz (confirmed in their Sec. III.A). At fs=1000 Hz, a 6-level
dyadic decomposition gives:
    A6: 0-7.8 Hz      (P-wave, T-wave, ST-segment, baseline)
    D6: 7.8-15.6 Hz   (low-frequency QRS)
    D5: 15.6-31.25 Hz (full QRS complex)
    D4: 31.25-62.5 Hz (high-frequency QRS / Q-wave edges)

PTB-XL (this repo) is sampled at 100 Hz — 10x lower. ECG wave content is a
fixed physiological fact in absolute Hz, independent of sampling rate. Using
J=6 here would push real QRS energy (~8-50 Hz absolute) almost entirely out
of the "A6/D6/D5/D4"-named bands (which would only span 0-6.25 Hz total at
fs=100) and into D3/D2/D1 — the bands the paper calls "predominantly noise"
at ITS sampling rate, but which contain our actual QRS information.

Fix: use J=3 (since log2(1000/100) ~ 3.3, i.e. ~3 fewer levels are needed to
cover the same absolute-Hz range at 10x lower fs). At fs=100 Hz:
    A3: 0-6.25 Hz      (P/T-wave, ST-segment, baseline)   ~ paper's A6
    D3: 6.25-12.5 Hz   (low-frequency QRS)                 ~ paper's D6
    D2: 12.5-25 Hz     (full QRS complex)                  ~ paper's D5
    D1: 25-50 Hz       (high-frequency QRS edges, Nyquist) ~ paper's D4

This preserves the same CLINICAL correspondence (one slow-wave band, three
ascending QRS-detail bands) rather than blindly copying level indices.
Confirmed with the project owner before implementation.
"""

from __future__ import annotations

import numpy as np
import pywt

WAVELET = "bior4.4"
LEVELS = 3
SUBBAND_NAMES = ["A3", "D3", "D2", "D1"]  # approximation, then details coarsest (D3) to finest (D1)

# Clinical labels, ordered to match SUBBAND_NAMES — analogous to the paper's
# A6/D6/D5/D4 roles, re-derived for our sampling rate (see module docstring).
SUBBAND_CLINICAL_LABEL = {
    "A3": "P/T-wave, ST-segment, baseline",
    "D3": "low-frequency QRS",
    "D2": "full QRS complex",
    "D1": "high-frequency QRS edges",
}


def subband_frequency_ranges(fs: float, levels: int = LEVELS) -> dict[str, tuple[float, float]]:
    """Analytic dyadic subband frequency ranges for an `levels`-level DWT at
    sampling rate fs. Dj = [fs/2^(j+1), fs/2^j]; AJ = [0, fs/2^(J+1)]."""
    ranges: dict[str, tuple[float, float]] = {}
    nyquist = fs / 2.0
    for j in range(1, levels + 1):
        lo = fs / (2 ** (j + 1))
        hi = fs / (2 ** j)
        ranges[f"D{j}"] = (lo, hi)
    ranges[f"A{levels}"] = (0.0, fs / (2 ** (levels + 1)))
    return ranges


def decompose_signal(signal_1d: np.ndarray, levels: int = LEVELS) -> dict[str, np.ndarray]:
    """`levels`-level DWT of a 1-D signal -> {subband_name: coeffs}.

    pywt.wavedec returns [cA_J, cD_J, cD_{J-1}, ..., cD_1] (coarsest first).
    """
    coeffs = pywt.wavedec(signal_1d, WAVELET, level=levels)
    cA_J = coeffs[0]
    cD_list = coeffs[1:]  # [cD_J, cD_{J-1}, ..., cD_1]
    out = {f"A{levels}": cA_J}
    for j, cD in zip(range(levels, 0, -1), cD_list):
        out[f"D{j}"] = cD
    return out


def subband_energy(coeffs: np.ndarray) -> float:
    """Mean squared coefficient — matches the paper's Eq. (1)/(2) exactly."""
    return float(np.mean(np.square(coeffs)))


def extract_subband_energy_features(signal_12lead: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    """(1000, 12) real-valued 12-lead ECG -> (len(SUBBAND_NAMES) * 12,) energy
    feature vector, ordered [A3_lead0..11, D3_lead0..11, D2_lead0..11, D1_lead0..11].
    Same flat-vector convention as similarity_metrics.extract_features, so it
    drops directly into mahalanobis_distance / bhattacharyya_distance /
    matched_cosine_similarity without modification.
    """
    n_leads = signal_12lead.shape[1]
    out = np.zeros(len(SUBBAND_NAMES) * n_leads, dtype=np.float64)
    for lead_idx in range(n_leads):
        sub = decompose_signal(signal_12lead[:, lead_idx], levels=levels)
        for band_i, band_name in enumerate(SUBBAND_NAMES):
            out[band_i * n_leads + lead_idx] = subband_energy(sub[band_name])
    return out


def extract_subband_energy_batch(signals: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    """(N, 1000, 12) -> (N, len(SUBBAND_NAMES) * 12) energy feature matrix."""
    return np.stack([extract_subband_energy_features(s, levels=levels) for s in signals])


def subband_energy_per_lead(signal_12lead: np.ndarray, band_name: str, levels: int = LEVELS) -> np.ndarray:
    """(1000, 12) -> (12,) energy of ONE named subband, one value per lead.
    Used by the decomposition table / box plots, which facet by subband.
    """
    n_leads = signal_12lead.shape[1]
    out = np.zeros(n_leads, dtype=np.float64)
    for lead_idx in range(n_leads):
        sub = decompose_signal(signal_12lead[:, lead_idx], levels=levels)
        out[lead_idx] = subband_energy(sub[band_name])
    return out
