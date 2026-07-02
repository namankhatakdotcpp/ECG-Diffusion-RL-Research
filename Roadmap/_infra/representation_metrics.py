"""
representation_metrics.py

Two complementary metrics for "does this layer's hidden-state population
actually separate by class," used in the Representation Collapse Analysis
(Stage 2, Tier 0, item 8).

fisher_ratio()
    Cheap, closed-form, no training required. Scans every block in
    seconds. Use it FIRST, across all blocks, to find where separability
    drops.

    Trace-based multivariate generalization of the one-way ANOVA F-ratio:

        F = tr(S_B) / tr(S_W)
          = [ sum_c  n_c * ||mu_c - mu||^2 ]
            -------------------------------
            [ sum_c  sum_{i in c} ||x_i - mu_c||^2 ]

    where mu_c is the class-c mean feature vector, mu is the global mean,
    and n_c is the number of samples in class c. Using the trace (rather
    than the full S_W^-1 S_B eigen-decomposition used in classical Fisher
    LDA) avoids inverting a covariance matrix estimated from very few
    samples, which would be numerically unstable here — model_dim is 256
    and some classes have single-digit sample counts.

    ASSUMES roughly comparable within-class spread across classes (shared-
    covariance-like assumption), and -- because it is trace-based, summing
    variance equally across all D dimensions -- it can UNDER-REPORT
    separability that lives in a low-dimensional subspace against many
    noise-only dimensions. Verified empirically: 256-d features with 5
    class centers spaced ~9 within-class std-devs apart along a SINGLE
    discriminating direction, plus unit-variance isotropic noise applied
    identically across all 256 dims (so 255 of them carry only noise),
    gave a Fisher ratio well short of what "essentially perfectly
    separable" should look like (a middling value, not a large one),
    while linear_probe_accuracy() on the same data correctly found 100%
    test accuracy (chance = 20%). The exact Fisher-ratio number is a
    function of the synthetic setup's specific noise/spacing parameters,
    not a universal constant to expect elsewhere -- the reproducible
    finding is the qualitative pattern (Fisher ratio understates
    separability the probe finds trivially), not a specific numeric
    threshold. Do NOT use a low OR middling Fisher ratio to decide to skip
    running linear_probe_accuracy() at that layer -- run both at every
    block. Only linear_probe_accuracy() is the confirmatory test;
    fisher_ratio() is a cheap, fast, informative-but-incomplete first
    pass, not a gate.

linear_probe_accuracy()
    More expensive (trains a small classifier) but far more robust — makes
    no distributional assumption about the features, only asks "can a
    linear boundary decode the class label from this layer's features at
    all." This is the standard "probing classifier" methodology from the
    representation-learning literature. Use it ONLY at the 1-2 blocks that
    fisher_ratio() flags as interesting, not at every block — it's Tier
    0.5, not Tier 0: still no diffusion retraining required, but it does
    require fitting a small sklearn model, so don't run it 6x per
    experiment by default.

Both functions exclude any class with fewer than `min_class_count` samples
and report which classes were excluded, rather than silently including a
class whose "variance" is really a single pairwise distance in disguise.

Correction (2026-07-02): an earlier version of this docstring justified
the guard with "the OTHER class problem: n=2 training sequences
project-wide" — that number is wrong for this project. The real,
verified count (outputs/processed/class_counts.json) is OTHER: train=254,
val=29, test=30 (339 after the Finding-5 tie-break fix in
utils/label_assignment.py) — well above both this module's default
min_class_count thresholds (5 for fisher_ratio, 10 for linear_probe_accuracy).
The guard is not defending against OTHER's *project-wide* total; it
defends against any class ending up with very few samples inside a
SMALL PER-BLOCK PROBE BATCH specifically — Tier 0 diagnostics intentionally
work on small generated/pooled batches (e.g. n_gen=20/class), not the
full split, so a project-wide-common class can still be underrepresented
in one specific batch. Keep this distinction in mind when interpreting
any "class excluded" warning this module emits: it describes the batch
passed in, not the class's real prevalence in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FisherRatioResult:
    fisher_ratio: float
    n_classes_used: int
    classes_used: list
    classes_excluded_low_n: list
    n_total_samples_used: int
    warning: Optional[str] = None


def fisher_ratio(
    features: np.ndarray,
    labels: np.ndarray,
    min_class_count: int = 5,
) -> FisherRatioResult:
    """
    Args:
        features: (N, D) array — one pooled feature vector per sample
                  (e.g. mean-pooled over the 600 tokens at a given block).
        labels:   (N,) array of class indices.
        min_class_count: classes with fewer samples than this are excluded
                  from the calculation entirely (not just down-weighted) —
                  a variance estimate from 2 samples is not an estimate.

    Returns:
        FisherRatioResult. If fewer than 2 classes survive the min-count
        filter, fisher_ratio is np.nan and `warning` explains why — this
        is a valid, informative result (it tells you the sample size is
        too small at this layer/condition to say anything), not a bug.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"features has {len(features)} rows but labels has {len(labels)} entries"
        )

    unique, counts = np.unique(labels, return_counts=True)
    classes_used = [int(c) for c, n in zip(unique, counts) if n >= min_class_count]
    classes_excluded = [int(c) for c, n in zip(unique, counts) if n < min_class_count]

    if len(classes_used) < 2:
        return FisherRatioResult(
            fisher_ratio=float("nan"),
            n_classes_used=len(classes_used),
            classes_used=classes_used,
            classes_excluded_low_n=classes_excluded,
            n_total_samples_used=0,
            warning=(
                f"Only {len(classes_used)} class(es) had >= {min_class_count} "
                f"samples; Fisher ratio requires at least 2. Excluded: "
                f"{classes_excluded}. Increase samples per class or lower "
                f"min_class_count (not recommended below ~5)."
            ),
        )

    mask = np.isin(labels, classes_used)
    X = features[mask]
    y = labels[mask]
    global_mean = X.mean(axis=0)

    between = 0.0
    within = 0.0
    for c in classes_used:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        n_c = len(Xc)
        between += n_c * float(np.sum((mu_c - global_mean) ** 2))
        within += float(np.sum((Xc - mu_c) ** 2))

    if within == 0.0:
        ratio = float("inf") if between > 0 else float("nan")
        warning = "Within-class scatter is exactly zero — check for duplicate/degenerate features."
    else:
        ratio = between / within
        warning = None

    return FisherRatioResult(
        fisher_ratio=ratio,
        n_classes_used=len(classes_used),
        classes_used=classes_used,
        classes_excluded_low_n=classes_excluded,
        n_total_samples_used=int(mask.sum()),
        warning=warning,
    )


@dataclass
class LinearProbeResult:
    accuracy: float
    chance_accuracy: float
    n_classes_used: int
    classes_used: list
    classes_excluded_low_n: list
    train_accuracy: float
    n_train_samples: int
    feature_dim_used: int
    pca_components: Optional[int] = None
    warning: Optional[str] = None


def linear_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    min_class_count: int = 10,
    test_fraction: float = 0.3,
    seed: int = 42,
    samples_per_dim_floor: float = 5.0,
) -> LinearProbeResult:
    """
    Fits logistic regression on a held-out split and reports test accuracy.

    CRITICAL: when feature_dim (D) is comparable to or larger than the
    number of training samples (n_train), an unregularized linear
    classifier can fit ANY labeling -- including pure noise -- to ~100%
    TRAIN accuracy. This is not a hypothetical edge case for this project:
    model_dim=256, and any realistic Tier-0 probe batch will have tens,
    not hundreds, of samples per class. An n_train < D probe proves
    nothing about real separability. Verified directly: on genuinely
    unrelated features/labels (D=256, n_train=31), an unguarded probe hit
    train_accuracy=1.000 while test_accuracy collapsed to 0.214 (below the
    0.333 chance rate) -- a large train/test gap that IS visible if you
    check for it, but is easy to miss if only train accuracy (or a single
    accuracy number with no train/test split at all) is reported. The
    `train_accuracy - test_accuracy > 0.25` check below exists specifically
    to surface this gap automatically.

    Fix applied here: if n_train < samples_per_dim_floor * D, features are
    first reduced via PCA (fit on the training split only, to avoid
    test-set leakage) to n_components = max(2, n_train // samples_per_dim_floor)
    before the probe is fit. This is recorded in the returned
    pca_components field -- if that field is populated, the reported
    accuracy is on PCA-reduced features, not raw ones, and that should be
    stated wherever this result is cited.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "linear_probe_accuracy requires scikit-learn. Install it or "
            "restrict Stage 2 representation-collapse analysis to "
            "fisher_ratio() only."
        ) from e

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    unique, counts = np.unique(labels, return_counts=True)
    classes_used = [int(c) for c, n in zip(unique, counts) if n >= min_class_count]
    classes_excluded = [int(c) for c, n in zip(unique, counts) if n < min_class_count]

    if len(classes_used) < 2:
        return LinearProbeResult(
            accuracy=float("nan"),
            chance_accuracy=float("nan"),
            n_classes_used=len(classes_used),
            classes_used=classes_used,
            classes_excluded_low_n=classes_excluded,
            train_accuracy=float("nan"),
            n_train_samples=0,
            feature_dim_used=features.shape[1] if features.ndim == 2 else 0,
            warning=(
                f"Only {len(classes_used)} class(es) had >= {min_class_count} "
                f"samples; a linear probe needs at least 2 classes with "
                f"enough samples each to hold out a test split. Excluded: "
                f"{classes_excluded}."
            ),
        )

    mask = np.isin(labels, classes_used)
    X, y = features[mask], labels[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )

    D = X_train.shape[1]
    n_train = X_train.shape[0]
    pca_components: Optional[int] = None

    if n_train < samples_per_dim_floor * D:
        pca_components = max(2, min(D, int(n_train // samples_per_dim_floor)))
        pca = PCA(n_components=pca_components, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # multi_class="multinomial" was the explicit way to request this before
    # scikit-learn 1.5; lbfgs (the default solver) now always fits a proper
    # multinomial model for >2 classes, so no extra kwarg is needed here.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    chance = 1.0 / len(classes_used)

    warnings_list = []
    if pca_components is not None:
        warnings_list.append(
            f"n_train ({n_train}) < {samples_per_dim_floor:.0f}x feature_dim ({D}) "
            f"-- reduced to {pca_components} PCA components before probing to "
            f"avoid the interpolation regime. Accuracy is on reduced features."
        )
    if train_acc - test_acc > 0.25:
        warnings_list.append(
            f"train_accuracy ({train_acc:.3f}) exceeds test_accuracy "
            f"({test_acc:.3f}) by more than 0.25 -- residual overfitting. "
            f"Trust test_accuracy, not train_accuracy, for any conclusion."
        )

    return LinearProbeResult(
        accuracy=test_acc,
        chance_accuracy=chance,
        n_classes_used=len(classes_used),
        classes_used=classes_used,
        classes_excluded_low_n=classes_excluded,
        train_accuracy=train_acc,
        n_train_samples=n_train,
        feature_dim_used=D,
        pca_components=pca_components,
        warning="; ".join(warnings_list) if warnings_list else None,
    )
