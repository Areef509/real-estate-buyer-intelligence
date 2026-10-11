"""Internal validity metrics and stability analysis."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .clustering import FittedModel, assign_to_centroids, fit_model


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette (higher better), Calinski-Harabasz (higher), Davies-Bouldin (lower)."""
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def smallest_cluster_share(labels: np.ndarray) -> float:
    return float(np.bincount(labels).min() / len(labels))


def seed_stability(X: np.ndarray, name: str, k: int, seeds: list[int], n_init: int = 1) -> float:
    """Mean pairwise ARI across random restarts (n_init=1 deliberately exposes init sensitivity).
    Deterministic algorithms (hierarchical) return 1.0 by construction."""
    if name.startswith("hier_"):
        return 1.0
    labs = [fit_model(name, X, k, seed=s, n_init=n_init).labels for s in seeds]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris))


def bootstrap_stability(X: np.ndarray, name: str, k: int, ref_labels: np.ndarray,
                        n_boot: int, frac: float, seed: int = 42) -> tuple[float, float]:
    """Fit on a random subsample, assign *all* points to its centroids, compare with the
    reference partition via ARI. Returns (mean, std)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        m = fit_model(name, X[idx], k, seed=int(rng.randint(1e6)), n_init=5)
        lab, _ = assign_to_centroids(X, m.centroids)
        aris.append(adjusted_rand_score(ref_labels, lab))
    return float(np.mean(aris)), float(np.std(aris))


def feature_sensitivity(build_X: Callable[[list[str]], np.ndarray], features: list[str], name: str,
                        k: int, ref_labels: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Leave-one-feature-out: ARI between the full-feature partition and the partition without
    that feature. Low ARI => the feature matters a lot to the segmentation."""
    rows = []
    for f in features:
        sub = [c for c in features if c != f]
        Xs = build_X(sub)
        lab = fit_model(name, Xs, k, seed=seed, n_init=10).labels
        rows.append({"dropped_feature": f, "ari_vs_full": adjusted_rand_score(ref_labels, lab)})
    return pd.DataFrame(rows).sort_values("ari_vs_full")


def assignment_stability(X: np.ndarray, ref_centroids: np.ndarray, ref_labels: np.ndarray, n_boot: int = 50,
                         frac: float = 0.8, seed: int = 42, n_init: int = 5) -> np.ndarray:
    """Per-buyer share of resampled K-Means refits that put the buyer in the SAME segment.

    Each refit is trained on a random ``frac`` subsample; its clusters are matched to the reference
    segments with the Hungarian algorithm on centroid distance, then every buyer is assigned by
    nearest centroid. 1.0 = always the same segment, ~1/K = no better than arbitrary.
    """
    from scipy.optimize import linear_sum_assignment
    rng = np.random.RandomState(seed)
    n, k = len(X), len(ref_centroids)
    agree = np.zeros(n)
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        m = fit_model("kmeans", X[idx], k, seed=int(rng.randint(1e6)), n_init=n_init)
        cost = ((m.centroids[:, None, :] - ref_centroids[None, :, :]) ** 2).sum(axis=2)
        r, c = linear_sum_assignment(cost)
        mapping = np.empty(k, dtype=int); mapping[r] = c
        lab, _ = assign_to_centroids(X, m.centroids)
        agree += (mapping[lab] == ref_labels)
    return agree / n_boot


def stability_by_closeness(ratio: np.ndarray, stability: np.ndarray, n_bins: int = 5) -> dict:
    """Calibration table: closeness ratio (quantile bins) -> mean assignment stability.

    Lets a NEW buyer (who has no resampling history) receive a data-backed stability estimate.
    """
    edges = np.unique(np.quantile(ratio, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(ratio, edges[1:-1]), 0, len(edges) - 2)
    return {"bin_edges": [float(e) for e in edges],
            "mean_stability": [float(stability[idx == b].mean()) for b in range(len(edges) - 1)],
            "n_buyers": [int((idx == b).sum()) for b in range(len(edges) - 1)],
            "overall_mean_stability": float(stability.mean())}
