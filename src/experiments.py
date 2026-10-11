"""Model-selection experiments: feature sets x algorithms x K, with a null baseline and stability."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from .clustering import fit_model
from .evaluation import (bootstrap_stability, internal_metrics, seed_stability, smallest_cluster_share)
from .feature_engineering import make_full_pipeline

ALGOS = ["kmeans", "hier_ward", "hier_complete", "hier_average", "gmm"]


def null_silhouette(X: np.ndarray, name: str, k: int, n_rep: int = 5, seed: int = 0) -> float:
    """Silhouette on data whose columns are independently shuffled.

    This keeps every marginal distribution but destroys joint structure, so it shows what
    'cluster quality' looks like when there is *no* multivariate structure at all.
    """
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_rep):
        Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        lab = fit_model(name, Xn, k, seed=42, n_init=5).labels
        vals.append(internal_metrics(Xn, lab)["silhouette"])
    return float(np.nanmean(vals))


def encode(df: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    pipe = make_full_pipeline(spec["numeric"], spec["binary"], spec["categorical"],
                              cfg["features"]["log_transform"], spec["winsor"])
    return pipe.fit_transform(df)


def run_grid(df: pd.DataFrame, feature_sets: dict[str, dict[str, Any]], cfg: dict[str, Any],
             with_stability: bool = True) -> pd.DataFrame:
    """Evaluate every (feature set, algorithm, K). Stability is computed for K-Means, Ward, GMM.

    ``stability.n_seeds`` and ``stability.n_bootstrap`` from config.yaml are used exactly as
    written (an earlier version silently halved them, so reported stability did not match the config).
    Inertia is recorded only for K-Means, the one algorithm that defines it (NaN otherwise).
    ``fit_seconds`` is the wall-clock time of the single reference fit.
    """
    cl, st = cfg["clustering"], cfg["stability"]
    rows = []
    for fs_name, spec in feature_sets.items():
        X = encode(df, spec, cfg)
        for algo in ALGOS:
            for k in cl["k_range"]:
                t0 = time.perf_counter()
                m = fit_model(algo, X, k, seed=cfg["random_state"], n_init=cl["n_init"])
                fit_s = time.perf_counter() - t0
                shares = np.bincount(m.labels, minlength=k) / len(m.labels)
                r = {"feature_set": fs_name, "algorithm": algo, "k": k, **internal_metrics(X, m.labels),
                     "inertia": float(m.estimator.inertia_) if algo == "kmeans" else np.nan,
                     "min_cluster_share": float(shares.min()), "max_cluster_share": float(shares.max()),
                     "fit_seconds": round(fit_s, 4)}
                r["null_silhouette"] = null_silhouette(X, algo, k)
                r["silhouette_gap"] = r["silhouette"] - r["null_silhouette"]
                if with_stability and algo in ("kmeans", "hier_ward", "gmm"):
                    r["seed_ari"] = seed_stability(X, algo, k, list(range(st["n_seeds"])))
                    mu, sd = bootstrap_stability(X, algo, k, m.labels, st["n_bootstrap"],
                                                 st["bootstrap_fraction"], seed=cfg["random_state"])
                    r["bootstrap_ari"], r["bootstrap_ari_sd"] = mu, sd
                rows.append(r)
    return pd.DataFrame(rows)


def pca_comparison(df: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any], k: int,
                   variance: float = 0.90) -> dict[str, float]:
    """Does clustering in a PCA-reduced space beat the original engineered space?"""
    X = encode(df, spec, cfg)
    pca = PCA(n_components=variance, random_state=cfg["random_state"]).fit(X)
    Z = pca.transform(X)
    a = fit_model("kmeans", X, k, seed=cfg["random_state"], n_init=20)
    b = fit_model("kmeans", Z, k, seed=cfg["random_state"], n_init=20)
    ma, mb = internal_metrics(X, a.labels), internal_metrics(Z, b.labels)
    return {"n_components": int(pca.n_components_), "variance_kept": float(pca.explained_variance_ratio_.sum()),
            "sil_original": ma["silhouette"], "sil_pca": mb["silhouette"],
            "db_original": ma["davies_bouldin"], "db_pca": mb["davies_bouldin"],
            "ari_original_vs_pca": float(adjusted_rand_score(a.labels, b.labels))}
