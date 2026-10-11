"""Model-selection rule, its sensitivity to weights/constraints, and explicit K comparisons.

The rule is deliberately transparent. Every candidate gets a score in [0, 1]-ish units:

    score = w_boot * bootstrap_ARI            (higher is better, already in [0, 1])
          + w_seed * seed_ARI                 (higher is better, already in [0, 1])
          + w_gap  * minmax(silhouette - null silhouette)         (higher is better)
          + w_db   * minmax(-Davies-Bouldin)  (lower DB is better; min-max within feature set)
          + w_ch   * minmax(Calinski-Harabasz) (higher is better; min-max within feature set)

Constraints (candidates failing them are never scored): smallest segment >= ``min_share`` of
buyers, K inside ``k_range``, stability measured. Because the weights are a judgement call,
``selection_sensitivity`` re-runs the rule under several reasonable weight schemes and
constraint sets so the report can say whether the winner is robust or not.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .clustering import fit_kmeans

BASELINE_WEIGHTS: dict[str, float] = {"bootstrap": 0.4, "seed": 0.2, "gap": 0.2, "db": 0.1, "ch": 0.1}

WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "baseline (40/20/20/10/10)": BASELINE_WEIGHTS,
    "equal weights": {"bootstrap": 0.2, "seed": 0.2, "gap": 0.2, "db": 0.2, "ch": 0.2},
    "stability-heavy": {"bootstrap": 0.5, "seed": 0.3, "gap": 0.1, "db": 0.05, "ch": 0.05},
    "separation-heavy": {"bootstrap": 0.1, "seed": 0.1, "gap": 0.4, "db": 0.2, "ch": 0.2},
    "bootstrap-only": {"bootstrap": 1.0, "seed": 0.0, "gap": 0.0, "db": 0.0, "ch": 0.0},
    "internal-indices-only": {"bootstrap": 0.0, "seed": 0.0, "gap": 0.34, "db": 0.33, "ch": 0.33},
}

CONSTRAINT_SETS: dict[str, dict[str, Any]] = {
    "K 3-6, min segment >= 5%": {"k_range": (3, 6), "min_share": 0.05},
    "K 3-6, min segment >= 10%": {"k_range": (3, 6), "min_share": 0.10},
    "K 2-8, min segment >= 5%": {"k_range": (2, 8), "min_share": 0.05},
    "K 2-8, min segment >= 10%": {"k_range": (2, 8), "min_share": 0.10},
}

_REQUIRED = ["feature_set", "algorithm", "k", "silhouette", "silhouette_gap", "calinski_harabasz",
             "davies_bouldin", "min_cluster_share", "bootstrap_ari", "seed_ari"]


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-12) if rng > 0 else s * 0.0


def _validate(grid: pd.DataFrame, weights: dict[str, float]) -> None:
    missing = [c for c in _REQUIRED if c not in grid.columns]
    if missing:
        raise ValueError(f"Model grid is missing columns required for selection: {missing}")
    if set(weights) != set(BASELINE_WEIGHTS):
        raise ValueError(f"weights must have exactly the keys {sorted(BASELINE_WEIGHTS)}")
    if any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")


def score_candidates(grid: pd.DataFrame, weights: dict[str, float] | None = None,
                     k_range: tuple[int, int] = (3, 6), min_share: float = 0.05) -> pd.DataFrame:
    """Return the eligible candidates with a ``selection_score`` column, best first."""
    weights = weights or BASELINE_WEIGHTS
    _validate(grid, weights)
    c = grid[(grid["min_cluster_share"] >= min_share) & grid["k"].between(*k_range)
             & grid["bootstrap_ari"].notna() & grid["seed_ari"].notna()].copy()
    if c.empty:
        return c.assign(selection_score=pd.Series(dtype=float))
    c["gap_n"] = _minmax(c["silhouette_gap"])
    c["db_n"] = c.groupby("feature_set")["davies_bouldin"].transform(lambda s: _minmax(-s))
    c["ch_n"] = c.groupby("feature_set")["calinski_harabasz"].transform(_minmax)
    w = {k: v / sum(weights.values()) for k, v in weights.items()}  # weights are normalised to sum to 1
    c["selection_score"] = (w["bootstrap"] * c["bootstrap_ari"] + w["seed"] * c["seed_ari"]
                            + w["gap"] * c["gap_n"] + w["db"] * c["db_n"] + w["ch"] * c["ch_n"])
    return c.sort_values("selection_score", ascending=False)


def select_model(grid: pd.DataFrame, weights: dict[str, float] | None = None,
                 k_range: tuple[int, int] = (3, 6), min_share: float = 0.05) -> pd.Series:
    """Best eligible candidate under the rule. Also writes ``selection_score`` back into ``grid``."""
    c = score_candidates(grid, weights, k_range, min_share)
    if c.empty:
        raise ValueError("No model satisfies the selection constraints; relax k_range / min_share.")
    grid["selection_score"] = np.nan
    grid.loc[c.index, "selection_score"] = c["selection_score"]
    return c.iloc[0]


def selection_sensitivity(grid: pd.DataFrame, restrict_feature_set: str | None = None) -> pd.DataFrame:
    """Winner and runner-up under every (weight scheme x constraint set) combination."""
    g = grid if restrict_feature_set is None else grid[grid["feature_set"] == restrict_feature_set]
    rows = []
    for cname, cons in CONSTRAINT_SETS.items():
        for wname, w in WEIGHT_SCHEMES.items():
            c = score_candidates(g, w, cons["k_range"], cons["min_share"])
            if c.empty:
                rows.append({"constraints": cname, "weights": wname, "winner": "none eligible"})
                continue
            top = c.iloc[0]
            row = {"constraints": cname, "weights": wname,
                   "winner": f"{top.algorithm} K={int(top.k)} [{top.feature_set}]",
                   "winner_algorithm": top.algorithm, "winner_k": int(top.k), "winner_feature_set": top.feature_set,
                   "winner_score": float(top.selection_score)}
            if len(c) > 1:
                sec = c.iloc[1]
                row.update(runner_up=f"{sec.algorithm} K={int(sec.k)} [{sec.feature_set}]",
                           margin=float(top.selection_score - sec.selection_score))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_sensitivity(sens: pd.DataFrame, baseline_winner: str) -> dict[str, Any]:
    """How often does the baseline winner stay on top? Which alternatives win otherwise?"""
    valid = sens[sens["winner"] != "none eligible"]
    share = float((valid["winner"] == baseline_winner).mean()) if len(valid) else float("nan")
    return {"n_scenarios": int(len(valid)), "baseline_winner": baseline_winner,
            "share_of_scenarios_baseline_wins": share,
            "winner_counts": valid["winner"].value_counts().to_dict()}


def compare_k(grid: pd.DataFrame, feature_set: str, algorithm: str = "kmeans",
              ks: tuple[int, ...] = (4, 5)) -> pd.DataFrame:
    """Side-by-side metric table for specific K values (one column per K)."""
    sub = grid[(grid["feature_set"] == feature_set) & (grid["algorithm"] == algorithm) & grid["k"].isin(ks)]
    if sub.empty:
        raise ValueError(f"No rows for {feature_set}/{algorithm} with K in {ks}")
    cols = [c for c in ["inertia", "silhouette", "null_silhouette", "silhouette_gap", "davies_bouldin",
                        "calinski_harabasz", "min_cluster_share", "max_cluster_share", "seed_ari",
                        "bootstrap_ari", "bootstrap_ari_sd"] if c in sub.columns]
    out = sub.set_index("k")[cols].T
    out.columns = [f"K={int(k)}" for k in out.columns]
    return out


def partition_agreement(X: np.ndarray, k_a: int = 4, k_b: int = 5, seed: int = 42,
                        n_init: int = 20) -> dict[str, Any]:
    """How different are the K=a and K=b partitions of the same data?

    Returns ARI and, for each K=a segment, the share of its members landing in its single most
    common K=b segment (1.0 = the K=a segment is never split by moving to K=b).
    """
    a = fit_kmeans(X, k_a, seed=seed, n_init=n_init).labels
    b = fit_kmeans(X, k_b, seed=seed, n_init=n_init).labels
    ct = pd.crosstab(pd.Series(a, name=f"K={k_a}"), pd.Series(b, name=f"K={k_b}"))
    kept = (ct.max(axis=1) / ct.sum(axis=1)).round(3).to_dict()
    return {"ari": float(adjusted_rand_score(a, b)), "crosstab": ct,
            "share_in_dominant_child": {int(k): float(v) for k, v in kept.items()},
            "new_segment_sizes_kb": np.bincount(b, minlength=k_b).tolist()}
