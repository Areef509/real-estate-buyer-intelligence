"""Tests for the model-selection rule, its sensitivity analysis and K comparison."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection import (BASELINE_WEIGHTS, compare_k, partition_agreement, score_candidates,  # noqa: E402
                           select_model, selection_sensitivity, summarise_sensitivity)


def _grid():
    rows = []
    for k, (sil, gap, db, ch, mn, seed, boot) in {
        3: (0.15, 0.010, 1.7, 380, 0.30, 0.90, 0.80), 4: (0.16, 0.012, 1.6, 400, 0.23, 0.60, 0.85),
        5: (0.17, 0.015, 1.4, 390, 0.06, 0.60, 0.72), 6: (0.17, 0.014, 1.5, 370, 0.03, 0.55, 0.70)}.items():
        rows.append(dict(feature_set="f", algorithm="kmeans", k=k, silhouette=sil, silhouette_gap=gap, davies_bouldin=db,
                         calinski_harabasz=ch, min_cluster_share=mn, max_cluster_share=0.4, seed_ari=seed,
                         bootstrap_ari=boot, bootstrap_ari_sd=0.1, inertia=1000.0 / k))
    return pd.DataFrame(rows)


def test_constraints_are_enforced():
    c = score_candidates(_grid(), min_share=0.05)
    assert 6 not in set(c["k"])                       # 3% smallest segment is ineligible
    assert set(score_candidates(_grid(), k_range=(4, 5))["k"]) == {4, 5}
    assert score_candidates(_grid(), min_share=0.9).empty


def test_select_model_returns_best_scored_candidate_and_writes_scores():
    g = _grid(); best = select_model(g)
    assert best["selection_score"] == g["selection_score"].max()
    assert g.loc[g["k"] == 6, "selection_score"].isna().all()


def test_weights_are_validated_and_normalised():
    with pytest.raises(ValueError):
        score_candidates(_grid(), {"bootstrap": 1.0})
    with pytest.raises(ValueError):
        score_candidates(_grid(), {k: 0.0 for k in BASELINE_WEIGHTS})
    doubled = {k: 2 * v for k, v in BASELINE_WEIGHTS.items()}
    a, b = score_candidates(_grid()), score_candidates(_grid(), doubled)
    np.testing.assert_allclose(a["selection_score"].values, b["selection_score"].values)


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="missing columns"):
        select_model(_grid().drop(columns=["bootstrap_ari"]))


def test_no_eligible_model_raises():
    with pytest.raises(ValueError):
        select_model(_grid(), min_share=0.9)


def test_sensitivity_covers_all_scenarios_and_summarises():
    s = selection_sensitivity(_grid())
    assert len(s) == 24 and {"constraints", "weights", "winner"} <= set(s.columns)
    summ = summarise_sensitivity(s, "kmeans K=4 [f]")
    assert 0.0 <= summ["share_of_scenarios_baseline_wins"] <= 1.0
    assert sum(summ["winner_counts"].values()) == summ["n_scenarios"]


def test_compare_k_table():
    t = compare_k(_grid(), "f", "kmeans", (4, 5))
    assert list(t.columns) == ["K=4", "K=5"] and "inertia" in t.index
    with pytest.raises(ValueError):
        compare_k(_grid(), "missing", "kmeans")


def test_partition_agreement_on_blobs():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(c, 0.3, (60, 2)) for c in ([0, 0], [6, 0], [0, 6], [6, 6])])
    r = partition_agreement(X, 4, 5, n_init=5)
    assert 0.0 <= r["ari"] <= 1.0 and sum(r["new_segment_sizes_kb"]) == len(X)
    assert all(0 < v <= 1 for v in r["share_in_dominant_child"].values())
