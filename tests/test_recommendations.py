"""Recommendation layer + stability-aware scorer (needs models/ from `python run_pipeline.py`)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import load_buyer_table, load_config  # noqa: E402
from src.predict import SegmentScorer  # noqa: E402
from src.recommendations import MIN_ASSIGNMENT_STABILITY, structured_recommendations  # noqa: E402

REQUIRED = {"id", "basis", "action", "evidence", "customer_benefit", "business_benefit", "confidence", "rule",
            "limitations", "human_review"}
FORBIDDEN = re.compile(r"\b(gender|country|region|nationality|income|wealth|net worth|high-net|age)\b", re.I)


@pytest.fixture(scope="module")
def world():
    cfg = load_config(ROOT / "config.yaml")
    df, _ = load_buyer_table(cfg, ROOT)
    sc = SegmentScorer(ROOT / "models")
    return df, sc, sc.score(df)


def _recs(world, i):
    df, sc, out = world
    seg = str(int(out["segment"].iloc[i]))
    return structured_recommendations(df.iloc[i], sc.profiles["segments"][seg], sc.profiles["overall"],
                                      float(out["est_assignment_stability"].iloc[i]))


def test_every_recommendation_is_complete_and_labelled(world):
    for i in range(0, len(world[0]), 7):
        for r in _recs(world, i):
            assert REQUIRED <= set(r) and r["basis"] in {"observed", "ML-derived inference"}
            assert r["evidence"] and all(isinstance(e, str) and e for e in r["evidence"])


def test_no_sensitive_attribute_is_used_as_evidence(world):
    for i in range(0, len(world[0]), 5):
        for r in _recs(world, i):
            text = " ".join([r["action"], *r["evidence"], r["rule"]])
            assert not FORBIDDEN.search(text), (r["id"], text)


def test_low_satisfaction_gets_review_and_no_promotion(world):
    df, _, _ = world
    idx = np.flatnonzero(df["satisfaction_score"].to_numpy() <= 2)[:50]
    for i in idx:
        recs = _recs(world, int(i))
        assert recs[0]["id"] == "OBS-LOW-SAT" and "person should review" in recs[0]["human_review"]
        assert not any(r["id"].startswith("SEG-") or r["id"] == "OBS-INVEST" for r in recs)


def test_ambiguous_assignment_asks_instead_of_personalising(world):
    df, _, out = world
    idx = np.flatnonzero((out["est_assignment_stability"].to_numpy() < MIN_ASSIGNMENT_STABILITY) & (df["satisfaction_score"].to_numpy() > 2))
    assert len(idx) > 0
    for i in idx[:50]:
        ids = [r["id"] for r in _recs(world, int(i))]
        assert "AMBIG-ASSIGN" in ids and not any(x.startswith("SEG-") for x in ids)


def test_stability_estimates_are_valid_and_scoring_is_reproducible(world):
    df, sc, out = world
    assert out["est_assignment_stability"].between(0, 1).all()
    saved = pd.read_csv(ROOT / "reports" / "segment_assignments.csv")
    assert (out["segment"].to_numpy() == saved["segment"].to_numpy()).all()
    assert saved["assignment_stability"].between(0, 1).all()
    # buyers closest to a segment boundary are, on average, the least stable
    assert saved.groupby(pd.qcut(saved["closeness_ratio"], 5, duplicates="drop"), observed=True)["assignment_stability"].mean().is_monotonic_decreasing


def test_new_customer_scoring_returns_stability(world):
    _, sc, _ = world
    res = sc.score_new_customer(dict(purchase_count=4, average_purchase_price_usd=340000, average_price_per_sqft_usd=300,
                                     office_share=0.0, purchase_span_months=12))
    assert 0 <= res["est_assignment_stability"] <= 1 and res["segment_name"]
