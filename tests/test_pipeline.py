"""Run with:  python -m pytest tests -q   (requires `python run_pipeline.py` to have produced models/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import fit_kmeans  # noqa: E402
from src.data_processing import integrity_checks, load_buyer_table, load_config, load_raw  # noqa: E402
from src.feature_engineering import Winsorizer, build_buyer_table  # noqa: E402
from src.predict import SegmentScorer, engineer_new_customer  # noqa: E402


@pytest.fixture(scope="module")
def data():
    cfg = load_config(ROOT / "config.yaml")
    t = load_raw(cfg, ROOT)
    return cfg, t, load_buyer_table(cfg, ROOT, t)[0]


def test_data_integrity(data):
    _, t, _ = data
    chk = integrity_checks(t)
    assert chk["orphan_client_refs"] == 0
    assert chk["missing_client_ref_is_exactly_available_listings"]


def test_saved_assignments_reproduced_by_scorer(data):
    _, _, df = data
    sc = SegmentScorer(ROOT / "models")
    saved = pd.read_csv(ROOT / "reports/segment_assignments.csv")
    assert (sc.score(df)["segment"].values == saved["segment"].values).all()


def test_kmeans_is_deterministic(data):
    X = np.random.RandomState(0).normal(size=(300, 4))
    assert (fit_kmeans(X, 3, seed=42).labels == fit_kmeans(X, 3, seed=42).labels).all()


def test_new_customer_requires_fields():
    with pytest.raises(ValueError):
        engineer_new_customer({"purchase_count": 3})


def test_new_customer_scoring_and_range_warning():
    sc = SegmentScorer(ROOT / "models")
    ok = sc.score_new_customer(dict(purchase_count=4, average_purchase_price_usd=340000, average_price_per_sqft_usd=300,
                                    office_share=0.0, purchase_span_months=12))
    assert ok["segment_name"] and 0 <= ok["confidence"] <= 1
    odd = sc.score_new_customer(dict(purchase_count=1, average_purchase_price_usd=340000, average_price_per_sqft_usd=300,
                                     office_share=0.0, purchase_span_months=12))
    assert odd["low_fit_warning"]


def test_winsorizer_uses_training_bounds():
    w = Winsorizer(columns=["x"]).fit(pd.DataFrame({"x": range(100)}))
    out = w.transform(pd.DataFrame({"x": [-1000, 50, 1000]}))
    assert out["x"].min() >= 0.99 - 1e-9 and out["x"].max() <= 98.01


def test_segment_names_unique_and_no_forbidden_fields():
    sc = SegmentScorer(ROOT / "models")
    names = [p["name"] for p in sc.profiles["segments"].values()]
    assert len(names) == len(set(names))
    blob = str(sc.profiles).lower()
    assert "income" not in blob and "net worth" not in blob
