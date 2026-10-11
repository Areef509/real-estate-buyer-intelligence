"""Data-quality and feature-rebuild tests (run against the real data files and small synthetic frames)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import load_config, load_raw  # noqa: E402
from src.feature_engineering import rebuild_buyer_table, reconcile_buyer_tables  # noqa: E402


@pytest.fixture(scope="module")
def data():
    cfg = load_config(ROOT / "config.yaml")
    return cfg, load_raw(cfg, ROOT)


def _tiny():
    clients = pd.DataFrame({"client_id": ["C1", "C2"], "client_type": ["Individual", "Company"], "gender": ["F", "M"],
                            "country": ["A", "B"], "region": ["r1", "r2"], "date_of_birth": ["1990-06-15", "1980-01-01"],
                            "acquisition_purpose": ["Home", "Investment"], "loan_applied": ["No", "Yes"],
                            "referral_channel": ["Web", "Agent"], "satisfaction_score": [4, 5]})
    props = pd.DataFrame({"listing_id": [1, 2, 3, 4], "tower_number": [1, 1, 2, 2],
                          "transaction_date": ["2024-01-01", "2024-03-01", "2024-02-01", None],
                          "unit_category": ["Apartment", "Office", "Apartment", "Apartment"],
                          "floor_area_sqft": [500.0, 1000.0, 600.0, 700.0], "sale_price": [100.0, 400.0, 150.0, 999.0],
                          "listing_status": ["Sold", "Sold", "Sold", "Available"],
                          "client_ref": ["C1", "C1", "C2", None]})
    return clients, props


def test_source_schema_and_keys(data):
    _, t = data
    assert not t["clients"]["client_id"].duplicated().any()
    assert not t["properties"]["listing_id"].duplicated().any()
    sold = t["properties"][t["properties"]["listing_status"] == "Sold"]
    assert sold["client_ref"].notna().all()
    assert t["properties"].loc[t["properties"]["listing_status"] != "Sold", "client_ref"].isna().all()
    assert sold["client_ref"].isin(t["clients"]["client_id"]).all()


def test_values_are_valid_and_units_consistent(data):
    _, t = data
    sold = t["properties"][t["properties"]["listing_status"] == "Sold"]
    assert (sold["sale_price"] > 0).all() and (sold["floor_area_sqft"] > 0).all()
    np.testing.assert_allclose(sold["sale_price"] / sold["floor_area_sqft"], sold["price_per_sqft"], rtol=1e-9)
    pd.to_datetime(sold["transaction_date"], errors="raise")
    assert set(t["properties"]["unit_category"]) <= {"Apartment", "Office"}


def test_rebuild_matches_provided_table_except_documented_defect(data):
    cfg, t = data
    rebuilt = rebuild_buyer_table(t["clients"], t["properties"], cfg["reference_date"])
    rec = reconcile_buyer_tables(rebuilt, t["buyers"])
    bad = set(rec.loc[rec["status"] != "match", "column"])
    assert bad <= {"office_share"}, f"unexpected reconciliation mismatches: {bad}"


def test_rebuilt_shares_are_consistent(data):
    cfg, t = data
    r = rebuild_buyer_table(t["clients"], t["properties"], cfg["reference_date"])
    np.testing.assert_allclose(r["office_share"] + r["apartment_share"], 1.0)
    np.testing.assert_allclose(r["office_share"], r["office_purchase_count"] / r["purchase_count"])
    assert r["client_id"].is_unique and len(r) == t["clients"].shape[0]
    assert "first_name" not in r.columns and "last_name" not in r.columns  # no needless PII downstream


def test_rebuild_excludes_available_listings_and_computes_values():
    clients, props = _tiny()
    r = rebuild_buyer_table(clients, props, "2025-12-31").set_index("client_id")
    assert r.loc["C1", "purchase_count"] == 2 and r.loc["C1", "total_spend_usd"] == 500.0  # 999 'Available' excluded
    assert r.loc["C1", "office_share"] == 0.5 and r.loc["C2", "office_share"] == 0.0
    assert r.loc["C1", "average_price_per_sqft_usd"] == pytest.approx((100 / 500 + 400 / 1000) / 2)
    assert r.loc["C1", "age_at_2025_12_31"] == 35


def test_rebuild_rejects_bad_inputs():
    clients, props = _tiny()
    with pytest.raises(ValueError):
        rebuild_buyer_table(pd.concat([clients, clients]), props)          # duplicate client ids
    bad = props.copy(); bad.loc[0, "client_ref"] = "C99"
    with pytest.raises(ValueError):
        rebuild_buyer_table(clients, bad)                                  # orphan reference
    bad = props.copy(); bad.loc[0, "sale_price"] = -5.0
    with pytest.raises(ValueError):
        rebuild_buyer_table(clients, bad)                                  # non-positive price
    with pytest.raises(ValueError):
        rebuild_buyer_table(clients.drop(columns=["gender"]), props)       # schema mismatch
