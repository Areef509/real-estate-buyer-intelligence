"""Safe handling of invalid / unusual scoring inputs (needs models/ from `python run_pipeline.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import SegmentScorer, validate_new_customer  # noqa: E402

BASE = dict(purchase_count=4, average_purchase_price_usd=340000, average_price_per_sqft_usd=300,
            office_share=0.1, purchase_span_months=12)


@pytest.fixture(scope="module")
def scorer():
    return SegmentScorer(ROOT / "models")


@pytest.mark.parametrize("field,value", [("purchase_count", "four"), ("average_purchase_price_usd", -5),
                                         ("average_purchase_price_usd", 0), ("average_price_per_sqft_usd", float("nan")),
                                         ("office_share", 1.7), ("office_share", -0.1), ("purchase_span_months", float("inf")),
                                         ("purchase_count", None), ("purchase_count", 0)])
def test_invalid_values_are_rejected_with_the_field_name(scorer, field, value):
    with pytest.raises(ValueError, match=field):
        scorer.score_new_customer({**BASE, field: value})


def test_all_problems_reported_together():
    with pytest.raises(ValueError) as e:
        validate_new_customer({"purchase_count": "x", "office_share": 3})
    msg = str(e.value)
    assert "purchase_count" in msg and "office_share" in msg and "average_purchase_price_usd" in msg


def test_missing_required_field_is_rejected(scorer):
    with pytest.raises(ValueError, match="office_share"):
        scorer.score_new_customer({k: v for k, v in BASE.items() if k != "office_share"})


def test_extra_fields_and_unseen_categories_are_ignored(scorer):
    a = scorer.score_new_customer(BASE)
    b = scorer.score_new_customer({**BASE, "favourite_colour": "red", "acquisition_purpose": "Flip", "client_type": "Trust"})
    assert a["segment"] == b["segment"]


def test_numeric_strings_are_coerced(scorer):
    assert scorer.score_new_customer({**BASE, "purchase_count": "4"})["segment"] == scorer.score_new_customer(BASE)["segment"]


def test_out_of_distribution_input_is_flagged_not_hidden(scorer):
    assert scorer.score_new_customer({**BASE, "average_purchase_price_usd": 5e9})["low_fit_warning"]
    assert scorer.score_new_customer({**BASE, "purchase_count": 1})["low_fit_warning"]


def test_batch_scoring_rejects_nan_and_missing_columns(scorer):
    df = pd.DataFrame([BASE])
    bad = df.copy(); bad.loc[0, "purchase_span_months"] = np.nan
    with pytest.raises(ValueError):
        scorer.score(bad)
    with pytest.raises(ValueError, match="missing model features"):
        scorer.score(df.drop(columns=["office_share"]))
