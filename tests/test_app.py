"""Streamlit smoke test using Streamlit's own AppTest harness (skipped if streamlit is not installed).

Needs models/ and reports/ from `python run_pipeline.py`. Every page must render without raising.
"""
from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Dashboard", "Buyer Segmentation", "Customer Intelligence", "New Customer Prediction", "Investment Behavior",
         "Geographic Intelligence", "Customer Experience", "Model Performance", "Methodology"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    assert not at.error, f"{page}: {[e.value for e in at.error]}"
