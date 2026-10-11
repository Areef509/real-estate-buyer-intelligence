"""Data loading, auditing and cross-table integrity checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_raw(cfg: dict[str, Any], root: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Load the three cleaned CSVs. Raises a clear error if a file is missing."""
    root = Path(root)
    out: dict[str, pd.DataFrame] = {}
    for key, fname in cfg["files"].items():
        fp = root / cfg["paths"]["data_dir"] / fname
        if not fp.exists():
            raise FileNotFoundError(f"Expected data file not found: {fp}")
        out[key] = pd.read_csv(fp)
    return out


def audit_dataframe(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Rows, columns, missing cells and duplicate rows for one table."""
    return {
        "file": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def audit_all(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit every table."""
    return pd.DataFrame([audit_dataframe(df, k) for k, df in tables.items()])


def integrity_checks(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Cross-table checks: orphan references, buyer coverage, meaning of nulls."""
    clients, props, buyers = tables["clients"], tables["properties"], tables["buyers"]
    sold = props[props["listing_status"] == "Sold"]
    return {
        "unique_clients": int(clients["client_id"].nunique()),
        "properties_total": int(len(props)),
        "properties_sold": int(len(sold)),
        "properties_available_no_client": int((props["listing_status"] == "Available").sum()),
        "missing_client_ref_is_exactly_available_listings": bool(
            props["client_ref"].isna().equals(props["listing_status"].eq("Available"))
        ),
        "orphan_client_refs": int((~sold["client_ref"].isin(clients["client_id"])).sum()),
        "clients_without_purchase": int((buyers["purchase_count"] == 0).sum()),
        "min_purchases_per_client": int(buyers["purchase_count"].min()),
        "purchase_date_range": [str(sold["transaction_date"].min()), str(sold["transaction_date"].max())],
    }


def load_buyer_table(cfg: dict[str, Any], root: str | Path = ".", tables: dict[str, pd.DataFrame] | None = None):
    """Return (modelling table, reconciliation report) with the buyer table REBUILT from source.

    The provided ``buyer_features_model_ready.csv`` is kept untouched and used only for
    reconciliation: its ``office_share`` column does not match the source transactions for ~45% of
    buyers, so the pipeline derives every feature from clients + sold properties instead.
    """
    from .feature_engineering import build_buyer_table, rebuild_buyer_table, reconcile_buyer_tables
    tables = tables or load_raw(cfg, root)
    rebuilt = rebuild_buyer_table(tables["clients"], tables["properties"], cfg["reference_date"])
    recon = reconcile_buyer_tables(rebuilt, tables["buyers"])
    return build_buyer_table(rebuilt, cfg["reference_date"]), recon
