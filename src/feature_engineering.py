"""Feature engineering, redundancy analysis and the reusable preprocessing pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Raw input columns a *new customer* needs to be scored (see predict_segment).
NEW_CUSTOMER_FIELDS = [
    "age", "satisfaction_score", "purchase_count", "average_purchase_price_usd",
    "average_area_sqft", "average_price_per_sqft_usd", "office_share",
    "distinct_towers_purchased", "purchase_span_months", "acquisition_purpose",
    "loan_applied", "client_type", "country", "referral_channel",
]


def build_buyer_table(buyers: pd.DataFrame, reference_date: str = "2025-12-31") -> pd.DataFrame:
    """Derive the engineered columns from buyer_features_model_ready.csv.

    Every derived column is a deterministic function of observed data; nothing
    such as income, wealth or budget is created.
    """
    df = buyers.copy()
    df = df.rename(columns={"age_at_2025_12_31": "age"})
    first = pd.to_datetime(df["first_purchase_date"])
    last = pd.to_datetime(df["last_purchase_date"])
    # months between first and last purchase, inclusive (>= 1)
    df["purchase_span_months"] = ((last.dt.year - first.dt.year) * 12 + (last.dt.month - first.dt.month) + 1).astype(int)
    # purchases per active month (purchase intensity / frequency)
    df["purchase_intensity"] = df["purchase_count"] / df["purchase_span_months"]
    # share of purchases that landed in a distinct tower: 1 = fully spread, <1 = repeat towers
    df["tower_diversification"] = df["distinct_towers_purchased"] / df["purchase_count"]
    df["is_international"] = (df["country"] != "USA").astype(int)
    df["acquisition_purpose"] = df["acquisition_purpose"].astype(str)
    return df


def correlation_report(df: pd.DataFrame, cols: list[str], threshold: float = 0.85) -> pd.DataFrame:
    """Pairs of candidate features whose absolute Pearson correlation >= threshold."""
    corr = df[cols].corr().abs()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": round(float(corr.loc[a, b]), 3)})
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "abs_corr"])


def low_variance_report(df: pd.DataFrame, cols: list[str], max_top_share: float = 0.90) -> pd.DataFrame:
    """Flag near-constant columns.

    Min-max variance is misleading for skewed columns (one outlier shrinks it), so
    the primary test is the share of rows taken by the single most common value.
    """
    rows = []
    for c in cols:
        top = float(df[c].value_counts(normalize=True).iloc[0])
        rows.append({"feature": c, "n_unique": int(df[c].nunique()), "top_value_share": round(top, 3),
                     "std": round(float(df[c].std()), 4), "flag_near_constant": top > max_top_share})
    return pd.DataFrame(rows).sort_values("top_value_share", ascending=False)


def _log1p(x: Any) -> Any:
    return np.log1p(x)


def make_preprocessor(numeric: list[str], binary: list[str], categorical: list[str],
                      log_cols: list[str] | None = None) -> ColumnTransformer:
    """Mixed-type preprocessing.

    * numeric  -> (optional log1p) -> StandardScaler
    * binary   -> passed through untouched (already 0/1; scaling would distort them)
    * categorical -> OneHotEncoder(handle_unknown='ignore'), so unseen categories
      in a new customer do not crash scoring.
    """
    log_cols = [c for c in (log_cols or []) if c in numeric]
    plain = [c for c in numeric if c not in log_cols]
    transformers: list[tuple[str, Any, list[str]]] = []
    if log_cols:
        transformers.append(("num_log", Pipeline([("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                                                  ("scale", StandardScaler())]), log_cols))
    if plain:
        transformers.append(("num", StandardScaler(), plain))
    if binary:
        transformers.append(("bin", "passthrough", binary))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip selected columns to training-set quantiles.

    Fitted bounds are stored so a *new* customer is clipped with the same limits
    (no refitting at scoring time). Works on DataFrames.
    """

    def __init__(self, columns: list[str] | None = None, lower: float = 0.01, upper: float = 0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Winsorizer":
        cols = self.columns or []
        self.bounds_ = {c: (float(X[c].quantile(self.lower)), float(X[c].quantile(self.upper))) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return np.array(input_features)


def make_full_pipeline(numeric: list[str], binary: list[str], categorical: list[str],
                       log_cols: list[str], winsor_cols: list[str]) -> Pipeline:
    """Winsorise -> (log/scale, passthrough binaries, one-hot categoricals)."""
    return Pipeline([
        ("winsor", Winsorizer(columns=winsor_cols)),
        ("encode", make_preprocessor(numeric, binary, categorical, log_cols)),
    ])


# --------------------------------------------------------------------------------------------
# Rebuild the buyer-level table directly from the two source tables (single source of truth).
# --------------------------------------------------------------------------------------------
def rebuild_buyer_table(clients: pd.DataFrame, properties: pd.DataFrame,
                        reference_date: str = "2025-12-31") -> pd.DataFrame:
    """One row per client, computed only from clients + SOLD transactions.

    * Available (unsold) listings have no buyer and are excluded from every aggregate.
    * ``office_share`` / ``apartment_share`` are shares of the buyer's *purchases* (unit counts),
      so they always sum to 1.
    * ``average_price_per_sqft_usd`` is the mean of per-transaction price/area ratios.
    * Currency is assumed to be USD and area square feet, as in the column names of the source
      files (the source tables carry no explicit unit metadata, so this cannot be verified from data).
    * Personal name columns are dropped; they are not needed for modelling.
    """
    need_c = ["client_id", "client_type", "gender", "country", "region", "date_of_birth", "acquisition_purpose",
              "loan_applied", "referral_channel", "satisfaction_score"]
    need_p = ["listing_id", "tower_number", "transaction_date", "unit_category", "floor_area_sqft", "sale_price",
              "listing_status", "client_ref"]
    for name, need, df in (("clients", need_c, clients), ("properties", need_p, properties)):
        miss = set(need) - set(df.columns)
        if miss:
            raise ValueError(f"{name} table is missing required columns: {sorted(miss)}")
    if clients["client_id"].duplicated().any():
        raise ValueError("Duplicate client_id values in clients table")

    sold = properties[properties["listing_status"] == "Sold"].copy()
    if sold["client_ref"].isna().any():
        raise ValueError("Sold listings without a client_ref")
    orphans = ~sold["client_ref"].isin(clients["client_id"])
    if orphans.any():
        raise ValueError(f"{int(orphans.sum())} sold listings reference unknown clients")
    sold["sale_price"] = pd.to_numeric(sold["sale_price"], errors="raise")
    sold["floor_area_sqft"] = pd.to_numeric(sold["floor_area_sqft"], errors="raise")
    if (sold["sale_price"] <= 0).any() or (sold["floor_area_sqft"] <= 0).any():
        raise ValueError("Non-positive sale_price or floor_area_sqft among sold listings")
    sold["transaction_date"] = pd.to_datetime(sold["transaction_date"], errors="raise")
    sold["_ppsf"] = sold["sale_price"] / sold["floor_area_sqft"]
    sold["_office"] = (sold["unit_category"] == "Office").astype(int)
    sold["_apartment"] = (sold["unit_category"] == "Apartment").astype(int)

    g = sold.groupby("client_ref")
    agg = pd.DataFrame({
        "purchase_count": g["listing_id"].count(),
        "total_spend_usd": g["sale_price"].sum(),
        "average_purchase_price_usd": g["sale_price"].mean(),
        "median_purchase_price_usd": g["sale_price"].median(),
        "average_area_sqft": g["floor_area_sqft"].mean(),
        "total_area_sqft": g["floor_area_sqft"].sum(),
        "average_price_per_sqft_usd": g["_ppsf"].mean(),
        "apartment_purchase_count": g["_apartment"].sum(),
        "office_purchase_count": g["_office"].sum(),
        "distinct_towers_purchased": g["tower_number"].nunique(),
        "distinct_property_categories": g["unit_category"].nunique(),
        "first_purchase_date": g["transaction_date"].min().dt.strftime("%Y-%m-%d"),
        "last_purchase_date": g["transaction_date"].max().dt.strftime("%Y-%m-%d"),
    })
    agg["apartment_share"] = agg["apartment_purchase_count"] / agg["purchase_count"]
    agg["office_share"] = agg["office_purchase_count"] / agg["purchase_count"]

    c = clients[need_c].copy()
    ref = pd.Timestamp(reference_date)
    dob = pd.to_datetime(c["date_of_birth"], errors="raise")
    c["age_at_2025_12_31"] = ((ref - dob).dt.days / 365.2425).astype(int)  # column name kept for compatibility
    out = c.merge(agg, left_on="client_id", right_index=True, how="inner", validate="one_to_one")
    out["is_investment_purpose"] = (out["acquisition_purpose"] == "Investment").astype(int)
    out["is_loan_applied"] = (out["loan_applied"] == "Yes").astype(int)
    out["is_company"] = (out["client_type"] == "Company").astype(int)
    return out.reset_index(drop=True)


def reconcile_buyer_tables(rebuilt: pd.DataFrame, provided: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Column-by-column comparison of the rebuilt table against the provided model-ready CSV."""
    a = rebuilt.set_index("client_id").sort_index()
    b = provided.set_index("client_id").sort_index()
    if not a.index.equals(b.index):
        return pd.DataFrame([{"column": "client_id", "n_mismatch": -1, "max_abs_diff": float("nan"),
                              "status": "different client sets"}])
    rows = []
    for col in [c for c in a.columns if c in b.columns]:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            diff = (a[col].astype(float) - b[col].astype(float)).abs()
            bad, mx = int((diff > tol).sum()), float(diff.max())
        else:
            bad, mx = int((a[col].astype(str) != b[col].astype(str)).sum()), float("nan")
        rows.append({"column": col, "n_mismatch": bad, "max_abs_diff": mx, "status": "match" if bad == 0 else "MISMATCH"})
    return pd.DataFrame(rows)
