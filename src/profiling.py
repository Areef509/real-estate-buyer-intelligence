"""Cluster profiling, significance testing and data-driven naming."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC_PROFILE = ["age", "satisfaction_score", "purchase_count", "total_spend_usd", "average_purchase_price_usd",
                   "average_area_sqft", "average_price_per_sqft_usd", "office_share", "purchase_span_months",
                   "purchase_intensity", "distinct_towers_purchased"]
BINARY_PROFILE = ["is_investment_purpose", "is_loan_applied", "is_company", "is_international"]

# trait -> (adjective if high, adjective if low). Names are only built from traits that
# are statistically and practically distinguishing for that cluster.
DESCRIPTORS: dict[str, tuple[str, str]] = {
    "purchase_count": ("High-Volume", "Lower-Volume"),
    "average_purchase_price_usd": ("Premium-Ticket", "Value-Ticket"),
    "average_price_per_sqft_usd": ("Premium-Rate", "Lower-Rate"),
    "office_share": ("Commercial-Leaning", "Residential-Focused"),
    "purchase_span_months": ("Long-Horizon", "Fast-Cycle"),
    "age": ("Senior", "Younger"),
    "satisfaction_score": ("Highly-Satisfied", "Low-Satisfaction"),
    "is_investment_purpose": ("Investment-Oriented", "Home-Oriented"),
    "is_loan_applied": ("Financing-Reliant", "Cash-Leaning"),
    "is_company": ("Corporate", "Individual"),
}
LABELS = {
    "age": "Age", "satisfaction_score": "Satisfaction", "purchase_count": "Purchase count",
    "total_spend_usd": "Total spend (USD)", "average_purchase_price_usd": "Avg purchase price (USD)",
    "average_area_sqft": "Avg area (sq ft)", "average_price_per_sqft_usd": "Avg price per sq ft (USD)",
    "office_share": "Office share", "purchase_span_months": "Purchase span (months)",
    "purchase_intensity": "Purchases per active month", "distinct_towers_purchased": "Distinct towers",
    "is_investment_purpose": "Investment purpose rate", "is_loan_applied": "Loan rate",
    "is_company": "Company share", "is_international": "International share",
}


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference (cluster vs rest) using the pooled SD."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def distinguishing_traits(df: pd.DataFrame, labels: np.ndarray, numeric: list[str] = NUMERIC_PROFILE,
                          binary: list[str] = BINARY_PROFILE, alpha: float = 0.05,
                          min_effect: float = 0.5) -> pd.DataFrame:
    """One-vs-rest tests per cluster and variable.

    numeric -> Mann-Whitney U p-value + Cohen's d; binary -> chi-square p-value + risk difference
    expressed as a standardised effect (Cohen's h). Bonferroni-corrected across all tests.
    A trait is 'distinguishing' only if it passes BOTH significance and the effect-size floor.
    """
    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        for v in numeric:
            a, b = df.loc[m, v], df.loc[~m, v]
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            rows.append({"segment": int(c), "variable": v, "kind": "numeric", "effect": cohens_d(a, b),
                         "cluster_value": float(a.mean()), "rest_value": float(b.mean()), "p_value": float(p)})
        for v in binary:
            a, b = df.loc[m, v], df.loc[~m, v]
            tab = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
            p = stats.chi2_contingency(tab, correction=False)[1] if tab.min() > 0 or tab.sum(axis=0).min() > 0 else 1.0
            h = 2 * np.arcsin(np.sqrt(a.mean())) - 2 * np.arcsin(np.sqrt(b.mean()))
            rows.append({"segment": int(c), "variable": v, "kind": "binary", "effect": float(h),
                         "cluster_value": float(a.mean()), "rest_value": float(b.mean()), "p_value": float(p)})
    out = pd.DataFrame(rows)
    out["p_adj"] = (out["p_value"] * len(out)).clip(upper=1.0)
    # for binaries a smaller effect floor (Cohen's h 0.2 = small) is used; numerics use d >= min_effect
    floor = np.where(out["kind"] == "numeric", min_effect, 0.2)
    out["distinguishing"] = (out["p_adj"] < alpha) & (out["effect"].abs() >= floor)
    return out


def name_segments(traits: pd.DataFrame, used_in_model: list[str]) -> dict[int, str]:
    """Build unique, data-derived names from each cluster's strongest distinguishing traits."""
    names: dict[int, str] = {}
    taken: set[str] = set()
    for seg, g in traits.groupby("segment"):
        g = g[g["distinguishing"] & g["variable"].isin(DESCRIPTORS)].copy()
        g["abs"] = g["effect"].abs()
        g = g.sort_values("abs", ascending=False)
        adjs = [DESCRIPTORS[r.variable][0 if r.effect > 0 else 1] for r in g.itertuples()]
        base = " ".join(adjs[:2]) + " Buyers" if adjs else "Mainstream Balanced Buyers"
        name, i = base, 2
        while name in taken:  # disambiguate with the next trait, else a numeric suffix
            if len(adjs) >= i + 1 and i == 2:
                name = f"{adjs[0]} {adjs[i]} Buyers"
            else:
                name = f"{base} ({i})"
            i += 1
        taken.add(name)
        names[int(seg)] = name
    return names


def _share(s: pd.Series, top: int = 3) -> dict[str, float]:
    return {str(k): round(float(v), 4) for k, v in s.value_counts(normalize=True).head(top).items()}


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, names: dict[int, str],
                     traits: pd.DataFrame) -> dict[str, Any]:
    """Full per-segment profile dictionary (saved as cluster_profiles.json)."""
    d = df.copy()
    d["segment"] = labels
    overall = {
        **{v: float(d[v].mean()) for v in NUMERIC_PROFILE + BINARY_PROFILE},
        "sd": {v: float(d[v].std()) for v in NUMERIC_PROFILE + BINARY_PROFILE},
        "n": int(len(d)), "gender_female_share": float((d["gender"] == "F").mean()),
        "office_buyer_share": float((d["office_purchase_count"] > 0).mean()),
        "mixed_category_share": float((d["distinct_property_categories"] > 1).mean()),
        "top_country_share": float(d["country"].value_counts(normalize=True).iloc[0]),
    }
    profiles: dict[str, Any] = {"overall": overall, "segments": {}}
    for seg, g in d.groupby("segment"):
        t = traits[(traits.segment == seg) & traits.distinguishing].copy()
        t["abs"] = t["effect"].abs()
        t = t.sort_values("abs", ascending=False)
        sat = g["satisfaction_score"].value_counts(normalize=True).sort_index()
        age_bins = pd.cut(g["age"], [0, 34, 49, 64, 120], labels=["<35", "35-49", "50-64", "65+"])
        profiles["segments"][str(int(seg))] = {
            "segment_id": int(seg), "name": names[int(seg)], "n": int(len(g)), "share": float(len(g) / len(d)),
            "means": {v: float(g[v].mean()) for v in NUMERIC_PROFILE + BINARY_PROFILE},
            "medians": {v: float(g[v].median()) for v in NUMERIC_PROFILE},
            "age_quartiles": [float(x) for x in g["age"].quantile([.25, .5, .75])],
            "age_band_share": {str(k): round(float(v), 4) for k, v in age_bins.value_counts(normalize=True).sort_index().items()},
            "satisfaction_distribution": {str(int(k)): round(float(v), 4) for k, v in sat.items()},
            "client_type_share": _share(g["client_type"], 2), "purpose_share": _share(g["acquisition_purpose"], 2),
            "referral_share": _share(g["referral_channel"], 3), "top_countries": _share(g["country"], 3),
            "top_regions": _share(g["region"], 3), "total_spend_sum": float(g["total_spend_usd"].sum()),
            "office_buyer_share": float((g["office_purchase_count"] > 0).mean()),
            "mixed_category_share": float((g["distinct_property_categories"] > 1).mean()),
            "same_tower_repeat_share": float((g["distinct_towers_purchased"] < g["purchase_count"]).mean()),
            "distinguishing_traits": [
                {"variable": r.variable, "label": LABELS.get(r.variable, r.variable), "effect": round(float(r.effect), 3),
                 "segment_value": round(float(r.cluster_value), 4), "rest_value": round(float(r.rest_value), 4),
                 "p_adj": float(r.p_adj)} for r in t.itertuples()],
        }
    return profiles


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """(Cramér's V, chi-square p-value) for two categorical variables."""
    tab = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if min(tab.shape) > 1 else 0.0
    return v, float(p)
