"""Reusable scoring: raw customer fields -> engineered features -> saved pipeline -> segment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import Winsorizer  # noqa: F401  (needed to unpickle the pipeline)

REQUIRED = ["purchase_count", "average_purchase_price_usd", "average_price_per_sqft_usd", "office_share",
            "purchase_span_months"]


# (min, max) inclusive; None = unbounded on that side. Strictly-positive fields are checked separately.
_RANGES = {"purchase_count": (1, 1000), "average_purchase_price_usd": (None, None), "average_price_per_sqft_usd": (None, None),
           "office_share": (0.0, 1.0), "purchase_span_months": (0, 1200)}
_POSITIVE = {"average_purchase_price_usd", "average_price_per_sqft_usd"}


def validate_new_customer(raw: dict[str, Any]) -> dict[str, float]:
    """Check and coerce the required numeric fields; raise ONE ValueError listing every problem.

    Unknown extra fields are ignored. Missing, non-numeric, non-finite or out-of-range values are
    rejected instead of being scored (a NaN would otherwise silently map to an arbitrary segment).
    """
    problems, clean = [], {}
    for name in REQUIRED:
        v = raw.get(name)
        if v is None:
            problems.append(f"{name}: missing")
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            problems.append(f"{name}: {v!r} is not a number")
            continue
        lo, hi = _RANGES[name]
        if not np.isfinite(x):
            problems.append(f"{name}: must be a finite number")
        elif name in _POSITIVE and x <= 0:
            problems.append(f"{name}: must be > 0 (got {x:g})")
        elif (lo is not None and x < lo) or (hi is not None and x > hi):
            problems.append(f"{name}: {x:g} outside allowed range [{lo}, {hi}]")
        else:
            clean[name] = x
    if problems:
        raise ValueError("Invalid customer record: " + "; ".join(problems))
    return clean


def engineer_new_customer(raw: dict[str, Any]) -> pd.DataFrame:
    """Turn a dict of raw fields into the single-row frame the saved pipeline expects.

    Required: purchase_count, average_purchase_price_usd, average_price_per_sqft_usd,
    office_share (0-1), purchase_span_months. Optional profile fields are carried through for insights.
    """
    row = {**raw, **validate_new_customer(raw)}
    row.setdefault("age", np.nan)
    row["purchase_intensity"] = row["purchase_count"] / max(row["purchase_span_months"], 1)
    row["is_investment_purpose"] = int(str(row.get("acquisition_purpose", "")).lower() == "investment")
    row["is_loan_applied"] = int(str(row.get("loan_applied", "")).lower() in ("yes", "1", "true"))
    row["is_company"] = int(str(row.get("client_type", "")).lower() == "company")
    row["is_international"] = int(str(row.get("country", "USA")) != "USA")
    return pd.DataFrame([row])


class SegmentScorer:
    """Loads saved artifacts once; scoring never refits anything."""

    def __init__(self, models_dir: str | Path = "models"):
        d = Path(models_dir)
        self.pipeline = joblib.load(d / "preprocessing_pipeline.pkl")
        self.model = joblib.load(d / "clustering_model.pkl")
        self.config = json.loads((d / "feature_config.json").read_text(encoding="utf-8"))
        self.profiles = json.loads((d / "cluster_profiles.json").read_text(encoding="utf-8"))
        self.features = self.config["clustering_features"]
        f = d / "assignment_stability.json"
        self.stability_table = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return segment id, name, relative-closeness confidence and an out-of-range warning flag."""
        absent = [c for c in self.features if c not in df.columns]
        if absent:
            raise ValueError(f"Input is missing model features {absent}; expected {self.features}")
        X = self.pipeline.transform(df[self.features])
        bad = ~np.isfinite(X).all(axis=1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} row(s) contain missing, non-numeric or out-of-domain feature values "
                             f"(e.g. non-positive prices); first offending row index: {df.index[np.flatnonzero(bad)[0]]}")
        seg, prob = self.model.predict_with_confidence(X)
        names = [self.profiles["segments"][str(s)]["name"] for s in seg]
        d2 = ((X - self.model.centroids[seg]) ** 2).sum(axis=1)
        out = pd.DataFrame({"segment": seg, "segment_name": names, "confidence": prob[np.arange(len(seg)), seg],
                            "distance_ratio": d2 / self.model.distance_scale}, index=df.index)
        # flag inputs far outside the training distribution (>3x the median squared distance) or <3 purchases
        out["closeness_ratio"] = self.model.closeness_ratio(X)
        out["est_assignment_stability"] = self._estimate_stability(out["closeness_ratio"].to_numpy())
        out["low_fit_warning"] = (out["distance_ratio"] > 3) | (df["purchase_count"] < self.config["min_train_purchase_count"])
        return out

    def _estimate_stability(self, ratio: np.ndarray) -> np.ndarray:
        """Expected share of resampled refits that keep a buyer at this closeness in the same segment."""
        t = self.stability_table
        if not t:
            return np.full(len(ratio), np.nan)
        b = np.clip(np.digitize(ratio, t["bin_edges"][1:-1]), 0, len(t["mean_stability"]) - 1)
        return np.asarray(t["mean_stability"])[b]

    def score_new_customer(self, raw: dict[str, Any]) -> dict[str, Any]:
        df = engineer_new_customer(raw)
        res = self.score(df).iloc[0]
        seg = int(res["segment"])
        return {"segment": seg, "segment_name": res["segment_name"], "confidence": float(res["confidence"]),
                "low_fit_warning": bool(res["low_fit_warning"]),
                "closeness_ratio": float(res["closeness_ratio"]), "est_assignment_stability": float(res["est_assignment_stability"]),
                "profile": self.profiles["segments"][str(seg)]}
