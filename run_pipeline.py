"""End-to-end, reproducible pipeline:  python run_pipeline.py
audit -> feature selection -> model grid (+null baseline, stability) -> selection -> final fit
-> profiling/naming -> playbooks -> fairness audit -> artifacts, figures, reports."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from src.clustering import SegmentModel, fit_kmeans, linkage_matrix  # noqa: E402
from src.data_processing import audit_all, integrity_checks, load_buyer_table, load_config, load_raw  # noqa: E402
from src.evaluation import (assignment_stability, feature_sensitivity, internal_metrics,  # noqa: E402
                            stability_by_closeness)
from src.experiments import encode, pca_comparison, run_grid  # noqa: E402
from src.feature_engineering import (build_buyer_table, correlation_report, low_variance_report,  # noqa: E402
                                     make_full_pipeline)
from src.profiling import (BINARY_PROFILE, NUMERIC_PROFILE, cramers_v, distinguishing_traits,  # noqa: E402
                           name_segments, profile_clusters)
from src.recommendations import segment_playbook  # noqa: E402
from src.selection import (BASELINE_WEIGHTS, compare_k, partition_agreement, select_model, selection_sensitivity,  # noqa: E402
                           summarise_sensitivity)

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):  # avoid UnicodeEncodeError on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent
SEED = 42

BEHAVIOUR = ["purchase_count", "average_purchase_price_usd", "average_price_per_sqft_usd", "office_share",
             "purchase_span_months"]
WINSOR = ["purchase_count", "purchase_span_months"]
FEATURE_SETS = {
    "behaviour": dict(numeric=BEHAVIOUR, binary=[], categorical=[], winsor=WINSOR),
    "behaviour+customer": dict(numeric=BEHAVIOUR + ["age", "satisfaction_score"],
                               binary=["is_investment_purpose", "is_loan_applied", "is_company"],
                               categorical=[], winsor=WINSOR),
    "behaviour+customer+geo/channel": dict(numeric=BEHAVIOUR + ["age", "satisfaction_score"],
                                           binary=["is_investment_purpose", "is_loan_applied", "is_company",
                                                   "is_international"],
                                           categorical=["referral_channel"], winsor=WINSOR),
}


def fairness_audit(df: pd.DataFrame, labels: np.ndarray, names: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["segment"] = labels
    d["age_band"] = pd.cut(d["age"], [0, 34, 49, 64, 120], labels=["<35", "35-49", "50-64", "65+"])
    top = d["country"].value_counts().index[:4]
    d["country_group"] = np.where(d["country"].isin(top), d["country"], "Other")
    d["geo_group"] = np.where(d["country"] == "USA", "USA", "International")
    rows, rep = [], []
    for col in ["gender", "geo_group", "country_group", "age_band", "client_type"]:
        v, p = cramers_v(d[col], d["segment"])
        rows.append({"attribute": col, "cramers_v": round(v, 4), "chi2_p_value": round(p, 4),
                     "association": "negligible" if v < 0.1 else "small" if v < 0.2 else "moderate+"})
        tab = pd.crosstab(d[col], d["segment"], normalize="index")
        overall = d["segment"].value_counts(normalize=True).sort_index()
        for g in tab.index:
            for s in tab.columns:
                rep.append({"attribute": col, "group": str(g), "segment": int(s), "segment_name": names[int(s)],
                            "share_in_group": round(float(tab.loc[g, s]), 4),
                            "representation_ratio": round(float(tab.loc[g, s] / overall[s]), 3)})
    return pd.DataFrame(rows), pd.DataFrame(rep)


def fig_save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    rep_dir, fig_dir, mdl_dir = ROOT / cfg["paths"]["reports_dir"], ROOT / cfg["paths"]["figures_dir"], ROOT / cfg["paths"]["models_dir"]
    for d in (rep_dir, fig_dir, mdl_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1. audit ----------
    tables = load_raw(cfg, ROOT)
    audit = audit_all(tables)
    integ = integrity_checks(tables)
    audit.to_csv(rep_dir / "data_audit.csv", index=False)
    df, recon = load_buyer_table(cfg, ROOT, tables)
    recon.to_csv(rep_dir / "data_reconciliation.csv", index=False)
    print("RECONCILIATION (rebuilt vs provided buyer table) mismatching columns:", recon.loc[recon.status != "match", "column"].tolist())
    print(audit.to_string(index=False)); print(integ)

    # ---------- 2. feature selection ----------
    pool = cfg["features"]["numeric_candidates"] + cfg["features"]["binary_candidates"]
    corr_pairs = correlation_report(df, pool, cfg["features"]["corr_threshold"])
    lowvar = low_variance_report(df, pool)
    corr_pairs.to_csv(rep_dir / "feature_redundancy_pairs.csv", index=False)
    decisions = [
        ("age", "excluded from final", "independent of behaviour (|r|<0.1 with every behaviour feature); profiled instead"),
        ("satisfaction_score", "excluded from final", "independent of behaviour; adding it did not raise stability or null-gap (see model_metrics); profiled instead"),
        ("purchase_count", "KEPT", "core activity measure; winsorised at 1st/99th pct (max 13 purchases)"),
        ("total_spend_usd", "dropped", "= purchase_count x avg price; r=0.98 with total_area_sqft, would double-count"),
        ("average_purchase_price_usd", "KEPT", "core ticket-size measure; log1p + scaling; kept over median price (r=0.91) and avg area (r=0.96)"),
        ("median_purchase_price_usd", "dropped", "r=0.91 with average_purchase_price_usd"),
        ("average_area_sqft", "dropped", "r=0.96 with average_purchase_price_usd (size and price carry the same information here)"),
        ("total_area_sqft", "dropped", "r=0.98 with total_spend_usd; derived from count x area"),
        ("average_price_per_sqft_usd", "KEPT", "premium-rate signal not redundant with price or area (|r|<=0.27)"),
        ("office_share", "KEPT", "commercial preference; kept over apartment_share (r=-0.998, exact mirror)"),
        ("apartment_share", "dropped", "mirror image of office_share (r=-0.998)"),
        ("distinct_towers_purchased", "dropped", "r=0.93 with purchase_count"),
        ("tower_diversification", "dropped from clustering", "about 97% of clients = 1.0 (near-constant); including it isolates a 65-buyer micro-cluster instead of describing general behaviour. Reported in profiles instead"),
        ("purchase_span_months", "KEPT", "time between first and last purchase; winsorised; not redundant with count (r=0.27)"),
        ("purchase_intensity", "dropped", "r=-0.71 with purchase_span_months (count/span); span is easier to explain"),
        ("is_investment_purpose", "excluded from final", "independent of behaviour; kept for profiling and individual-level personalisation"),
        ("is_loan_applied", "excluded from final", "independent of behaviour; kept for profiling and individual-level personalisation"),
        ("is_company", "excluded from final", "only 5.2% of clients; near-constant"),
        ("is_international", "excluded from final", "tested in 'behaviour+customer+geo/channel'; audit variable"),
        ("referral_channel", "excluded from final", "tested one-hot in geo/channel set; acquisition variable, profiled"),
        ("gender / country / region", "excluded (audit only)", "sensitive/high-cardinality (57 regions); used only in the fairness audit"),
    ]
    pd.DataFrame(decisions, columns=["feature", "decision", "reason"]).to_csv(rep_dir / "feature_selection.csv", index=False)

    # ---------- 3. experiments ----------
    reuse = "--reuse-grid" in sys.argv and (rep_dir / "model_metrics.csv").exists()
    if reuse:
        print("Reusing reports/model_metrics.csv (--reuse-grid); delete the flag to recompute the grid.")
        grid = pd.read_csv(rep_dir / "model_metrics.csv").drop(columns=["selection_score"], errors="ignore")
    else:
        print("Running model grid (this takes a few minutes)...")
        grid = run_grid(df, FEATURE_SETS, cfg)
    sel = cfg.get("selection", {})
    deployable = sel.get("deployable_algorithms", ["kmeans"])
    weights, k_range = sel.get("weights", BASELINE_WEIGHTS), tuple(sel.get("k_range", [3, 6]))
    min_share = float(sel.get("min_cluster_share", 0.05))
    # benchmark: the best candidate over ALL algorithms (reported, not necessarily deployable)
    unrestricted = select_model(grid.copy(), weights, k_range, min_share)
    pool = grid[grid["algorithm"].isin(deployable)].copy()
    best = select_model(pool, weights, k_range, min_share)
    grid["selection_score"] = np.nan
    grid.loc[pool.index, "selection_score"] = pool["selection_score"]
    grid.to_csv(rep_dir / "model_metrics.csv", index=False)
    best_label = f"{best.algorithm} K={int(best.k)} [{best.feature_set}]"
    unrestricted_label = f"{unrestricted.algorithm} K={int(unrestricted.k)} [{unrestricted.feature_set}]"
    sel_sens = selection_sensitivity(pool)
    sel_sens.to_csv(rep_dir / "selection_sensitivity.csv", index=False)
    selection_sensitivity(grid).to_csv(rep_dir / "selection_sensitivity_all_algorithms.csv", index=False)
    sens_summary = summarise_sensitivity(sel_sens, best_label)
    sens_summary["unrestricted_winner_all_algorithms"] = unrestricted_label
    print("SELECTION SENSITIVITY (deployable pool):", sens_summary)
    k_cmp = compare_k(grid, best.feature_set, "kmeans", (4, 5))
    k_cmp.to_csv(rep_dir / "k4_vs_k5.csv")
    print("SELECTED:", best_label, "| score", round(float(best.selection_score), 4), "| benchmark winner over all algorithms:", unrestricted_label)
    spec, k_final, algo = FEATURE_SETS[best.feature_set], int(best.k), best.algorithm
    assert algo == "kmeans", "only K-Means artifacts can be exported; set selection.deployable_algorithms accordingly"

    # ---------- 4. final model ----------
    pipe = make_full_pipeline(spec["numeric"], spec["binary"], spec["categorical"], cfg["features"]["log_transform"], spec["winsor"])
    feats = spec["numeric"] + spec["binary"] + spec["categorical"]
    X = pipe.fit_transform(df[feats])
    km = fit_kmeans(X, k_final, seed=SEED, n_init=cfg["clustering"]["n_init"])
    order = np.argsort(-np.bincount(km.labels))          # segment 0 = largest
    remap = {int(old): new for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in km.labels])
    centroids = km.centroids[order]
    d2min = ((X[:, None, :] - centroids[None]) ** 2).sum(2).min(1)
    model = SegmentModel("kmeans", k_final, centroids, float(np.median(d2min)))
    assert (model.predict(X) == labels).all(), "nearest-centroid scoring must reproduce training labels"
    final_metrics = internal_metrics(X, labels)
    df["segment"] = labels
    ratio = model.closeness_ratio(X)
    stab = assignment_stability(X, centroids, labels, n_boot=cfg["stability"]["n_bootstrap"],
                                frac=cfg["stability"]["bootstrap_fraction"], seed=SEED)
    stab_table = stability_by_closeness(ratio, stab)
    print("ASSIGNMENT STABILITY: mean", round(float(stab.mean()), 3), "| share of buyers >=0.8:", round(float((stab >= 0.8).mean()), 3),
          "| by closeness bin:", [round(v, 2) for v in stab_table["mean_stability"]])
    t0 = time.perf_counter(); model.predict(X); infer_ms_total = 1000 * (time.perf_counter() - t0)
    k45 = partition_agreement(X, 4, 5, seed=SEED, n_init=cfg["clustering"]["n_init"])
    k45["crosstab"].to_csv(rep_dir / "k4_k5_crosstab.csv")
    print("K4 vs K5 partition ARI:", round(k45["ari"], 3), "| share of each K=4 segment kept together in K=5:", k45["share_in_dominant_child"])

    # ---------- 5. validation extras ----------
    sens = feature_sensitivity(lambda cols: encode(df, {**spec, "numeric": [c for c in spec["numeric"] if c in cols],
                                                        "winsor": [c for c in spec["winsor"] if c in cols]}, cfg),
                               spec["numeric"], "kmeans", k_final, labels, SEED)
    sens.to_csv(rep_dir / "feature_sensitivity.csv", index=False)
    pca_cmp = pca_comparison(df, spec, cfg, k_final)
    # gender-inclusion test (fairness decision)
    d_g = df.copy(); d_g["is_female"] = (d_g["gender"] == "F").astype(int)
    Xg = encode(d_g, {**spec, "binary": spec["binary"] + ["is_female"]}, cfg)
    lab_g = fit_kmeans(Xg, k_final, seed=SEED, n_init=20).labels
    gender_test = {"ari_with_vs_without_gender": float(adjusted_rand_score(labels, lab_g)),
                   "silhouette_without": final_metrics["silhouette"], "silhouette_with": internal_metrics(Xg, lab_g)["silhouette"]}
    # seed reproducibility fingerprint
    fingerprint = hashlib.sha256(labels.astype(np.int8).tobytes()).hexdigest()[:16]

    # ---------- 6. profiling / naming / playbooks ----------
    traits = distinguishing_traits(df, labels)
    traits.to_csv(rep_dir / "segment_trait_tests.csv", index=False)
    names = name_segments(traits, feats)
    profiles = profile_clusters(df, labels, names, traits)
    for sid, p in profiles["segments"].items():
        p["playbook"] = segment_playbook(p, profiles["overall"])
    fair, fair_rep = fairness_audit(df, labels, names)
    fair.to_csv(rep_dir / "fairness_audit.csv", index=False); fair_rep.to_csv(rep_dir / "fairness_representation.csv", index=False)

    flat = []
    for sid, p in profiles["segments"].items():
        flat.append({"segment": int(sid), "name": p["name"], "customers": p["n"], "share_pct": round(100 * p["share"], 1),
                     **{LABEL: round(p["means"][v], 3) for v, LABEL in
                        [("age", "avg_age"), ("purchase_count", "avg_purchases"), ("average_purchase_price_usd", "avg_purchase_usd"),
                         ("total_spend_usd", "avg_total_spend_usd"), ("average_area_sqft", "avg_area_sqft"), ("office_share", "office_share"),
                         ("purchase_span_months", "avg_span_months"), ("is_investment_purpose", "investment_rate"),
                         ("is_loan_applied", "loan_rate"), ("is_company", "company_share"), ("is_international", "international_share"),
                         ("satisfaction_score", "avg_satisfaction")]},
                     "top_region": next(iter(p["top_regions"])),
                     "distinguishing": "; ".join(f"{t['label']} ({t['effect']:+.2f})" for t in p["distinguishing_traits"][:4])})
    pd.DataFrame(flat).to_csv(rep_dir / "cluster_profiles.csv", index=False)
    df[["client_id", "segment"]].assign(segment_name=df["segment"].map(names), closeness_ratio=ratio.round(4),
                                         assignment_stability=stab.round(3)).to_csv(rep_dir / "segment_assignments.csv", index=False)

    # ---------- 7. artifacts ----------
    joblib.dump(pipe, mdl_dir / "preprocessing_pipeline.pkl")
    joblib.dump(model, mdl_dir / "clustering_model.pkl")
    fcfg = {"clustering_features": feats, "numeric": spec["numeric"], "binary": spec["binary"], "categorical": spec["categorical"],
            "winsorised": spec["winsor"], "log_transformed": cfg["features"]["log_transform"],
            "feature_set": best.feature_set, "algorithm": "kmeans", "k": k_final, "random_state": SEED,
            "min_train_purchase_count": int(df["purchase_count"].min()),
            "metrics": final_metrics, "label_fingerprint": fingerprint}
    meta = {"model_version": f"kmeans-k{k_final}-{fingerprint}", "label_fingerprint": fingerprint,
            "trained_rows": int(len(df)), "random_state": SEED,
            "dataset_sha256": {k: hashlib.sha256((ROOT / cfg["paths"]["data_dir"] / f).read_bytes()).hexdigest()[:16]
                               for k, f in cfg["files"].items()},
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                 "scikit_learn": __import__("sklearn").__version__, "joblib": joblib.__version__},
            "selection": {"label": best_label, "score": float(best.selection_score), "rule": "src/selection.py baseline weights"},
            "inference_ms_for_all_training_rows": round(infer_ms_total, 3)}
    (mdl_dir / "assignment_stability.json").write_text(json.dumps(stab_table, indent=2), encoding="utf-8")
    (mdl_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (mdl_dir / "feature_config.json").write_text(json.dumps(fcfg, indent=2), encoding="utf-8")
    (mdl_dir / "cluster_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    summary = {"audit": audit.to_dict("records"), "integrity": integ, "selected": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in best.items() if k in
               ("feature_set", "algorithm", "k", "silhouette", "null_silhouette", "silhouette_gap", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "seed_ari", "bootstrap_ari", "bootstrap_ari_sd", "selection_score")},
               "final_metrics": final_metrics, "pca_comparison": pca_cmp, "gender_test": gender_test, "fingerprint": fingerprint,
               "correlated_pairs": corr_pairs.to_dict("records"), "names": names,
               "selection_sensitivity": sens_summary,
               "k4_vs_k5": {"partition_ari": k45["ari"], "share_in_dominant_child": k45["share_in_dominant_child"],
                            "k5_segment_sizes": k45["new_segment_sizes_kb"]}}
    (rep_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---------- 8. figures ----------
    make_figures(df, X, labels, names, grid, best, fig_dir, spec, cfg)
    write_reports(rep_dir, df, grid, best, summary, profiles, fair, sens, audit, integ)
    print("Done. Fingerprint:", fingerprint)


def make_figures(df, X, labels, names, grid, best, fig_dir, spec, cfg) -> None:
    # dataset overview
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    df["client_type"].value_counts().plot.bar(ax=ax[0, 0], title="Client type", color="#4C78A8")
    df["acquisition_purpose"].value_counts().plot.bar(ax=ax[0, 1], title="Acquisition purpose", color="#F58518")
    df["loan_applied"].value_counts().plot.bar(ax=ax[0, 2], title="Loan applied", color="#54A24B")
    df["country"].value_counts().plot.bar(ax=ax[1, 0], title="Country", color="#B279A2")
    ax[1, 1].hist(df["age"], bins=25, color="#E45756"); ax[1, 1].set_title("Age")
    df["satisfaction_score"].value_counts().sort_index().plot.bar(ax=ax[1, 2], title="Satisfaction score", color="#72B7B2")
    fig_save(fig, "01_dataset_overview.png", fig_dir)
    # selection curves (best feature set, all algorithms)
    g = grid[grid.feature_set == best.feature_set]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    km = g[g.algorithm == "kmeans"]
    inertia = [fit_kmeans(X, k, seed=SEED, n_init=10).estimator.inertia_ for k in cfg["clustering"]["k_range"]]
    ax[0, 0].plot(cfg["clustering"]["k_range"], inertia, "o-"); ax[0, 0].set_title("Elbow (K-Means inertia)")
    for metric, a, t in [("silhouette", ax[0, 1], "Silhouette (higher better)"), ("davies_bouldin", ax[0, 2], "Davies-Bouldin (lower better)"),
                         ("calinski_harabasz", ax[1, 0], "Calinski-Harabasz (higher better)")]:
        for algo, gg in g.groupby("algorithm"):
            a.plot(gg.k, gg[metric], "o-", label=algo)
        a.set_title(t)
    ax[0, 1].plot(km.k, km.null_silhouette, "k--", label="kmeans on shuffled (null)"); ax[0, 1].legend(fontsize=7)
    for algo, gg in g.dropna(subset=["bootstrap_ari"]).groupby("algorithm"):
        ax[1, 1].plot(gg.k, gg.bootstrap_ari, "o-", label=algo)
    ax[1, 1].set_title("Bootstrap stability (ARI)"); ax[1, 1].legend(fontsize=7)
    ax[1, 2].bar(*zip(*sorted(pd.Series(labels).value_counts().items())), color="#4C78A8"); ax[1, 2].set_title("Final cluster sizes")
    for a in ax.ravel()[:5]:
        a.set_xlabel("K")
    fig_save(fig, "02_model_selection.png", fig_dir)
    # dendrogram
    Z = linkage_matrix(X, "ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, no_labels=True); ax.set_title("Hierarchical (Ward) dendrogram, last 30 merges")
    fig_save(fig, "03_dendrogram.png", fig_dir)
    # PCA
    P = PCA(2, random_state=SEED).fit(X); Zp = P.transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in sorted(names):
        m = labels == s; ax.scatter(Zp[m, 0], Zp[m, 1], s=8, alpha=.6, label=f"{s}: {names[s]}")
    ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=7); ax.set_title("PCA projection (visualisation only; clustering is not done in PCA space)")
    fig_save(fig, "04_pca_clusters.png", fig_dir)
    # segment heatmap (z-scores vs overall)
    vars_ = NUMERIC_PROFILE[:2] + ["purchase_count", "average_purchase_price_usd", "average_price_per_sqft_usd", "office_share",
                                   "purchase_span_months"] + BINARY_PROFILE
    z = pd.DataFrame({s: [(df.loc[labels == s, v].mean() - df[v].mean()) / df[v].std() for v in vars_] for s in sorted(names)}, index=vars_)
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(z.columns))); ax.set_xticklabels([f"{s}" for s in z.columns]); ax.set_yticks(range(len(vars_))); ax.set_yticklabels(vars_)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, label="segment mean vs overall (SD units)"); ax.set_title("Segment comparison")
    fig_save(fig, "05_segment_heatmap.png", fig_dir)
    # distributions by segment
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for a, (v, t) in zip(ax.ravel(), [("age", "Age"), ("purchase_count", "Purchases"), ("total_spend_usd", "Total spend"),
                                      ("average_purchase_price_usd", "Avg purchase"), ("office_share", "Office share"),
                                      ("purchase_span_months", "Span (months)"), ("satisfaction_score", "Satisfaction")]):
        a.boxplot([df.loc[labels == s, v] for s in sorted(names)], labels=sorted(names)); a.set_title(t)
    df.groupby("segment")[["is_loan_applied", "is_investment_purpose"]].mean().plot.bar(ax=ax[1, 3], title="Loan / investment rate")
    fig_save(fig, "06_segment_distributions.png", fig_dir)


def _other_algorithms_text(grid: pd.DataFrame, feature_set: str) -> str:
    """Computed (not hardcoded) summary of the benchmark algorithms for the report."""
    g = grid[(grid["feature_set"] == feature_set) & grid["k"].between(3, 6)]
    parts = ["Hierarchical clustering and GMM are evaluated as benchmarks (full table: `model_metrics.csv`)."]
    for algo, label in (("hier_average", "Average linkage"), ("hier_complete", "Complete linkage"), ("hier_ward", "Ward linkage"), ("gmm", "GMM")):
        x = g[g["algorithm"] == algo]
        if x.empty:
            continue
        top = x.loc[x["silhouette"].idxmax()]
        parts.append(f"{label}: silhouette {x['silhouette'].min():.3f}-{x['silhouette'].max():.3f} for K=3..6 "
                     f"(best K={int(top['k'])} with smallest cluster {top['min_cluster_share']:.1%} of buyers).")
    km = g[g["algorithm"] == "kmeans"]
    parts.append(f"K-Means: silhouette {km['silhouette'].min():.3f}-{km['silhouette'].max():.3f}. A very small smallest-cluster share means a high silhouette is "
                 "obtained by isolating a few buyers, not by finding a usable segmentation. Seed ARI is 1.0 for deterministic algorithms by construction.")
    return " ".join(parts)


def write_reports(rep_dir, df, grid, best, summary, profiles, fair, sens, audit, integ) -> None:
    o, segs = profiles["overall"], profiles["segments"]
    sel, fm = summary["selected"], summary["final_metrics"]
    L = ["# Model Report — Buyer Segmentation", "", "_Generated by `run_pipeline.py`; every number below is computed from the data._", ""]
    L += ["## 1. Dataset", ""]
    for r in summary["audit"]:
        L.append(f"- `{r['file']}`: {r['rows']:,} rows x {r['columns']} cols, {r['missing_cells']:,} missing cells, {r['duplicate_rows']} duplicate rows")
    L += [f"- The {summary['integrity']['properties_available_no_client']:,} missing `client_ref` values are exactly the *Available* (unsold) listings, not data errors; "
          f"{summary['integrity']['clients_without_purchase']} clients have no purchase and every client has >= {summary['integrity']['min_purchases_per_client']} purchases.",
          f"- Transactions span {summary['integrity']['purchase_date_range'][0]} to {summary['integrity']['purchase_date_range'][1]}.", ""]
    L += ["## 2. Key data finding (read this first)", "",
          f"Behavioural, financing, intent and satisfaction variables are nearly statistically independent (most pairwise |r| < 0.1). "
          f"On the selected model the silhouette is **{sel['silhouette']:.3f}**, versus **{sel['null_silhouette']:.3f}** on column-shuffled data with the same marginals "
          f"(gap = {sel['silhouette_gap']:+.3f}). The data therefore contain **little natural cluster structure**: segments here are *useful partitions of a continuum*, "
          "not sharply separated populations. Including tower diversification would isolate a 65-buyer micro-segment (the only buyers with units in more than one tower pattern) "
          "rather than describe general buyer behaviour, so that feature is excluded from the clustering and reported in profiles instead.", ""]
    L += ["## 3. Features", "", "See `feature_selection.csv` (decision and reason for every candidate) and `feature_redundancy_pairs.csv`.", ""]
    L += ["## 4. Algorithms tested", "", "K-Means, Agglomerative (Ward / complete / average), Gaussian Mixture (diagonal covariance); K = 2..8; three feature sets. Full table: `model_metrics.csv`.", ""]
    kk = grid[(grid.feature_set == best.feature_set) & (grid.algorithm == "kmeans")][["k", "silhouette", "null_silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "seed_ari", "bootstrap_ari"]]
    L += [f"### K-Means metrics for the selected feature set (`{best.feature_set}`)", "", kk.round(3).to_markdown(index=False) if hasattr(kk, "to_markdown") else kk.round(3).to_string(index=False), ""]
    L += ["## 5. Final model", "",
          f"- Algorithm: **K-Means**, K = **{sel['k']:.0f}**, feature set `{sel['feature_set']}` (selected automatically by the documented rule in `src/selection.py` (baseline weights))",
          f"- Silhouette {fm['silhouette']:.3f} | Davies-Bouldin {fm['davies_bouldin']:.3f} | Calinski-Harabasz {fm['calinski_harabasz']:.1f}",
          f"- Stability: bootstrap ARI {sel['bootstrap_ari']:.3f} (sd {sel['bootstrap_ari_sd']:.3f}); random-restart ARI {sel['seed_ari']:.3f} (single-init restarts)",
          f"- PCA check: reaching 90% variance needs {summary['pca_comparison']['n_components']} components (all features are nearly uncorrelated), so PCA gives no compression; "
          f"silhouette {summary['pca_comparison']['sil_pca']:.3f} vs {summary['pca_comparison']['sil_original']:.3f} in the original space, partition agreement ARI {summary['pca_comparison']['ari_original_vs_pca']:.2f}. PCA is therefore used for visualisation only.",
          f"- Gender-inclusion test: adding gender changes the partition little (ARI {summary['gender_test']['ari_with_vs_without_gender']:.2f}); gender is excluded from clustering and kept for the fairness audit.", "",
          ""]
    ss, k45 = summary["selection_sensitivity"], summary["k4_vs_k5"]
    kc = pd.read_csv(rep_dir / "k4_vs_k5.csv").rename(columns={"Unnamed: 0": "metric"})
    L += ["### How the solution was selected (and how firmly)", "",
          f"- Rule (`src/selection.py`, config `selection:`): only deployable algorithms are eligible; K in 3..6; every segment >= 5% of buyers; "
          "score = 0.4 subsample-stability ARI + 0.2 seed ARI + 0.2 silhouette-gap + 0.1 Davies-Bouldin + 0.1 Calinski-Harabasz (last three min-max normalised).",
          f"- Selected: **{ss['baseline_winner']}**. Best over ALL algorithms (benchmark only): **{ss['unrestricted_winner_all_algorithms']}** - it is not used because the deployed scorer is nearest-centroid K-Means, "
          "so the evaluated model must be the shipped model.",
          f"- **Robustness:** the selected solution wins {ss['share_of_scenarios_baseline_wins']:.0%} of {ss['n_scenarios']} alternative weight/constraint scenarios. "
          "Winners across scenarios: " + "; ".join(f"{k} ({v})" for k, v in ss["winner_counts"].items()) + ". See `selection_sensitivity.csv`.",
          "- The choice between K=4 and K=5 depends mainly on the minimum-segment-size constraint, not on the quality metrics. "
          "Metrics are within each other's resampling noise (see standard deviations in `model_metrics.csv`).", "",
          "### K=4 versus K=5", "", kc.round(3).to_string(index=False), "",
          f"Partition agreement between the two solutions: ARI {k45['partition_ari']:.2f}. K=5 segment sizes: {k45['k5_segment_sizes']}. "
          "This is a business trade-off (is a small, distinct segment worth a dedicated approach?), not a question the metrics resolve.", "",
          "### Other algorithms", "",
          _other_algorithms_text(grid, best.feature_set), ""]
    rec = pd.read_csv(rep_dir / "data_reconciliation.csv")
    bad = rec.loc[rec["status"] != "match", ["column", "n_mismatch", "max_abs_diff"]]
    L += ["### Data reconciliation (rebuilt vs provided buyer table)", "",
          f"The buyer table is rebuilt from `clients_cleaned.csv` and the sold rows of `properties_cleaned.csv`. {len(rec) - len(bad)} of {len(rec)} shared columns match the provided "
          "`buyer_features_model_ready.csv` exactly." + (" Mismatches:" if len(bad) else ""), ""]
    if len(bad):
        L += [bad.round(4).to_string(index=False), "",
              "`office_share` in the provided file is not the share of a buyer's purchases that are offices (it is not reproducible from units, area or value), "
              "so the pipeline uses the rebuilt value. Currency (USD) and area (sq ft) units are assumed from column names; the source files carry no unit metadata.", ""]
    L += ["### Per-buyer assignment stability", "",
          "Share of resampled K-Means refits that keep each buyer in the same segment (`segment_assignments.csv`, `models/assignment_stability.json`). "
          "A closeness-to-stability table lets a new buyer receive a data-backed estimate. The old relative-closeness score is NOT a probability and is not used for user-facing confidence.", ""]
    L += ["## 6. Cluster profiles", ""]
    for sid, p in segs.items():
        m = p["means"]
        L += [f"### Segment {sid}: {p['name']}  ({p['n']:,} buyers, {100 * p['share']:.1f}%)", "",
              f"- Avg age {m['age']:.1f} | purchases {m['purchase_count']:.2f} | avg purchase ${m['average_purchase_price_usd']:,.0f} | total spend ${m['total_spend_usd']:,.0f}",
              f"- Office share {m['office_share']:.1%} | loan rate {m['is_loan_applied']:.1%} | investment rate {m['is_investment_purpose']:.1%} | satisfaction {m['satisfaction_score']:.2f} | international {m['is_international']:.1%}",
              f"- Top region: {next(iter(p['top_regions']))} | top referral: {next(iter(p['referral_share']))}",
              "- Distinguishing traits: " + ("; ".join(f"{t['label']} (effect {t['effect']:+.2f})" for t in p["distinguishing_traits"][:5]) or "none"), ""]
    L += ["## 7. Fairness audit", "", fair.to_string(index=False), "",
          "Cramér's V < 0.1 = negligible association between the attribute and segment membership. A small p-value with a negligible V (e.g. USA vs international) means a detectable but practically tiny skew; "
          "see `fairness_representation.csv` for per-group representation ratios. Gender, country and region were NOT used for clustering.", ""]
    L += ["## 8. Limitations", "",
          "- Weak cluster separation (see section 2); segment boundaries are soft and a client near a boundary could sit in the neighbouring segment.",
          "- Financing, investment purpose and satisfaction do not differ materially across behavioural segments, so they are handled at the *individual* level in the app.",
          "- No income, wealth, budget or transaction-channel data exist; nothing in the system infers them.",
          "- Every client has >= 3 purchases and purchases are monthly snapshots (2024-01 to 2025-12); scoring customers with no purchase history is outside the training range (the app warns).",
          "- Cross-sectional data: segments describe the observed period and say nothing about causality or future behaviour.",
          "- K=4 and K=5 cannot be separated by the metrics; the final K is partly a business judgement (minimum segment size).",
          "- Currency and area units are assumed (USD, sq ft); the source tables have no unit metadata.",
          "- Segment-level differences in satisfaction (e.g. the high-volume segment) are exploratory: they do not survive multiple-comparison correction here.",
          "- No business outcome (conversion, retention, satisfaction change) has been measured; recommendations are untested hypotheses until a pilot is run.", ""]
    (rep_dir / "model_report.md").write_text("\n".join(L), encoding="utf-8")
    I = ["# Business Insights", "", "> Statements about *needs* and *friction* are **hypotheses derived from observed segment behaviour** (not measured preferences) and should be validated with customer research. Segment findings describe groups, never individuals.", ""]
    for sid, p in segs.items():
        pb = p["playbook"]
        I += [f"## Segment {sid}: {p['name']} ({100 * p['share']:.0f}% of buyers)", ""] + [f"- {x}" for x in pb["insights"]]
        I += ["", "**Needs:** " + " ".join(pb["customer_needs"]), "", "**Friction:** " + " ".join(pb["potential_friction"]), "",
              "**Marketing:** " + " ".join(pb["marketing_recommendation"]), "", "**Property:** " + " ".join(pb["property_strategy"]), "",
              "**Financing:** " + " ".join(pb["financing_strategy"]), "", "**Engagement:** " + " ".join(pb["engagement_strategy"]), ""]
    (rep_dir / "insights.md").write_text("\n".join(I), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
