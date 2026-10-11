"""Buyer Intelligence Platform — Streamlit front end.
Run from the project root:  streamlit run app/streamlit_app.py
The app only LOADS the saved artifacts in models/ (run `python run_pipeline.py` first); it never refits.
"""
from __future__ import annotations
 
import json
import sys
from pathlib import Path
 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
from src.data_processing import load_buyer_table, load_config, load_raw  # noqa: E402
from src.feature_engineering import build_buyer_table  # noqa: E402
from src.predict import SegmentScorer  # noqa: E402
from src.profiling import BINARY_PROFILE, LABELS  # noqa: E402
from src.recommendations import describe_segment, segment_playbook, structured_recommendations  # noqa: E402
 
ISO3 = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA", "Belgium": "BEL",
        "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}
 
st.set_page_config(page_title="Buyer Intelligence Platform", page_icon="🏢", layout="wide")
 
 
@st.cache_resource(show_spinner=False)
def load_artifacts() -> SegmentScorer:
    return SegmentScorer(ROOT / "models")
 
 
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(ROOT / "config.yaml")
    t = load_raw(cfg, ROOT)
    return load_buyer_table(cfg, ROOT, t)[0], t["properties"]
 
 
@st.cache_data(show_spinner=False)
def load_reports() -> dict[str, pd.DataFrame]:
    r = ROOT / "reports"
    out = {}
    for n in ["model_metrics", "fairness_audit", "feature_selection", "feature_sensitivity", "cluster_profiles", "k4_vs_k5",
              "selection_sensitivity", "selection_sensitivity_all_algorithms", "data_reconciliation", "segment_assignments"]:
        f = r / f"{n}.csv"
        out[n] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    s = r / "run_summary.json"
    out["summary"] = json.loads(s.read_text(encoding="utf-8")) if s.exists() else {}
    return out
 
 
def render_rec(rec: dict) -> None:
    """One recommendation with its evidence, benefits, confidence, responsible rule and limitations."""
    with st.expander(f"{rec['action']}  ·  {rec['id']}"):
        st.markdown("**Evidence**")
        for e in rec["evidence"]:
            st.markdown(f"- {e}")
        st.markdown(f"**Customer benefit:** {rec['customer_benefit']}  \n**Business benefit:** {rec['business_benefit']}")
        st.markdown(f"**Confidence:** {rec['confidence']}  \n**Produced by:** {rec['rule']}")
        st.markdown(f"**Limitations:** {rec['limitations']}  \n**Human review:** {rec['human_review']}")
 
 
try:
    scorer = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python run_pipeline.py` from the project root first.")
    st.stop()
 
buyers, props = load_data()
reports = load_reports()
profiles = scorer.profiles
overall = profiles["overall"]
seg_df = scorer.score(buyers)
buyers = buyers.join(seg_df[["segment", "segment_name", "confidence", "low_fit_warning", "closeness_ratio", "est_assignment_stability"]])
SEG_NAMES = {int(k): v["name"] for k, v in profiles["segments"].items()}
COLOR = {n: c for n, c in zip(SEG_NAMES.values(), px.colors.qualitative.Safe)}
 
# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏢 Buyer Intelligence")
page = st.sidebar.radio("Navigate", ["Dashboard", "Buyer Segmentation", "Customer Intelligence", "New Customer Prediction",
                                     "Investment Behavior", "Geographic Intelligence", "Customer Experience",
                                     "Model Performance", "Methodology"])
st.sidebar.markdown("### Filters")
f_country = st.sidebar.multiselect("Country", sorted(buyers["country"].unique()))
region_opts = sorted(buyers[buyers["country"].isin(f_country)]["region"].unique()) if f_country else sorted(buyers["region"].unique())
f_region = st.sidebar.multiselect("Region", region_opts)
f_purpose = st.sidebar.multiselect("Acquisition purpose", sorted(buyers["acquisition_purpose"].unique()))
f_type = st.sidebar.multiselect("Client type", sorted(buyers["client_type"].unique()))
view = buyers.copy()
for col, sel in [("country", f_country), ("region", f_region), ("acquisition_purpose", f_purpose), ("client_type", f_type)]:
    if sel:
        view = view[view[col].isin(sel)]
st.sidebar.caption(f"{len(view):,} of {len(buyers):,} buyers selected")
if view.empty:
    st.warning("No buyers match the current filters.")
    st.stop()
 
 
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"
 
 
def honesty_banner() -> None:
    sel = reports["summary"].get("selected", {})
    if sel:
        gap = sel.get("silhouette_gap", float("nan"))
        verdict = ("Segments are *useful partitions of a continuum*, not sharply separated groups. " if gap < 0.05
                   else "Separation is clearly above the structure-free baseline. ")
        st.info(f"**How to read the segments:** silhouette {sel.get('silhouette', float('nan')):.3f} vs "
                f"{sel.get('null_silhouette', float('nan')):.3f} for randomly shuffled data. {verdict}"
                "Segment statements describe groups, never individuals.")
 
 
# ------------------------------------------------------------------ pages
if page == "Dashboard":
    st.title("Buyer Intelligence Dashboard")
    sold = props[props["listing_status"] == "Sold"]
    c = st.columns(6)
    c[0].metric("Total buyers", f"{len(view):,}")
    c[1].metric("Transactions", f"{int(view['purchase_count'].sum()):,}")
    c[2].metric("Segments", view["segment"].nunique())
    c[3].metric("Avg satisfaction", f"{view['satisfaction_score'].mean():.2f}/5")
    c[4].metric("Investment rate", pct(view["is_investment_purpose"].mean()))
    c[5].metric("Loan adoption", pct(view["is_loan_applied"].mean()))
    st.caption(f"Portfolio: {len(props):,} listings, {len(sold):,} sold, {len(props) - len(sold):,} still available (no client).")
    sold_v = sold[sold["client_ref"].isin(view["client_id"])]
    sm, integ = reports["summary"], reports["summary"].get("integrity", {})
    sel_s = sm.get("selected", {})
    c = st.columns(5)
    c[0].metric("Observed transaction value", f"${sold_v['sale_price'].sum():,.0f}")
    c[1].metric("Average / median purchase", f"${sold_v['sale_price'].mean():,.0f} / ${sold_v['sale_price'].median():,.0f}")
    c[2].metric("Selected model", f"K-Means, K={int(sel_s['k'])}" if sel_s else "n/a")
    checks_ok = bool(integ) and integ.get("orphan_client_refs", 1) == 0 and integ.get("clients_without_purchase", 1) == 0 \
        and integ.get("missing_client_ref_is_exactly_available_listings", False)
    c[3].metric("Data-integrity checks", "Passed" if checks_ok else "Review")
    c[4].metric("Subsample stability (ARI)", f"{sel_s['bootstrap_ari']:.2f}" if sel_s else "n/a")
    rec_df = reports["data_reconciliation"]
    if not rec_df.empty and (rec_df["status"] != "match").any():
        bad_cols = ", ".join(rec_df.loc[rec_df["status"] != "match", "column"])
        st.caption(f"Data note: the provided model-ready buyer table disagrees with the source transactions on: {bad_cols}. "
                   "All features here are rebuilt from source.")
    a, b = st.columns(2)
    d = view["segment_name"].value_counts().reset_index()
    d.columns = ["Segment", "Buyers"]
    a.plotly_chart(px.pie(d, names="Segment", values="Buyers", hole=.45, color="Segment", color_discrete_map=COLOR,
                          title="Segment distribution"), use_container_width=True)
    b.plotly_chart(px.histogram(view, x="age", nbins=25, color="segment_name", color_discrete_map=COLOR,
                                title="Age distribution by segment"), use_container_width=True)
    a, b, c3 = st.columns(3)
    a.plotly_chart(px.bar(view["country"].value_counts().reset_index(), x="country", y="count", title="Buyers by country"),
                   use_container_width=True)
    b.plotly_chart(px.pie(view, names="acquisition_purpose", title="Acquisition purpose"), use_container_width=True)
    c3.plotly_chart(px.pie(view, names="loan_applied", title="Loan applied"), use_container_width=True)
    honesty_banner()
 
elif page == "Buyer Segmentation":
    st.title("Buyer Segmentation")
    honesty_banner()
    d = view.groupby("segment_name").size().reset_index(name="Buyers")
    st.plotly_chart(px.bar(d, x="segment_name", y="Buyers", color="segment_name", color_discrete_map=COLOR,
                           title="Cluster distribution"), use_container_width=True)
    st.subheader("Segment profiles")
    for sid, p in profiles["segments"].items():
        desc, chars = describe_segment(p, overall)
        with st.expander(f"Segment {sid} — {p['name']}", expanded=int(sid) == 0):
            st.write(desc)
            cols = st.columns(4)
            for i, (k, v) in enumerate(chars.items()):
                cols[i % 4].metric(k, v)
            st.markdown("**Profile (relative to all buyers):** " + " · ".join(f"{k}: **{v}**" for k, v in segment_playbook(p, overall)["profile_levels"].items()))
            if p["distinguishing_traits"]:
                st.dataframe(pd.DataFrame(p["distinguishing_traits"])[["label", "segment_value", "rest_value", "effect"]]
                             .rename(columns={"label": "Distinguishing trait", "segment_value": "Segment", "rest_value": "Rest", "effect": "Effect size"}),
                             hide_index=True, use_container_width=True)
    st.subheader("Cluster comparison (z-score vs overall)")
    vars_ = ["age", "satisfaction_score", "purchase_count", "average_purchase_price_usd", "average_price_per_sqft_usd", "office_share",
             "purchase_span_months", "is_investment_purpose", "is_loan_applied", "is_company", "is_international"]
    z = pd.DataFrame({n: [(view.loc[view.segment_name == n, v].mean() - buyers[v].mean()) / buyers[v].std() for v in vars_]
                      for n in SEG_NAMES.values() if (view.segment_name == n).any()}, index=[LABELS.get(v, v) for v in vars_])
    st.plotly_chart(px.imshow(z, color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, text_auto=".2f", aspect="auto"), use_container_width=True)
    pca_png = ROOT / "reports" / "figures" / "04_pca_clusters.png"
    if pca_png.exists():
        st.image(str(pca_png), caption="PCA projection (visualisation only)")
    else:
        st.caption("PCA projection image not found. Run `python run_pipeline.py` to generate it.")
 
elif page == "Customer Intelligence":
    st.title("Customer Intelligence")
    q = st.text_input("Search client ID", placeholder="e.g. C0123").strip().upper()
    pick = st.selectbox("…or pick a client", buyers["client_id"].tolist(), index=0) if not q else q
    row_df = buyers[buyers["client_id"] == pick]
    if row_df.empty:
        st.error(f"Client '{pick}' not found.")
        st.stop()
    r = row_df.iloc[0]
    p = profiles["segments"][str(int(r["segment"]))]
    pb = segment_playbook(p, overall)
    a, b = st.columns(2)
    with a:
        st.subheader("Customer profile")
        st.table(pd.DataFrame({"Field": ["Client ID", "Age", "Country", "Region", "Client type", "Purpose", "Loan", "Satisfaction", "Referral"],
                               "Value": [r.client_id, int(r.age), r.country, r.region, r.client_type, r.acquisition_purpose, r.loan_applied,
                                         f"{int(r.satisfaction_score)}/5", r.referral_channel]}).set_index("Field"))
        st.subheader("Observed behaviour")
        st.table(pd.DataFrame({"Field": ["Purchase count", "Total spending", "Average purchase", "Property preference", "Distinct towers", "Purchase span"],
                               "Value": [int(r.purchase_count), f"${r.total_spend_usd:,.0f}", f"${r.average_purchase_price_usd:,.0f}",
                                         f"{r.apartment_purchase_count} apartment / {r.office_purchase_count} office",
                                         int(r.distinct_towers_purchased), f"{int(r.purchase_span_months)} months"]}).set_index("Field"))
    with b:
        st.subheader("ML segment (inference)")
        st.success(f"**{p['name']}**")
        if pd.notna(r.est_assignment_stability):
            st.caption(f"Estimated assignment stability: {r.est_assignment_stability:.0%} of resampled model refits keep buyers at this "
                       "distance from a segment boundary in the same segment (calibrated on the training buyers).")
        if r.low_fit_warning:
            st.warning("This client sits far from every segment centre; treat the segment as a weak match.")
        desc, chars = describe_segment(p, overall)
        st.write(desc)
        st.markdown("**Segment profile:** " + " · ".join(f"{k}: **{v}**" for k, v in pb["profile_levels"].items()))
    st.subheader("Customer insights")
    for line in pb["insights"]:
        st.markdown(f"- {line}")
    st.subheader("Recommended experience")
    recs = structured_recommendations(r, p, overall, float(r.est_assignment_stability))
    a_col, b_col = st.columns(2)
    with a_col:
        st.markdown("**Observed: from this client's own recorded data**")
        for rec in [x for x in recs if x["basis"] == "observed"] or []:
            render_rec(rec)
        if not any(x["basis"] == "observed" for x in recs):
            st.caption("No rule fired on this client's own recorded fields.")
    with b_col:
        st.markdown("**ML-derived inference: hypotheses to validate**")
        for rec in [x for x in recs if x["basis"] != "observed"]:
            render_rec(rec)
        if not any(x["basis"] != "observed" for x in recs):
            st.caption("Segment-based personalisation is withheld for this client (see the first action card for why).")
    with st.expander("Customer-value guardrails"):
        for g in pb["customer_value_guardrails"]:
            st.markdown(f"- {g}")
 
elif page == "New Customer Prediction":
    st.title("Assign a new customer to a segment")
    st.caption("Uses the saved preprocessing pipeline and centroids; nothing is retrained.")
    with st.form("new_customer"):
        c = st.columns(3)
        pc = c[0].number_input("Purchase count", 1, 30, 4)
        price = c[1].number_input("Average purchase price (USD)", 50_000, 2_000_000, 340_000, step=5_000)
        ppsf = c[2].number_input("Average price per sq ft (USD)", 100.0, 800.0, 300.0)
        c = st.columns(3)
        office = c[0].slider("Office share of units", 0.0, 1.0, 0.0, 0.05)
        span = c[1].number_input("Months between first and last purchase", 1, 24, 12)
        age = c[2].number_input("Age", 18, 100, 45)
        c = st.columns(4)
        purpose = c[0].selectbox("Purpose", ["Home", "Investment"])
        loan = c[1].selectbox("Loan applied", ["No", "Yes"])
        ctype = c[2].selectbox("Client type", ["Individual", "Company"])
        sat = c[3].slider("Satisfaction", 1, 5, 3)
        go_btn = st.form_submit_button("Assign segment")
    if go_btn:
        raw = dict(purchase_count=pc, average_purchase_price_usd=price, average_price_per_sqft_usd=ppsf, office_share=office,
                   purchase_span_months=span, age=age, acquisition_purpose=purpose, loan_applied=loan, client_type=ctype,
                   satisfaction_score=sat)
        res = scorer.score_new_customer(raw)
        st.success(f"Segment: **{res['segment_name']}**")
        if pd.notna(res["est_assignment_stability"]):
            st.caption(f"Estimated assignment stability {res['est_assignment_stability']:.0%} (share of resampled model refits that would keep a buyer "
                       "at this distance from a segment boundary in the same segment).")
        if res["low_fit_warning"]:
            st.warning("Inputs are outside the range seen in training (e.g. fewer than 3 purchases); the assignment is a weak match.")
        pb = segment_playbook(res["profile"], overall)
        for line in pb["insights"]:
            st.markdown(f"- {line}")
        st.subheader("Recommended experience")
        for rec in structured_recommendations(pd.Series(raw), res["profile"], overall, res["est_assignment_stability"]):
            render_rec(rec)
 
elif page == "Investment Behavior":
    st.title("Investment Behavior")
    a, b = st.columns(2)
    a.plotly_chart(px.box(view, x="segment_name", y="total_spend_usd", color="segment_name", color_discrete_map=COLOR, title="Total spend"), use_container_width=True)
    b.plotly_chart(px.box(view, x="segment_name", y="average_purchase_price_usd", color="segment_name", color_discrete_map=COLOR, title="Average purchase price"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(view, x="purchase_count", color="segment_name", color_discrete_map=COLOR, barmode="group", title="Purchase frequency (count)"), use_container_width=True)
    cat = view.groupby("segment_name")[["apartment_purchase_count", "office_purchase_count"]].sum().reset_index().melt("segment_name", var_name="Category", value_name="Units")
    b.plotly_chart(px.bar(cat, x="segment_name", y="Units", color="Category", barmode="group", title="Property category purchased"), use_container_width=True)
    pur = pd.crosstab(view["segment_name"], view["acquisition_purpose"], normalize="index").reset_index().melt("segment_name", value_name="Share")
    st.plotly_chart(px.bar(pur, x="segment_name", y="Share", color="acquisition_purpose", title="Investment vs home purpose by segment"), use_container_width=True)
    st.caption("Investment purpose is roughly the same in every behavioural segment: purpose is an individual attribute, not a segment trait.")
 
elif page == "Geographic Intelligence":
    st.title("Geographic Intelligence")
    by_c = view.groupby("country").size().reset_index(name="Buyers")
    by_c["iso"] = by_c["country"].map(ISO3)
    st.plotly_chart(px.choropleth(by_c, locations="iso", color="Buyers", hover_name="country", color_continuous_scale="Blues", title="Buyers by country"), use_container_width=True)
    a, b = st.columns(2)
    cs = pd.crosstab(view["country"], view["segment_name"], normalize="index").reset_index().melt("country", value_name="Share")
    a.plotly_chart(px.bar(cs, x="country", y="Share", color="segment_name", color_discrete_map=COLOR, title="Segment mix within each country"), use_container_width=True)
    top_regions = view["region"].value_counts().head(15).index
    rs = view[view["region"].isin(top_regions)].groupby(["region", "segment_name"]).size().reset_index(name="Buyers")
    b.plotly_chart(px.bar(rs, x="region", y="Buyers", color="segment_name", color_discrete_map=COLOR, title="Top 15 regions by segment"), use_container_width=True)
    st.metric("Share of buyers in the USA", pct((view["country"] == "USA").mean()))
    st.caption("Geography was not used to build segments (fairness decision); this page shows how segments are distributed geographically.")
 
elif page == "Customer Experience":
    st.title("Customer Experience")
    a, b = st.columns(2)
    sat = pd.crosstab(view["segment_name"], view["satisfaction_score"], normalize="index").reset_index().melt("segment_name", var_name="Score", value_name="Share")
    a.plotly_chart(px.bar(sat, x="segment_name", y="Share", color="Score", title="Satisfaction distribution by segment"), use_container_width=True)
    b.plotly_chart(px.bar(view.groupby("segment_name")["is_loan_applied"].mean().reset_index(), x="segment_name", y="is_loan_applied", title="Loan adoption by segment"), use_container_width=True)
    ref = pd.crosstab(view["segment_name"], view["referral_channel"], normalize="index").reset_index().melt("segment_name", value_name="Share")
    st.plotly_chart(px.bar(ref, x="segment_name", y="Share", color="referral_channel", title="Referral channel by segment"), use_container_width=True)
    st.subheader("Segment experience playbooks")
    for sid, p in profiles["segments"].items():
        pb = segment_playbook(p, overall)
        with st.expander(p["name"]):
            for title, key in [("Customer needs (hypotheses)", "customer_needs"), ("Potential friction", "potential_friction"),
                               ("Communication", "communication_strategy"), ("Engagement", "engagement_strategy")]:
                st.markdown(f"**{title}**")
                for x in pb[key]:
                    st.markdown(f"- {x}")
    low = view[view["satisfaction_score"] <= 2]
    st.metric("Clients with satisfaction 1-2 (service-recovery candidates)", f"{len(low):,}")
 
elif page == "Model Performance":
    st.title("Model Performance & Model Lab")
    s = reports["summary"]
    sel, fm, m = s.get("selected", {}), s.get("final_metrics", {}), reports["model_metrics"]
    if sel and fm:
        c = st.columns(6)
        c[0].metric("Algorithm", {"kmeans": "K-Means"}.get(sel["algorithm"], sel["algorithm"]))
        c[1].metric("Selected K", int(sel["k"]))
        c[2].metric("Silhouette", f"{fm['silhouette']:.3f}")
        c[3].metric("Davies-Bouldin", f"{fm['davies_bouldin']:.3f}")
        c[4].metric("Calinski-Harabasz", f"{fm['calinski_harabasz']:.0f}")
        c[5].metric("Subsample ARI", f"{sel['bootstrap_ari']:.2f} ± {sel['bootstrap_ari_sd']:.2f}")
        tbl = scorer.stability_table
        c = st.columns(3)
        c[0].metric("Seed-restart ARI", f"{sel['seed_ari']:.2f}")
        c[1].metric("Mean per-buyer assignment stability", f"{tbl['overall_mean_stability']:.0%}" if tbl else "n/a")
        c[2].metric("Smallest segment", f"{sel['min_cluster_share']:.1%}")
        gap = sel["silhouette_gap"]
        (st.warning if gap < 0.05 else st.info)(
            f"Silhouette on column-shuffled (structure-free) data is {sel['null_silhouette']:.3f}; the real-data gap is {gap:+.3f}. "
            + ("Cluster separation is weak: treat segments as practical groupings of a continuum, not natural populations."
               if gap < 0.05 else "Separation is clearly above the structure-free baseline."))
        st.caption(f"Selection score {sel['selection_score']:.3f} under the documented rule (src/selection.py). Only deployable algorithms are eligible to be selected; "
                   "other algorithms are benchmarks.")
    if m.empty:
        st.info("No model metrics found. Run `python run_pipeline.py`.")
    else:
        t1, t2, t3, t4, t5, t6 = st.tabs(["Cluster-count comparison", "K=4 vs K=5", "Selection sensitivity", "Algorithm benchmark",
                                           "Stability", "Features & fairness"])
        with t1:
            fs = st.selectbox("Feature set", m["feature_set"].unique(), index=0)
            algos = st.multiselect("Algorithms", sorted(m["algorithm"].unique()), default=["kmeans"])
            mm = m[(m["feature_set"] == fs) & (m["algorithm"].isin(algos))]
            metric = st.selectbox("Metric", [x for x in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia", "bootstrap_ari",
                                                          "seed_ari", "silhouette_gap", "min_cluster_share"] if x in mm.columns])
            if mm.empty or mm[metric].notna().sum() == 0:
                st.info("This metric is not defined for the chosen algorithms (e.g. inertia exists only for K-Means).")
            else:
                st.plotly_chart(px.line(mm.dropna(subset=[metric]), x="k", y=metric, color="algorithm", markers=True), use_container_width=True)
            km = m[(m["feature_set"] == fs) & (m["algorithm"] == "kmeans")]
            if "inertia" in km.columns and km["inertia"].notna().any():
                st.markdown("**Elbow (K-Means inertia)**")
                st.plotly_chart(px.line(km, x="k", y="inertia", markers=True), use_container_width=True)
            st.markdown("**Cluster-size balance**")
            bal = mm.melt(id_vars=["algorithm", "k"], value_vars=[c for c in ["min_cluster_share", "max_cluster_share"] if c in mm.columns])
            st.plotly_chart(px.line(bal, x="k", y="value", color="algorithm", line_dash="variable", markers=True), use_container_width=True)
            st.dataframe(mm.round(3), hide_index=True, use_container_width=True)
        with t2:
            kc = reports["k4_vs_k5"]
            if kc.empty:
                st.info("k4_vs_k5.csv not found.")
            else:
                kc = kc.rename(columns={kc.columns[0]: "metric"})
                st.dataframe(kc.round(3), hide_index=True, use_container_width=True)
                k45 = s.get("k4_vs_k5", {})
                if k45:
                    st.markdown(f"Partition agreement between the two solutions: **ARI {k45['partition_ari']:.2f}**. K=5 segment sizes: "
                                + ", ".join(f"{n:,}" for n in k45["k5_segment_sizes"]) + ".")
                    st.caption("Share of each K=4 segment that stays together in K=5: "
                               + ", ".join(f"segment {k}: {v:.0%}" for k, v in k45["share_in_dominant_child"].items()))
                st.info("The two solutions are close on the internal indices and within each other's stability noise. Which is preferable depends on "
                        "whether a small, behaviourally distinct segment is operationally worth a dedicated approach. This is a business judgement, "
                        "not something the metrics settle.")
        with t3:
            ss = reports["selection_sensitivity"]
            if ss.empty:
                st.info("selection_sensitivity.csv not found.")
            else:
                summ = s.get("selection_sensitivity", {})
                if summ:
                    st.markdown(f"Selected solution **{summ['baseline_winner']}** wins **{summ['share_of_scenarios_baseline_wins']:.0%}** of "
                                f"{summ['n_scenarios']} weight/constraint scenarios.")
                    st.plotly_chart(px.bar(x=list(summ["winner_counts"].keys()), y=list(summ["winner_counts"].values()),
                                           labels={"x": "winning solution", "y": "scenarios won"}), use_container_width=True)
                    if summ.get("unrestricted_winner_all_algorithms"):
                        st.caption(f"Best solution over ALL algorithms (benchmark, not deployable here): {summ['unrestricted_winner_all_algorithms']}.")
                st.dataframe(ss.drop(columns=["winner_algorithm", "winner_k", "winner_feature_set"], errors="ignore").round(3),
                             hide_index=True, use_container_width=True)
                alls = reports["selection_sensitivity_all_algorithms"]
                if not alls.empty:
                    with st.expander("Same scenarios over all algorithms (incl. GMM / hierarchical benchmarks)"):
                        st.dataframe(alls.drop(columns=["winner_algorithm", "winner_k", "winner_feature_set"], errors="ignore").round(3),
                                     hide_index=True, use_container_width=True)
        with t4:
            if sel:
                bm = m[(m["feature_set"] == sel["feature_set"]) & (m["k"] == int(sel["k"]))]
                cols = [c for c in ["algorithm", "silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "seed_ari",
                                    "bootstrap_ari", "fit_seconds"] if c in bm.columns]
                st.markdown(f"All algorithms at K = {int(sel['k'])}, feature set `{sel['feature_set']}`")
                st.dataframe(bm[cols].round(3), hide_index=True, use_container_width=True)
                st.caption("Seed ARI is 1.0 for deterministic algorithms by construction and is not evidence of quality. Hierarchical clustering has no "
                           "native assignment for unseen buyers, and a GMM would need its own scorer, so only K-Means is eligible for deployment here.")
        with t5:
            tbl = scorer.stability_table
            sa = reports["segment_assignments"]
            if sa.empty or "assignment_stability" not in sa.columns:
                st.info("Per-buyer assignment stability not available. Re-run the pipeline.")
            else:
                st.markdown("Per-buyer assignment stability = share of resampled K-Means refits that put the buyer in the same segment.")
                st.plotly_chart(px.histogram(sa, x="assignment_stability", nbins=20), use_container_width=True)
                st.dataframe(sa.groupby("segment_name")["assignment_stability"].agg(["mean", "min", "count"]).round(3).reset_index(),
                             hide_index=True, use_container_width=True)
            if tbl:
                cal = pd.DataFrame({"closeness-ratio bin": [f"{a:.2f}-{b:.2f}" for a, b in zip(tbl["bin_edges"][:-1], tbl["bin_edges"][1:])],
                                    "mean stability": tbl["mean_stability"], "buyers": tbl["n_buyers"]})
                st.markdown("**Calibration used for new buyers** (closeness ratio near 1 = near a segment boundary)")
                st.dataframe(cal.round(3), hide_index=True, use_container_width=True)
        with t6:
            for fig in ["02_model_selection.png", "03_dendrogram.png"]:
                if (ROOT / "reports/figures" / fig).exists():
                    st.image(str(ROOT / "reports/figures" / fig))
            st.subheader("Feature selection")
            st.dataframe(reports["feature_selection"], hide_index=True, use_container_width=True)
            st.subheader("Fairness audit (segment vs sensitive attribute)")
            st.dataframe(reports["fairness_audit"], hide_index=True, use_container_width=True)
            st.caption("Association tests describe whether segments over- or under-represent a group; they do not show the model is unbiased.")
            if s.get("gender_test"):
                st.caption(f"Adding gender to the clustering changes the partition very little (ARI {s['gender_test']['ari_with_vs_without_gender']:.2f}); "
                           "it is excluded and used only for auditing.")
 
elif page == "Methodology":
    st.title("Methodology")
    st.code("Data → Cleaning → Feature Engineering → Encoding → Scaling → Model Selection → Validation → Segmentation → Interpretation → Recommendations")
    _integ = reports["summary"].get("integrity", {})
    _rec = reports["data_reconciliation"]
    _bad = _rec.loc[_rec["status"] != "match", "column"].tolist() if not _rec.empty else []
    _note = (f" (it disagrees with the source on: {', '.join(_bad)})" if _bad else "")
    st.markdown("""
1. **Data & audit** – {n_clients:,} clients, {n_props:,} listings ({n_sold:,} sold). Missing `client_ref` = unsold listings, not errors. The buyer table is **rebuilt from the source clients and sold transactions**; the provided model-ready file is used only for reconciliation{recon_note}.
2. **Feature engineering** – span between first/last purchase, purchase intensity, tower diversification, international flag. No income, wealth or budget is created.
3. **Feature selection** – redundancy (|r| ≥ 0.85), near-constant check, leave-one-out sensitivity. Final clustering uses 5 behavioural features; age, satisfaction, loan, purpose, geography, gender are *profiled*, not clustered.
4. **Encoding & scaling** – log1p + StandardScaler for numeric features, winsorising of heavy tails, binaries unscaled, one-hot for categoricals in sensitivity runs.
5. **Model selection** – K-Means, Ward/complete/average hierarchical, GMM; K = 2…8; silhouette, Calinski-Harabasz, Davies-Bouldin, inertia (K-Means only), **null baseline**, seed and subsample stability. A weighted rule picks among *deployable* algorithms; its sensitivity to the weights and to the minimum-segment-size constraint is reported on the Model Performance page.
6. **Segmentation & interpretation** – one-vs-rest tests with effect sizes, Bonferroni-corrected; names generated from distinguishing traits.
7. **Recommendations** – rule-based; every recommendation lists its evidence, benefits, confidence, responsible rule, limitations and review need. Observed (client's own record) is kept separate from ML-derived inference; ambiguous assignments get a clarification action instead of personalisation.
8. **Fairness** – gender, geography, age and client type were tested for association with segment (Cramér's V) and excluded from clustering.
""".format(n_clients=_integ.get("unique_clients", 0), n_props=_integ.get("properties_total", 0),
           n_sold=_integ.get("properties_sold", 0), recon_note=_note))
    st.caption("Limitations: weak cluster separation; K=4 vs K=5 not separable by the metrics; cross-sectional data; no income/wealth data; currency/area units assumed (USD, sq ft); "
               "scoring requires purchase history; no business outcome (conversion, satisfaction change) has been measured.")
 

