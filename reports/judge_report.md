# Judge-Ready Report: Buyer Segmentation and Investment Profiling

Completed work is separated from proposed work. Every number below was produced by the pipeline in `reports/` or by the commands listed in section 9.

## 1. Original vs upgraded state

| Area | As found | Now |
|---|---|---|
| Buyer features | Used provided `buyer_features_model_ready.csv`; `office_share` (a clustering feature) differs from source for 893/2,000 buyers | Rebuilt from clients + sold transactions; reconciled against the provided file (27/28 columns identical) |
| Stability settings | Config said 20 seeds / 50 bootstraps; code ran 10 / 25 | Config values used as written |
| Selection | Hand-weighted rule, no sensitivity; could select a GMM but export a K-Means | Config-driven rule over deployable algorithms; 24-scenario sensitivity; explicit K=4 vs K=5 |
| Uncertainty | Relative-closeness "confidence" (0.31-0.44), not a probability | Per-buyer bootstrap assignment stability + calibration table for new buyers |
| Scoring input | NaN/negative/out-of-range values silently scored (NaN went to segment 0) | Validated; all problems reported in one error |
| Recommendations | Plain strings | Structured: action, evidence, benefits, confidence, rule, limitations, human review; observed vs inference |
| App text | Hardcoded "K-Means", "stability is good", counts | Computed from saved reports |
| Tests | 7 | 42 cases (+ a Streamlit AppTest file, not run here) |

## 2. Project structure
See `README.md` (layout section). Core modules: `data_processing`, `feature_engineering`, `clustering`, `evaluation`, `experiments`, `selection`, `profiling`, `recommendations`, `predict`; `run_pipeline.py`; `app/streamlit_app.py`; `config.yaml`.

## 3. Algorithms compared
K-Means, Ward / complete / average hierarchical, diagonal GMM; K=2..8; three feature sets (`model_metrics.csv`). Inertia is recorded for K-Means only. Hierarchical and GMM are benchmarks; only K-Means is eligible for deployment because the scorer is nearest-centroid.

## 4. Selected model and evidence
K-Means, K=5, 5 behaviour features (purchase count, average purchase price (log), price per sq ft, office share, purchase span).

| Metric | K=5 (selected) | K=4 |
|---|---|---|
| Silhouette | 0.173 | 0.167 |
| Null (shuffled) silhouette | 0.158 | 0.157 |
| Gap over null | +0.015 | +0.010 |
| Davies-Bouldin | 1.434 | 1.574 |
| Calinski-Harabasz | 389.5 | 398.8 |
| Smallest segment | 5.7% | 23.3% |
| Seed-restart ARI | 0.723 | 0.649 |
| Subsample ARI (mean, sd) | 0.818, 0.135 | 0.863, 0.105 |

**Weak separation:** the silhouette is only 0.015 above structure-free data. Segments are stable partitions of a continuum, not natural populations.

## 5. K=4 vs K=5 and selection sensitivity
- Under the baseline rule K=5 wins (score 0.791 vs 0.706). Across 24 weight/constraint scenarios K=4 wins 11, K=5 wins 10, K=3 wins 3.
- The result flips mainly on the minimum-segment-size constraint (5% -> K=5; 10% -> K=4), not on quality metrics. Metric differences are within resampling standard deviations.
- Partition agreement between K=4 and K=5: ARI 0.54. K=5 splits off a 114-buyer group (about 6.0 purchases each vs 3.3-3.7) from a K=4 segment.
- Over all algorithms, GMM K=3 scores best by a margin of 0.001 (benchmark only; not deployable here).
- **This is a business decision:** is a 5.7% frequent-buyer segment worth a dedicated approach? Set `selection.min_cluster_share: 0.10` to get K=4.

## 6. Segments (observed behaviour only)
| Segment | Buyers | Distinguishing observed behaviour |
|---|---|---|
| Long-Horizon Buyers | 534 (26.7%) | Longest purchase span (19.1 months), 3.7 purchases |
| Lower-Rate Value-Ticket Buyers | 489 (24.4%) | Lowest price per sq ft and average purchase ($283k) |
| Commercial-Leaning Premium-Rate Buyers | 452 (22.6%) | Office share 39% of purchases vs 15% overall |
| Fast-Cycle Residential-Focused Buyers | 411 (20.5%) | Shortest span (9.5 months), essentially no office units |
| High-Volume Long-Horizon Buyers | 114 (5.7%) | 6.0 purchases, more towers, high total spend |

Loan use, investment purpose and satisfaction do not differ materially by segment (so they are handled per client). Exploratory exception: the high-volume segment's mean satisfaction is 3.45 vs 3.00 elsewhere (raw p = 0.001, Bonferroni-adjusted p = 0.082, not significant). No income, wealth or "investor type" label is inferred. Names are generated from effect sizes; "Long-Horizon" and "High-Volume Long-Horizon" overlap and may merit a manual rename.

## 7. Example recommendations
- Client with satisfaction 2/5 and a loan: `OBS-LOW-SAT` (evidence: recorded score 2/5; human review recommended; promotion withheld), `OBS-LOAN` (evidence: recorded loan application).
- Buyer near a segment boundary (expected stability < 80%): `AMBIG-ASSIGN` (ask about preferences instead of personalising); no `SEG-*` actions are emitted.
- Stable buyer in a premium-ticket segment: `SEG-TICKET` (evidence: segment average vs overall average; labelled ML-derived inference with its stability estimate).
Gender, country/region, age and inferred wealth are never inputs (tested).

## 8. Assignment reliability
Mean per-buyer assignment stability 0.915; 89.2% of buyers at >= 0.80; 0.73 in the quintile closest to a segment boundary. The high-volume segment is the most stable (0.951).

## 9. Testing and exact commands
Run here (Python 3.12.3, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0):
- `python run_pipeline.py` (full grid) in a fresh copy and in the main copy: identical fingerprint `f14ca154c991bc8b`, centroids and assignments; metrics differ by <= 6e-14; sensitivity winners identical.
- 42 test cases across `test_pipeline`, `test_data_quality`, `test_selection`, `test_recommendations`, `test_predict_validation`, executed with a small stand-in for pytest (real pytest was unavailable: no network).
- All 9 app pages executed against real data and models with stub `streamlit`/`plotly` objects (logic only, not rendering).
- Notebooks 01-05 code cells run as plain Python (not executed as notebooks; no Jupyter available).
**Not run here:** real `pytest`, real Streamlit/Plotly rendering, `tests/test_app.py`, any dependency version other than those in `requirements-lock.txt`. Run `python -m pytest tests -q` and `streamlit run app/streamlit_app.py` once and report any failure.

## 10. Known limitations
Weak cluster separation; K=4 vs K=5 undecided by metrics; USD/sq ft units assumed (no unit metadata); every buyer has >= 3 purchases, so prospects without purchase history cannot be scored; cross-sectional data; the fairness audit shows negligible association (Cramer's V < 0.1) but does not prove the model is unbiased; the pipeline uses prints rather than logging; the app is a single file.

## 11. Evaluating business impact
See `reports/pilot_evaluation_plan.md` (proposed, not run): randomised pilot against the existing approach, pre-registered outcomes, guardrails by subgroup, human review for sensitive actions. The dataset has no contact, response or conversion data, so no impact claim can be made today.

## 12. Claims you can defend
- A reproducible pipeline: identical results from independent full runs.
- A data audit that found and fixed a defective clustering feature, with a reconciliation report.
- A transparent selection rule whose robustness was tested, with the K=4/K=5 uncertainty reported openly.
- The evaluated model is the deployed model; new-buyer assignment carries a data-calibrated stability estimate and rejects invalid input.
- Recommendations that are traceable to evidence and rules, separate observed facts from inference, and are guarded against use of sensitive attributes.

## Claims not supported
Natural or well-separated segments; improved satisfaction, conversion, retention or revenue; an unbiased model; statistical superiority of K=5 over K=4; any statement about buyer wealth or income.
