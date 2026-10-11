# Real-Estate Buyer Intelligence

Machine-learning buyer segmentation and investment profiling for a real-estate portfolio:
rebuild buyer features from source data, compare clustering models, select a segmentation with a
documented and stress-tested rule, explain each segment, assign new buyers without retraining, and
turn observed behaviour into evidence-backed customer-experience actions.

> **Honest summary.** Buyer behaviour in this dataset has little natural cluster structure
> (silhouette 0.173 vs 0.158 on structure-free shuffled data). The segments are stable, *useful partitions
> of a continuum*, not sharply separated populations. K=4 and K=5 cannot be separated by the metrics.
> No business outcome has been measured; recommendations are hypotheses until piloted
> (see `reports/pilot_evaluation_plan.md`).

## Workflow

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # exact validated versions: requirements-lock.txt

python run_pipeline.py                  # 1. train: audit -> features -> model grid -> selection -> artifacts -> reports (~5-10 min)
python run_pipeline.py --reuse-grid     #    faster rerun that reuses reports/model_metrics.csv (~1.5 min)
python -m pytest tests -q               # 2. tests (tests/test_app.py needs streamlit installed)
streamlit run app/streamlit_app.py      # 3. app (loads saved artifacts; never retrains)
```

Verify after a run: `models/` has 4 pkl/json artifacts plus `training_metadata.json` and `assignment_stability.json`;
`reports/model_report.md` states the selected model, its robustness and the data-reconciliation finding;
the console prints a `Fingerprint:` that must be identical on every rerun with the same data and config.

## Project layout

```
config.yaml            all tunables (seed, K range, stability replications, selection rule)
run_pipeline.py        end-to-end training/evaluation pipeline
src/data_processing    loading, audit, integrity checks, load_buyer_table (rebuild + reconcile)
src/feature_engineering  rebuild_buyer_table, derived features, preprocessing pipeline
src/clustering         K-Means / hierarchical / GMM wrappers, frozen SegmentModel
src/evaluation         metrics, seed/subsample stability, per-buyer assignment stability
src/experiments        model grid (feature sets x algorithms x K) with null baseline
src/selection          weighted selection rule, sensitivity analysis, K comparison
src/profiling          segment profiles, statistical tests, name generation, fairness audit
src/recommendations    segment playbook + structured, evidence-backed recommendations
src/predict            SegmentScorer: input validation, scoring, stability estimate
app/streamlit_app.py   9-page application
models/                preprocessing pipeline, clustering model, profiles, metadata, calibration
reports/               metrics, profiles, sensitivity, reconciliation, model/insight/pilot reports
tests/                 data quality, selection, recommendations, scoring validation, pipeline, app
```

## What the pipeline does

1. **Data audit and rebuild.** The buyer table is rebuilt from `clients_cleaned.csv` and the *sold* rows of
   `properties_cleaned.csv`. It is reconciled against the provided `buyer_features_model_ready.csv`: 27 of 28
   shared columns match exactly; `office_share` does not (893 of 2,000 buyers differ), so the rebuilt value is used.
   Currency (USD) and area (sq ft) are assumed from column names; the source files carry no unit metadata.
2. **Features.** Five behavioural features are clustered (purchase count, average purchase price (log), price per sq ft,
   office share, purchase span). Demographics, geography, financing, purpose and satisfaction are *profiled*, not
   clustered. Gender, country and region are used only in the fairness audit. Every include/exclude decision is in
   `reports/feature_selection.csv`. No income, wealth, budget or credit variable is created or implied.
3. **Models.** K-Means, Ward/complete/average hierarchical, diagonal GMM; K=2..8; three feature sets. Metrics:
   silhouette, Calinski-Harabasz, Davies-Bouldin, inertia (K-Means only), a column-shuffled null baseline, seed
   stability (20 restarts) and subsample stability (50 refits on 80% of buyers), fit time, cluster-size balance.
4. **Selection.** Weighted rule over *deployable* algorithms (K-Means), K in 3..6, smallest segment >= 5%.
   Re-run under 24 alternative weight/constraint scenarios (`reports/selection_sensitivity.csv`) and compared
   explicitly for K=4 vs K=5. The best solution over all algorithms (a GMM) is reported but not deployed, because
   the evaluated model must be the shipped model.
5. **Scoring a new buyer.** Validated input -> saved preprocessing -> nearest centroid -> segment, plus an
   estimated assignment stability calibrated on training buyers. Missing, non-numeric, non-finite or out-of-range
   fields are rejected with a clear message; out-of-distribution inputs are flagged.
6. **Recommendations.** Each one lists action, evidence, customer benefit, business benefit, confidence, the rule that
   produced it, limitations and human-review need, and is labelled *observed* (the client's own record) or
   *ML-derived inference*. Low-satisfaction clients get a service-recovery review and no promotion; ambiguous
   assignments get a clarification action instead of personalisation. Gender, geography, age and inferred wealth are never inputs.

## Known limitations

- Weak cluster separation; K=4 vs K=5 is partly a business judgement (is a 114-buyer, 5.7% frequent-buyer segment worth a dedicated approach?).
- `streamlit`/`plotly` rendering and real `pytest` were not executed in the build environment (see `reports/judge_report.md`).
- Cross-sectional data; every buyer has >= 3 purchases, so prospects without purchase history cannot be scored.
- Satisfaction differences between segments are exploratory (not significant after correction).
- Dependencies: only the versions in `requirements-lock.txt` were run; model pickles should be regenerated (`python run_pipeline.py`) in your own environment.
