# Streamlit churn predictor

Predict telecom churn with the saved models in [`models/`](../models/).

## Run

From the **project root** (`data_mining_ucsy/`):

```bash
source venv/bin/activate
pip install -r requirements.txt   # includes streamlit
streamlit run streamlit_app/app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What you can do

| Tab | Purpose |
|-----|---------|
| **Single customer** | Visual input cards + quick preset profiles → churn probability + visual risk meter + behavioral diagnosis + prescriptive retention playbook |
| **Batch CSV** | Upload customer portfolio CSV (or download ready-to-test sample CSV) → bulk score accounts → download enriched predictions CSV |
| **Descriptive Data Mining (EDA)** | Interactive gallery of 10 exploratory data mining visualizations from `eda_figures/` with key takeaways |
| **Model Architectures & Pipelines** | Architecture diagrams from `diagrams/` for Stages 1–4 with pipeline specifications & master benchmark table |
| **Data Dictionary** | Searchable reference guide for all 38+ modeling features across 5 feature families |

| Sidebar Model Choice | Benchmark Metrics | File |
|----------------------|-------------------|------|
| **Stage 4 — Ensemble (XGBoost + LightGBM)** [Champion] | Acc: 63.49% · AUC: 0.6899 · Lift: 1.59× | `models/stage4_ensemble.joblib` |
| **Stage 3 — Tuned XGBoost (Rich Features)** | Acc: 63.27% · AUC: 0.6889 · Lift: 1.58× | `models/final_model.joblib` |
| **Stage 2 — Random Forest (Lean Features)** | Acc: 61.33% · AUC: 0.6658 · Lift: 1.47× | `models/before_random_forest_lean.joblib` |
| **Stage 1 — Logistic Regression (Baseline)** | Acc: 58.02% · AUC: 0.6090 · Lift: 1.27× | `models/before_logistic_regression_lean.joblib` |

## Files

- `app.py` — Streamlit UI
- `model_bundle.py` — `XGBBundle` + feature helpers (also used when saving models)
- `ui_defaults.json` — median/mode defaults and category dropdown options

Full bundler documentation: [`docs/model_bundler.md`](../docs/model_bundler.md)

## Note

If `final_model.joblib` fails to load with an `XGBBundle` / `__main__` error, re-export models using `model_bundle.XGBBundle` (already done for this project).
