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
| **Single customer** | Fill a form → churn probability + risk band |
| **Batch CSV** | Upload many rows → download predictions |

| Sidebar model | File |
|---------------|------|
| Final — XGBoost | `models/final_model.joblib` |
| Before — Random Forest | `models/before_random_forest_lean.joblib` |
| Before — Logistic Regression | `models/before_logistic_regression_lean.joblib` |

## Files

- `app.py` — Streamlit UI
- `model_bundle.py` — `XGBBundle` + feature helpers (also used when saving models)
- `ui_defaults.json` — median/mode defaults and category dropdown options

Full bundler documentation: [`docs/model_bundler.md`](../docs/model_bundler.md)

## Note

If `final_model.joblib` fails to load with an `XGBBundle` / `__main__` error, re-export models using `model_bundle.XGBBundle` (already done for this project).
