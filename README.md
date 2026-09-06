# Telecom Churn Prediction — Company A

University data-mining project: explore anonymized telecom customer data, engineer features, train churn models, and demo predictions in a Streamlit UI.

**Primary model:** tuned XGBoost (~**63.3%** holdout accuracy, ROC-AUC ~**0.69**, top-10% risk lift ~**1.58×**).

> Accuracy plateaus near ~0.6 because leavers and stayers overlap in the available features — see [`docs/why_accuracy_plateau.md`](docs/why_accuracy_plateau.md). The model is most useful as a **risk ranking / targeting** tool, not a perfect crystal ball.

---

## What’s in this repo

| Path | Purpose |
|------|---------|
| [`main.ipynb`](main.ipynb) | Exploratory data analysis (EDA) |
| [`modeling.ipynb`](modeling.ipynb) | Feature engineering + LogReg → RF → XGBoost |
| [`residue.ipynb`](residue.ipynb) | Short EDA takeaways |
| [`telecom/`](telecom/) | `Client.csv` + `Record.csv` (local data; often gitignored) |
| [`models/`](models/) | Saved trained models (`.joblib`) |
| [`streamlit_app/`](streamlit_app/) | Churn prediction UI |
| [`docs/`](docs/) | Detailed write-ups (EDA, FE, ML, evolution, bundler, …) |
| [`eda_figures/`](eda_figures/) | Exported EDA plots |
| [`requirements.txt`](requirements.txt) | Python dependencies |

---

## Problem & data

- **Task:** predict whether a customer will **churn** (leave) 31–60 days after the observation date.
- **Tables:** `Client.csv` (profile / lifetime) ⋈ `Record.csv` (recent usage + `churn`) on `Customer_ID` → **100,000** customers × ~100 columns.
- **Label balance:** ~50/50 in this extract (convenient for accuracy; atypical vs many live carriers).
- **Download:** [Telecom Customer Churn 100K — cleaned records (Kaggle)](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records)

Place the CSVs under `telecom/` as:

```text
telecom/Client.csv
telecom/Record.csv
```

Column meanings: [`docs/telecom_column_dictionary.md`](docs/telecom_column_dictionary.md).

---

## Model evolution (summary)

| Stage | Model | Features | Accuracy | ROC-AUC |
|-------|--------|----------|----------|---------|
| 1 | Logistic Regression | Lean | ~0.58 | ~0.61 |
| 2 | Random Forest | Lean | ~0.61 | ~0.67 |
| 3 — **final** | **XGBoost (tuned)** | Richer FE | ~**0.63** | ~**0.69** |

Full chronology: [`docs/model_evolution.md`](docs/model_evolution.md).

### Saved model files

| File | Role |
|------|------|
| `models/before_logistic_regression_lean.joblib` | Stage 1 baseline |
| `models/before_random_forest_lean.joblib` | Stage 2 |
| `models/final_model.joblib` | **Stage 3 — use this** |
| `models/after_xgboost_tuned_rich.joblib` | Same as final |

Details: [`models/README.md`](models/README.md) · bundler: [`docs/model_bundler.md`](docs/model_bundler.md).

---

## Quick start

```bash
# clone / enter repo
cd data_mining_ucsy

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place data at:

```text
telecom/Client.csv
telecom/Record.csv
```

Dataset download (Kaggle):  
https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records

### Run notebooks

```bash
jupyter notebook main.ipynb      # EDA
jupyter notebook modeling.ipynb  # FE + training
```

### Run Streamlit demo

```bash
streamlit run streamlit_app/app.py
```

UI docs: [`streamlit_app/README.md`](streamlit_app/README.md).

### Score with the final model (Python)

```python
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path("streamlit_app").resolve()))
model = joblib.load("models/final_model.joblib")
# X_df = DataFrame with rich feature columns (see models/manifest.json)
proba = model.predict_proba(X_df)[:, 1]
```

---

## Documentation index

| Doc | Topic |
|-----|--------|
| [`docs/eda_approach_rationale.md`](docs/eda_approach_rationale.md) | Why each EDA step |
| [`docs/eda_why_this_order.md`](docs/eda_why_this_order.md) | Why that EDA order |
| [`docs/explain_eda_simple.md`](docs/explain_eda_simple.md) | EDA in plain language |
| [`docs/feature_engineering.md`](docs/feature_engineering.md) | Feature engineering detail |
| [`docs/machine_learning.md`](docs/machine_learning.md) | ML detail |
| [`docs/model_evolution.md`](docs/model_evolution.md) | LogReg → RF → XGB record |
| [`docs/explain_ml_fe_simple.md`](docs/explain_ml_fe_simple.md) | FE + ML in plain language |
| [`docs/model_bundler.md`](docs/model_bundler.md) | `XGBBundle` / score-time wrapper |
| [`docs/why_accuracy_plateau.md`](docs/why_accuracy_plateau.md) | Why accuracy ~0.6 |
| [`docs/kaggle_notes_vs_accuracy.md`](docs/kaggle_notes_vs_accuracy.md) | Kaggle “anomalies” vs our accuracy experiments |
| [`docs/telecom_column_dictionary.md`](docs/telecom_column_dictionary.md) | Column dictionary |

---

## Project pipeline

```text
Client.csv + Record.csv
        │
        ▼
   EDA (main.ipynb)
        │
        ▼
   Feature engineering (lean → rich)
        │
        ▼
   Train / compare models (modeling.ipynb)
        │
        ├─ before: LogReg, Random Forest  → models/before_*.joblib
        └─ after:  tuned XGBoost          → models/final_model.joblib
        │
        ▼
   Streamlit UI (streamlit_app/app.py)
```

---

## Tech stack

- Python 3, pandas, matplotlib/seaborn  
- scikit-learn, XGBoost  
- Jupyter  
- Streamlit + joblib  

---

## Notes for graders / readers

1. **EDA before modeling** — join audit, missingness, churn balance, segments, correlations.  
2. **Lean features first** — avoid dumping all redundant MOU/revenue windows.  
3. **Primary metric story** — report accuracy *and* AUC / top-decile lift.  
4. **Honest ceiling** — ~63% accuracy is expected given class overlap; see the plateau note.  
5. **Gitignore:** `telecom/` (raw CSVs) and `models/` (large `.joblib` files) are ignored by default — keep them locally; download data from [Kaggle](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records) and re-run training/export if cloning a bare repo.

---

## License / data

Course-style anonymized telecom extract (“Company A”). Respect any data-protection terms from your instructor; do not redistribute restricted files if your course forbids it.
