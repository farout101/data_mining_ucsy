# Telecom Churn Prediction — Company A

University data-mining project: explore anonymized telecom customer data, engineer features, train churn models, and demo predictions in a Streamlit UI.

**Primary model:** tuned XGBoost (~**63.3%** holdout accuracy, ROC-AUC ~**0.69**, top-10% risk lift ~**1.58×**).

> Accuracy plateaus near ~0.6 because leavers and stayers overlap in the available features — see [`docs/why_accuracy_plateau.md`](docs/why_accuracy_plateau.md). The model is most useful as a **risk ranking / targeting** tool, not a perfect crystal ball.

---

## Project Structure

```
data_mining_ucsy/
├── data/                    # Data directory
│   └── telecom/            # Raw data files (gitignored)
│       ├── Client.csv
│       └── Record.csv
├── notebooks/              # Jupyter notebooks
│   ├── main.ipynb         # Exploratory data analysis (EDA)
│   ├── modeling.ipynb     # Feature engineering + model training
│   └── residue.ipynb      # EDA takeaways
├── src/                    # Python source modules
│   ├── __init__.py
│   └── model_bundle.py    # Model bundler utilities
├── models/                 # Saved trained models (gitignored)
│   ├── final_model.joblib
│   └── manifest.json
├── streamlit_app/          # Streamlit UI application
│   ├── app.py
│   ├── model_bundle.py
│   └── README.md
├── config/                 # Configuration files
│   └── ui_defaults.json
├── docs/                   # Documentation
├── eda_figures/            # Exported EDA plots
├── tests/                  # Unit tests (optional)
├── pyproject.toml         # Package configuration
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
└── .gitattributes        # Git attributes for file handling
```

---

## What's in this repo

| Path | Purpose |
|------|---------|
| [`notebooks/main.ipynb`](notebooks/main.ipynb) | Exploratory data analysis (EDA) |
| [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb) | Feature engineering + LogReg → RF → XGBoost |
| [`notebooks/residue.ipynb`](notebooks/residue.ipynb) | Short EDA takeaways |
| [`data/telecom/`](data/telecom/) | `Client.csv` + `Record.csv` (local data; gitignored) |
| [`models/`](models/) | Saved trained models (`.joblib`) |
| [`streamlit_app/`](streamlit_app/) | Churn prediction UI |
| [`src/`](src/) | Reusable Python modules |
| [`config/`](config/) | Configuration files |
| [`docs/`](docs/) | Detailed write-ups (EDA, FE, ML, evolution, bundler, …) |
| [`eda_figures/`](eda_figures/) | Exported EDA plots |
| [`requirements.txt`](requirements.txt) | Python dependencies |

---

## Problem & data

- **Task:** predict whether a customer will **churn** (leave) 31–60 days after the observation date.
- **Tables:** `Client.csv` (profile / lifetime) ⋈ `Record.csv` (recent usage + `churn`) on `Customer_ID` → **100,000** customers × ~100 columns.
- **Label balance:** ~50/50 in this extract (convenient for accuracy; atypical vs many live carriers).
- **Download:** [Telecom Customer Churn 100K — cleaned records (Kaggle)](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records)

Place the CSVs under `data/telecom/` as:

```text
data/telecom/Client.csv
data/telecom/Record.csv
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

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

Place data at:

```text
data/telecom/Client.csv
data/telecom/Record.csv
```

Dataset download (Kaggle):  
https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records

### Run notebooks

```bash
jupyter notebook notebooks/main.ipynb      # EDA
jupyter notebook notebooks/modeling.ipynb  # FE + training
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

sys.path.insert(0, str(Path("src").resolve()))
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
| [`docs/kaggle_notes_vs_accuracy.md`](docs/kaggle_notes_vs_accuracy.md) | Kaggle "anomalies" vs our accuracy experiments |
| [`docs/telecom_column_dictionary.md`](docs/telecom_column_dictionary.md) | Column dictionary |

---

## Project pipeline

```text
Client.csv + Record.csv
        │
        ▼
   EDA (notebooks/main.ipynb)
        │
        ▼
   Feature engineering (lean → rich)
        │
        ▼
   Train / compare models (notebooks/modeling.ipynb)
        │
        ├─ before: LogReg, Random Forest  → models/before_*.joblib
        └─ after:  tuned XGBoost          → models/final_model.joblib
        │
        ▼
   Streamlit UI (streamlit_app/app.py)
```

---

## Tech stack

- Python 3.8+, pandas, matplotlib/seaborn  
- scikit-learn, XGBoost  
- Jupyter  
- Streamlit + joblib  

---

## Notes for graders / readers

1. **EDA before modeling** — join audit, missingness, churn balance, segments, correlations.  
2. **Lean features first** — avoid dumping all redundant MOU/revenue windows.  
3. **Primary metric story** — report accuracy *and* AUC / top-decile lift.  
4. **Honest ceiling** — ~63% accuracy is expected given class overlap; see the plateau note.  
5. **Gitignore:** `data/telecom/` (raw CSVs) and `models/` (large `.joblib` files) are ignored by default — keep them locally; download data from [Kaggle](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records) and re-run training/export if cloning a bare repo.

---

## Installation

### As a package

```bash
pip install -e .
```

### Development setup

```bash
pip install -e ".[dev]"
```

---