# Model Bundler — Documentation

This document explains [`streamlit_app/model_bundle.py`](../streamlit_app/model_bundle.py): what it is, why it exists, what each piece does, and how to use it with the saved models in [`models/`](../models/).

Related:

- Streamlit UI: [`streamlit_app/app.py`](../streamlit_app/app.py)
- Saved models: [`models/README.md`](../models/README.md)
- Model chronology: [`model_evolution.md`](model_evolution.md)
- Feature engineering: [`feature_engineering.md`](feature_engineering.md)

---

## 1. What is the “model bundler”?

The bundler is a small Python module that packages **everything needed to score a customer** into one loadable object.

For the **final** model (tuned XGBoost), that means:

| Contained piece | Purpose |
|-----------------|--------|
| Trained XGBoost classifier | Outputs churn probability |
| `ColumnTransformer` preprocessor | Median impute numerics + one-hot categoricals |
| Train-fitted clip bounds | Cap extreme tails on selected columns |
| Numeric / categorical feature lists | Exact column order expected at score time |
| Decision threshold (default `0.5`) | Turn probability into stay/churn |

The class that holds this package is **`XGBBundle`**.

Saved files that use it:

- `models/after_xgboost_tuned_rich.joblib`
- `models/final_model.joblib` (alias of the above)

The older **before-improvement** models (Logistic Regression, Random Forest) are plain **sklearn `Pipeline`** objects. They do not use `XGBBundle`, but this module still provides helpers (`lean_feature_frame`) so the Streamlit app can build their inputs.

---

## 2. Why it exists

### Problem

When the final XGBoost model was first saved, the wrapper class was defined inside a one-off training script (`__main__`).  
`joblib` stores the **class path** with the object. Loading later failed with:

```text
AttributeError: module '__main__' has no attribute 'XGBBundle'
```

### Fix

Define `XGBBundle` in a real importable module:

```text
streamlit_app/model_bundle.py
```

Re-save the model so pickle points at `model_bundle.XGBBundle`. Then any app or script that puts `streamlit_app/` on `sys.path` (or imports the package) can:

```python
import joblib
model = joblib.load("models/final_model.joblib")
proba = model.predict_proba(X_df)[:, 1]
```

### Extra benefit

Callers do not need to remember clip → select columns → `preprocessor.transform` → `model.predict_proba`. The bundle does that sequence inside `predict_proba` / `predict`.

---

## 3. What’s inside the module

```text
streamlit_app/model_bundle.py
├── XGBBundle                 # save/load wrapper for final XGBoost
├── engineer_rich_features()  # derive rich FE columns from raw-ish inputs
└── lean_feature_frame()      # build lean columns for before RF / LR pipelines
```

---

## 4. `XGBBundle` — API

### Constructor fields

| Attribute | Type | Meaning |
|-----------|------|---------|
| `preprocessor` | sklearn `ColumnTransformer` | Fitted impute + one-hot |
| `model` | `XGBClassifier` | Fitted booster |
| `clip_bounds` | `dict[str, (lo, hi)]` | 1st–99th pct bounds from **train** |
| `numeric` | `list[str]` | Numeric feature names in order |
| `categorical` | `list[str]` | Categorical feature names in order |
| `threshold` | `float` | Default `0.5` for hard labels |

### Methods

#### `_prepare(X)`

Internal. Given a DataFrame:

1. Clip columns listed in `clip_bounds` (if present).
2. Cast categoricals to string and fill nulls with `"Missing"`.
3. Return only `numeric + categorical` columns in the training order.

#### `predict_proba(X)`

1. `_prepare(X)`
2. `preprocessor.transform(...)`
3. `model.predict_proba(...)` → shape `(n_rows, 2)`; column 1 is \(P(\text{churn}=1)\).

#### `predict(X)`

Returns `1` if `predict_proba` ≥ `threshold`, else `0`.

### Typical usage

```python
import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path("streamlit_app").resolve()))
# optional: from model_bundle import engineer_rich_features

model = joblib.load("models/final_model.joblib")

# X_df must include the rich feature columns (see manifest.json)
proba = model.predict_proba(X_df)[:, 1]
label = model.predict(X_df)
```

---

## 5. Scoring pipeline (what happens when you call predict)

```text
Input DataFrame (customer features)
        │
        ▼
  Clip heavy tails          (train bounds: change_mou, overage_ratio, …)
        │
        ▼
  Categorical cleanup       (string + "Missing")
        │
        ▼
  Column select             (exact numeric + categorical lists)
        │
        ▼
  ColumnTransformer         (median impute + one-hot)
        │
        ▼
  XGBoost                   (probability of churn)
        │
        ▼
  Optional threshold        (0/1 class)
```

This mirrors training in [`modeling.ipynb`](../modeling.ipynb), so Streamlit scores match notebook scores for the same rows.

---

## 6. Helper: `engineer_rich_features(df)`

Builds **derived** columns used by the after-improvement (rich) model when inputs exist.

| Derived column | Built from | Meaning |
|----------------|------------|---------|
| `eqpdays` floored at 0 | `eqpdays` | No negative equipment age |
| `mou_decline_flag` | `change_mou < 0` | Usage falling? |
| `overage_ratio` | `ovrrev_Mean / \|totmrc_Mean\|` | Overage vs base plan |
| `eqpdays_x_change_mou` | product | Handset age × usage change |
| `mou_per_month` | `mou_Mean / months` | Usage intensity |
| `rev_per_mou` | `rev_Mean / mou_Mean` | Revenue density |
| `drop_rate` | `drop_vce_Mean / plcd_vce_Mean` | Dropped-call rate |
| `care_per_mou` | `custcare_Mean / mou_Mean` | Care intensity |
| `eqpdays_bin` | binned `eqpdays` | `0_6m`, `6_12m`, `1_2y`, `2y_plus` |

If a derived column is already present, some branches skip recomputation; interaction/ratio fields are generally recomputed when inputs exist so the UI stays consistent.

**Used by:** Streamlit single-customer and batch tabs before calling the final model.

---

## 7. Helper: `lean_feature_frame(df)`

1. Calls `engineer_rich_features` (so `mou_decline_flag` / `overage_ratio` exist).
2. Selects the **lean** column set used by before-improvement LR / RF.
3. Formats categoricals as string with `"Missing"`.

**Used by:** Streamlit when the sidebar model is “Before — Random Forest” or “Before — Logistic Regression”.

Lean columns match [`models/manifest.json`](../models/manifest.json) → `before_improvement.lean_numeric` / `lean_categorical`.

---

## 8. How this ties to saved files

| File | Wrapper type | Needs bundler module? |
|------|--------------|------------------------|
| `final_model.joblib` | `XGBBundle` | **Yes** — must import `model_bundle` |
| `after_xgboost_tuned_rich.joblib` | `XGBBundle` | **Yes** |
| `before_random_forest_lean.joblib` | sklearn `Pipeline` | No (helpers optional) |
| `before_logistic_regression_lean.joblib` | sklearn `Pipeline` | No |

If you move or rename `model_bundle.py` without re-saving, loads of `final_model.joblib` will break. After any class rename/move, **re-export** the joblib files.

---

## 9. Relationship to Streamlit

[`streamlit_app/app.py`](../streamlit_app/app.py) does:

1. `sys.path.insert(0, streamlit_app/)` so `import model_bundle` works.
2. `joblib.load(...)` the chosen model.
3. For final model: engineer features → ensure columns → `XGBBundle.predict_proba`.
4. For before models: `lean_feature_frame` → `Pipeline.predict_proba`.

Defaults for empty form fields live in [`streamlit_app/ui_defaults.json`](../streamlit_app/ui_defaults.json) (not in the bundler).

---

## 10. Re-saving a bundle (maintainers)

When retraining the final model:

1. Import `XGBBundle` from `model_bundle` (not define a local class).
2. Fit preprocessor + XGBoost as in the modeling notebook.
3. Build:

```python
bundle = XGBBundle(
    preprocessor=rich_pre,
    model=xgb_model,
    clip_bounds=rich_bounds,
    numeric=RICH_NUM,
    categorical=RICH_CAT,
    threshold=0.5,
)
joblib.dump(bundle, "models/final_model.joblib")
joblib.dump(bundle, "models/after_xgboost_tuned_rich.joblib")
```

4. Smoke-test in a **new** Python process:

```python
joblib.load("models/final_model.joblib").predict_proba(sample_df)
```

---

## 11. What the bundler is *not*

| Not responsible for | Handled elsewhere |
|---------------------|-------------------|
| Training / hyperparameter search | `modeling.ipynb` |
| EDA plots | `main.ipynb` |
| Streamlit widgets / layout | `app.py` |
| UI default medians / dropdown lists | `ui_defaults.json` |
| Business ROI narrative | docs / proposal write-up |

It is only the **score-time adapter** between a DataFrame and the saved final model.

---

## 12. One-sentence summary

**The model bundler (`XGBBundle` + feature helpers) packages the final XGBoost model with its preprocessor, clip rules, and column lists so churn scores can be loaded and run reliably from Streamlit or any script without redoing training transforms by hand.**
