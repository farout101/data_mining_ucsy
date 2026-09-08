# Saved trained models

Training happens in [`modeling.ipynb`](../modeling.ipynb).  
Until this folder existed, models lived **only in notebook memory** — they were not saved as files.  
They are now exported here with `joblib`.

Chronology write-up: [`model_evolution.md`](../docs/model_evolution.md)  
Bundler docs: [`model_bundler.md`](../docs/model_bundler.md)

---

## Quick map

| When | File | What it is | Test accuracy | ROC-AUC |
|------|------|------------|---------------|---------|
| **Before improvement** | `before_logistic_regression_lean.joblib` | Stage 1 — Logistic Regression (lean FE) | ~0.580 | ~0.609 |
| **Before improvement (primary then)** | `before_random_forest_lean.joblib` | Stage 2 — Random Forest (lean FE) | ~0.613 | ~0.666 |
| same | `before_primary_random_forest_lean.joblib` | Alias of the RF above | same | same |
| **After improvement (final)** | `after_xgboost_tuned_rich.joblib` | Stage 3 — tuned XGBoost (rich FE) | ~0.633 | ~0.689 |
| **Final alias** | `final_model.joblib` | Same as after XGBoost — **use this** | ~0.633 | ~0.689 |

Machine-readable details: [`manifest.json`](manifest.json)

---

## How to load

### Before — sklearn Pipelines (LR / RF)

```python
import joblib

lr = joblib.load("models/before_logistic_regression_lean.joblib")
rf = joblib.load("models/before_random_forest_lean.joblib")

# X_df must have the lean columns expected by the pipeline
proba = rf.predict_proba(X_df)[:, 1]
pred = rf.predict(X_df)
```

### After / final — XGBoost bundle

```python
import joblib

model = joblib.load("models/final_model.joblib")  # or after_xgboost_tuned_rich.joblib

proba = model.predict_proba(X_df)[:, 1]
pred = model.predict(X_df)
```

The final bundle includes preprocessor, clip bounds, and feature lists.  
Pass a DataFrame with the **rich** feature columns (see `manifest.json`).

---

## Notes

- Same stratified split recipe: `test_size=0.2`, `random_state=42`
- “Before” = lean feature set; “After” = richer FE + tuned XGBoost
- Re-run the export script / notebook training if you change features or seeds
