# Model Evolution Record — Logistic Regression → Random Forest → XGBoost

This document is the **project log** of how modeling changed, **why** each step happened, and **what the numbers were** after each change.

Related files:

- Code: [`modeling.ipynb`](modeling.ipynb)
- Feature engineering: [`feature_engineering.md`](feature_engineering.md)
- ML overview: [`machine_learning.md`](machine_learning.md)
- EDA takeaways: [`residue.ipynb`](residue.ipynb)

---

## 0. Timeline (short)

| Stage | Model | Features | Test accuracy @0.5 | ROC-AUC | Top-10% churn (lift) |
|-------|--------|----------|--------------------|---------|----------------------|
| **1 — Baseline** | Logistic Regression | Lean (~25) | **0.5802** | **0.6090** | 63.1% (**1.27×**) |
| **2 — First upgrade** | Random Forest | Lean (~25) | **0.6133** | **0.6658** | 72.9% (**1.47×**) |
| **3 — Improved (current primary)** | **XGBoost (tuned)** | Richer FE | **0.6327** | **0.6889** | 78.2% (**1.58×**) |

Same data, same stratified 80/20 split (`random_state=42`), same target `churn`.

---

## 1. Stage 1 — Logistic Regression (what happened)

### What we did

- Built a lean, EDA-driven feature set (~17 numeric + 8 categorical).
- Preprocessed with median imputation, `StandardScaler` for numerics, one-hot categoricals (`"Missing"` for nulls).
- Trained `LogisticRegression(max_iter=1000)` as the first / interpretable model.

### Results (holdout test, n=20,000)

| Metric | Value |
|--------|-------|
| Accuracy | 0.5802 (58.0%) |
| ROC-AUC | 0.6090 |
| Precision | 0.5762 |
| Recall | 0.5786 |
| F1 | 0.5774 |
| Top 10% risk — actual churn | ~63.1% |
| Lift vs random | ~1.27× |

### Why this was not enough

1. **Accuracy only ~8 points above a coin flip / majority baseline (~50%).** Better than chance, but weak as a “solved” classifier.
2. **EDA already showed weak linear correlation with churn** (`|r|` max ≈ 0.11 for `eqpdays`). Logistic regression mainly captures linear / additive effects.
3. Churners and stayers **overlap heavily** in revenue/usage space (pairplots). A linear boundary cannot separate them well.
4. Coefficients were still useful for storytelling (older `eqpdays` → higher churn), but **ranking quality (AUC 0.61)** was only modestly above random (0.50).

### Verdict on Stage 1

Keep logistic regression as the **interpretability counterpart / baseline**, but do **not** stop here for the primary predictive model.

---

## 2. Stage 2 — Why we changed to Random Forest

### Motivation (explicit)

| Problem with LR | Why RF addresses it |
|-----------------|---------------------|
| Weak linear signal | Trees learn thresholds and interactions (e.g. old handset **and** usage drop) |
| Needs careful scaling | Trees are scale-insensitive |
| Modest AUC | Ensemble of trees usually ranks better on tabular churn data |
| Assignment needs a defensible upgrade | Compare RF vs LR on the **same** features to prove the gain is from the algorithm, not from secret new columns |

### What we did

- Same lean feature set as LR (fair head-to-head).
- `RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, random_state=42)`.
- Median impute + one-hot (no scaling).

### Results

| Metric | Logistic Reg | Random Forest | Change |
|--------|--------------|---------------|--------|
| Accuracy | 0.5802 | **0.6133** | **+3.3 pp** |
| ROC-AUC | 0.6090 | **0.6658** | **+0.057** |
| F1 | 0.5774 | **0.6301** | **+0.053** |
| Top-10% lift | 1.27× | **1.47×** | better targeting |

### Interpretation

- RF **clearly beat LR** on the same inputs → non-linearity / interactions matter.
- Feature importances aligned with EDA: `eqpdays`, `months`, `change_mou`, `mou_Mean`, `hnd_price`.
- **But** ~61% accuracy still felt “close to coin-flip” in a casual reading, which motivated a further improvement round (richer features + stronger booster).

### Verdict on Stage 2

RF becomes the **first serious primary model**. LR stays as the linear baseline in the report.

---

## 3. Stage 3 — What we improved next (and why)

Accuracy ~0.61 is better than LR, but for the assignment we asked: *can we do better without cheating (no leakage, no peeking at test labels for training)?*

### Changes made

#### A. Richer feature engineering

Added explainable derived fields and a few extra usage columns:

| New / added feature | Intent |
|---------------------|--------|
| `eqpdays_x_change_mou` | Interaction: aging handset × usage change |
| `mou_per_month` | Usage intensity vs tenure |
| `rev_per_mou` | Revenue density (plan/value proxy) |
| `drop_rate` | Dropped calls per attempt |
| `care_per_mou` | Care intensity vs usage |
| `eqpdays_bin` | Coarse equipment-age segments |
| `avg6mou`, `avgrev` | Extra short/medium windows (still limited) |
| `complete_Mean`, `attempt_Mean`, `roam_Mean` | Completion / roaming friction |
| `marital`, `ethnic` | Light demographics (lower missingness than cars/income) |

Sparse ultra-missing demos (`numbcars`, dwelling size, etc.) stayed out.

#### B. Stronger algorithm — gradient boosting

Tried:

- `HistGradientBoostingClassifier` (sklearn)
- `XGBoost` with a **RandomizedSearchCV** (20 draws, 3-fold CV, scoring=`roc_auc`)

#### C. Best XGBoost hyperparameters (selected by CV AUC)

```text
max_depth=7
learning_rate=0.03
n_estimators=600
subsample=0.8
colsample_bytree=0.7
min_child_weight=1
reg_lambda=2.0
```

CV best ROC-AUC ≈ **0.686** (then confirmed on holdout).

### Results — full comparison (same split)

| Model | Features | Acc @0.5 | ROC-AUC | F1 @0.5 | Top-10% churn | Lift |
|-------|----------|----------|---------|---------|---------------|------|
| Logistic Regression | Lean | 0.5802 | 0.6090 | 0.5774 | 0.6305 | 1.27× |
| Random Forest | Lean | 0.6133 | 0.6658 | 0.6301 | 0.7285 | 1.47× |
| Random Forest | Rich | 0.6177 | 0.6745 | 0.6275 | 0.7605 | 1.53× |
| HistGradientBoosting | Rich | 0.6296 | 0.6838 | 0.6334 | 0.7845 | 1.58× |
| XGBoost (default-ish rich) | Rich | 0.6308 | 0.6885 | 0.6319 | 0.7825 | 1.58× |
| **XGBoost tuned (PRIMARY)** | **Rich** | **0.6327** | **0.6889** | **0.6330** | **0.7820** | **1.58×** |

Optional: threshold tuned on **train** F1 (not test) moved XGB accuracy@thr to ~0.63 and F1@thr to ~0.66 — useful if the business prefers F1 over the default 0.5 cut.

---

## 4. Final primary model (current)

**Primary:** Tuned **XGBoost** on the **rich** feature set.

| Metric | Final value |
|--------|-------------|
| Test accuracy (threshold 0.5) | **0.6327 (~63.3%)** |
| Test ROC-AUC | **0.6889** |
| Top 10% targeting lift | **~1.58×** (≈78% churn in the riskiest tenth) |

### Gain vs the first model (Logistic Regression)

| | Absolute gain |
|--|---------------|
| Accuracy | **+5.3 percentage points** (0.580 → 0.633) |
| ROC-AUC | **+0.080** (0.609 → 0.689) |
| Top-10% lift | **1.27× → 1.58×** |

### Gain vs Random Forest (lean)

| | Absolute gain |
|--|---------------|
| Accuracy | **+1.9 pp** (0.613 → 0.633) |
| ROC-AUC | **+0.023** (0.666 → 0.689) |
| Top-10% churn rate | 72.9% → 78.2% |

---

## 5. Honest reading — is this still “coin flip”?

**No — but it is also not magic.**

| Claim | Fair? |
|-------|-------|
| “Same as tossing a coin” | **No.** Coin / majority ≈ 50%; we are at **63%** accuracy and **0.69 AUC**, with **1.58×** concentration of churners in the top decile. |
| “Good enough to auto-fire every customer” | **No.** Many errors remain; use as a **ranking / targeting** tool. |
| “Dataset is useless” | **No.** Real messy telecom signal exists; separability is inherently limited. |
| “We can hit 90% accuracy easily” | **Unlikely** without leakage or a different label definition. High-60s AUC is a realistic band for this extract. |

**Point of ML here:** prioritize the riskiest customers for retention offers so budget buys more saved accounts than random outreach — demonstrated by lift rising from 1.27× (LR) to 1.58× (XGB).

---

## 6. Why each model was kept in the record

| Model | Role in the report |
|-------|--------------------|
| Logistic Regression | Stage-1 baseline; coefficient story; proves linear methods are weak |
| Random Forest | Stage-2 proof that trees help on the **same** lean features |
| XGBoost (tuned) | Stage-3 **primary** predictor after richer FE + boosting + light tuning |

---

## 7. What we did *not* do (on purpose)

- **SMOTE** — classes already ~50/50.
- **Using test labels to pick features** — would be leakage.
- **Customer_ID as a feature** — leakage / non-generalizable.
- **Claiming causation** from importances — predictive association only.

---

## 8. Where the trained models live (on disk)

Training originally ran only inside [`modeling.ipynb`](../modeling.ipynb) (in memory).  
Saved artifacts are now in [`models/`](../models/):

| Stage | File |
|-------|------|
| Before — Logistic Regression | [`before_logistic_regression_lean.joblib`](../models/before_logistic_regression_lean.joblib) |
| Before — Random Forest (then-primary) | [`before_random_forest_lean.joblib`](../models/before_random_forest_lean.joblib) |
| After — tuned XGBoost | [`after_xgboost_tuned_rich.joblib`](../models/after_xgboost_tuned_rich.joblib) |
| **Final (alias of after)** | [`final_model.joblib`](../models/final_model.joblib) |

See [`models/README.md`](../models/README.md) and [`models/manifest.json`](../models/manifest.json).

---

## 9. How to reproduce

1. Install deps (`scikit-learn`, `xgboost`, …) from [`requirements.txt`](../requirements.txt).
2. Run [`modeling.ipynb`](../modeling.ipynb) top to bottom (includes Stage 1–2, then improvement section with XGB).
3. Compare tables in the notebook with this document.
4. Load saved models from [`models/`](../models/) as in the models README.

---

## 10. One-paragraph summary for the assignment

We first trained **logistic regression** on a lean EDA-driven feature set and obtained only ~**58%** accuracy (AUC ~0.61), which is barely above chance and confirmed that churn is not a linear problem. We therefore switched the primary approach to **Random Forest**, which lifted accuracy to ~**61%** and AUC to ~**0.67** on the same features, showing that non-linear interactions matter. Because ~61% still looked weak as a headline score, we **improved** the pipeline with richer engineered features and a **tuned XGBoost** model, reaching ~**63.3%** accuracy, AUC ~**0.69**, and **1.58×** lift in the top 10% risk segment. The dataset and ML approach are worthwhile as a **targeting** system, not as a near-perfect individual classifier; further huge accuracy jumps are limited by class overlap, not by forgetting to “try ML.”
