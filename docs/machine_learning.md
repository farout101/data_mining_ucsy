# Machine Learning — Detailed Documentation

This document explains the **modeling route**, **algorithms**, **evaluation**, **interpretation**, and **business targeting view** for Company A telecom churn prediction.

Related files:

- Implementation: [`modeling.ipynb`](modeling.ipynb)
- Feature engineering write-up: [`feature_engineering.md`](feature_engineering.md)
- EDA: [`main.ipynb`](main.ipynb), [`residue.ipynb`](residue.ipynb)
- Column meanings: [`telecom_column_dictionary.md`](telecom_column_dictionary.md)

---

> **Status update:** The primary model is now **tuned XGBoost** (rich features).  
> Full chronology (LogReg → RF → XGB), motivations, and score tables: [`model_evolution.md`](model_evolution.md).  
> Final holdout accuracy ≈ **0.6327**; ROC-AUC ≈ **0.6889**; top-10% lift ≈ **1.58×**.  
> **Saved models:** [`models/final_model.joblib`](../models/final_model.joblib) (after) · before-improvement files in [`models/`](../models/) — see [`models/README.md`](../models/README.md).

---


## 1. Problem statement

**Task:** binary classification — predict whether a customer will churn (`churn = 1`) in the 31–60 day window after the observation date.

**Inputs:** lean engineered feature matrix from Client + Record (see feature engineering doc).

**Outputs:**

1. Class label at threshold 0.5 (for standard metrics)
2. Churn probability (for ranking / top-10% targeting)

**Business framing:** not only “maximize accuracy,” but *identify a high-risk segment worth a retention offer*, consistent with a proposal-style assignment.

---

## 2. Route chosen (and why)

| Choice | Why this route | Counterpart | How to look at it |
|--------|----------------|-------------|-------------------|
| **Random Forest as primary** | EDA showed weak linear association with churn (`\|r\|` max ≈ 0.11). Churn likely involves interactions (e.g. old handset × usage drop). Trees handle mixed types, non-linearity, and mild collinearity without feature scaling. | Logistic regression | If RF AUC ≫ LR → cite non-linearity. If close → prefer LR for clearer coefficients in the write-up. |
| **Logistic regression as counterpart** | Provides signed coefficients for a transparent linear story; standard teaching baseline. | RF-only | Always keep a simple linear baseline so gains are defensible. |
| **sklearn only** | RF + LR + metrics are enough for the course path; fewer install issues than boosted libraries. | XGBoost / LightGBM | Revisit only if RF clearly plateaus and you need more accuracy. |
| **No SMOTE / no class_weight** | Target is already ≈ 50/50. | Oversampling or balanced weights | SMOTE would invent synthetic customers and add noise when classes are balanced. |
| **Separate modeling notebook** | Keeps [`main.ipynb`](main.ipynb) as clean EDA evidence. | One mega-notebook | Easier grading and re-runs. |

### Important data caveat

Churn ≈ **49.6% / 50.4%** is **unusual** for live telecom portfolios (often single-digit monthly churn). Treat this extract as possibly **balanced for teaching**. Accuracy is therefore a usable headline metric *here*, but in industry you would emphasize recall@k, precision@k, or expected value under a rare-event prior.

---

## 3. Modeling workflow

```text
Lean X_train / X_test  (from FE)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 RF Pipeline                      LR Pipeline
  median impute                    median impute
  one-hot cats                     StandardScaler
  RandomForest                     one-hot cats
                                   LogisticRegression
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Metrics table + ROC + CM
                   │
                   ▼
         RF importances vs LR coefficients
                   │
                   ▼
         Top 10% risk lift (RF probabilities)
```

All fitting and evaluation are in [`modeling.ipynb`](modeling.ipynb).

---

## 4. Model specifications

### 4.1 Random Forest (primary)

```text
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42,
)
```

| Hyperparameter | Value | Intent |
|----------------|-------|--------|
| `n_estimators=200` | 200 trees | Stable importance / probability estimates |
| `max_depth=12` | Limited depth | Reduce overfitting on 80k rows with OHE categoricals |
| `min_samples_leaf=20` | Larger leaves | Smoother probabilities for ranking |
| `random_state=42` | Fixed | Reproducibility |
| `class_weight` | default (`None`) | Classes already balanced |

**Preprocessing for RF:** median impute numerics + one-hot categoricals. **No scaling.**

**Why RF over a single decision tree?** A single tree is unstable and easy to overfit; bagging averages many trees for better generalization and more reliable importances.

### 4.2 Logistic Regression (counterpart)

```text
LogisticRegression(max_iter=1000, random_state=42)
```

| Choice | Intent |
|--------|--------|
| `max_iter=1000` | Ensure convergence after one-hot expansion |
| Default L2 penalty | Shrinks noisy coefficients among correlated OHE levels |
| `StandardScaler` on numerics | Puts `eqpdays`, `mou_Mean`, etc. on comparable scales |

**Preprocessing for LR:** median impute → standardize numerics → one-hot categoricals.

### 4.3 Shared evaluation protocol

- Predict probability `P(churn=1)`
- Hard label: `pred = 1 if proba ≥ 0.5 else 0`
- Metrics on **held-out test** (20%, stratified): accuracy, ROC-AUC, precision, recall, F1, confusion matrix, ROC curve

---

## 5. Results (executed notebook)

Figures and tables are produced when [`modeling.ipynb`](modeling.ipynb) is run. Representative holdout numbers:

### 5.1 Side-by-side metrics

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| **Random Forest (primary)** | 0.6133 | **0.6658** | 0.5990 | 0.6645 | **0.6301** |
| Logistic Regression (counterpart) | 0.5802 | 0.6090 | 0.5762 | 0.5786 | 0.5774 |

### 5.2 How to read these numbers

| Observation | Interpretation |
|-------------|----------------|
| Both beat ~0.50 chance accuracy | Models learned *some* real signal |
| RF beats LR on AUC (~0.67 vs ~0.61) and F1 | Supports the EDA claim that relationships are partly **non-linear / interactive** |
| Absolute accuracy ~0.61 | Modest — expected when classes overlap heavily in feature space (EDA pairplots already suggested overlap) |
| RF recall (churn) ~0.66 | Catches more churners than LR at the 0.5 threshold — useful if missing a churner is costly |

**Baseline reference:** always predicting the majority class (“stay”) would get ≈ 50.4% accuracy. RF’s 61.3% is a real lift over that trivial rule, but not a solved problem — appropriate honesty for the write-up.

### 5.3 Confusion matrices & ROC

The notebook plots:

1. RF confusion matrix  
2. LR confusion matrix  
3. ROC curves for both vs the diagonal chance line  

Use ROC-AUC as the primary *ranking* quality metric; use the confusion matrix to discuss false alarms vs missed churners for the business proposal.

---

## 6. Interpretation

### 6.1 Random Forest — top feature importances

Importances are **mean decrease in impurity** across trees (not causal effects). Top contributors in the executed run:

| Rank | Feature (transformed name) | Importance |
|------|----------------------------|------------|
| 1 | `eqpdays` | 0.174 |
| 2 | `months` | 0.144 |
| 3 | `change_mou` | 0.070 |
| 4 | `mou_Mean` | 0.066 |
| 5 | `avg3mou` | 0.051 |
| 6 | `totmrc_Mean` | 0.051 |
| 7 | `hnd_price` | 0.044 |
| 8 | `change_rev` | 0.042 |
| 9 | `rev_Mean` | 0.041 |
| 10 | `overage_ratio` | 0.036 |

**Alignment with EDA:** handset age, tenure, usage level, and usage *change* dominate — the same levers highlighted in exploratory overlays. That consistency is a strong narrative point: *the model did not invent a contradictory story.*

**Caveat:** correlated features share importance (e.g. `mou_Mean` and `avg3mou`). Do not over-claim unique causal credit for each.

### 6.2 Logistic Regression — top |coefficients|

Signed coefficients (standardized numerics / one-hot levels). Example top signals:

| Feature | Coef (sign) | Plain-language read |
|---------|-------------|---------------------|
| `eqpdays` | +0.29 | Older equipment → higher churn odds |
| `area` Northwest/Rocky Mountain | +0.21 | Regionally elevated risk |
| `months` | −0.21 | Longer tenure → lower churn odds (linear view) |
| `mou_Mean` | −0.20 | Higher usage → lower churn odds |
| `asl_flag_Y` | −0.18 | Spending-limit flag associated with lower churn |
| `change_mou` | −0.11 | More positive change (usage up) → lower churn |

Orange vs blue in the notebook chart: **positive = associated with churn**, **negative = associated with staying** (all else equal in the linear model).

**How RF and LR disagree usefully:**

- RF ranks **global predictive utility** (including interactions / thresholds).
- LR ranks **linear partial associations**.
- Geographic one-hots loom larger in LR; continuous handset/usage features loom larger in RF. That is normal: trees carve continuous space; LR spends degrees of freedom on categorical levels.

---

## 7. Business view — top 10% predicted risk

### 7.1 Method

1. Score the **test set** with RF `predict_proba`.
2. Take customers at or above the **90th percentile** of predicted risk (top 10%).
3. Compare their **actual** churn rate to the overall test baseline.
4. Profile mean differences on key EDA levers.

### 7.2 Lift results (executed run)

| Quantity | Value |
|----------|-------|
| Test baseline churn | 49.56% |
| RF probability cutoff (P90) | ≈ 0.616 |
| Customers targeted | 2,000 / 20,000 |
| Actual churn in top 10% | **72.85%** |
| **Lift vs random** | **1.47×** |

**Meaning:** contacting the model’s highest-risk tenth yields about **1.47 times** as many churners as contacting a random tenth of the same size. That is the core “targeting beats spray-and-pray” argument for a retention pilot.

### 7.3 Profile of the top-risk group vs all test

| Feature | All test mean | Top 10% mean | Diff |
|---------|---------------|--------------|------|
| `eqpdays` | 392.6 | 547.0 | **+154** (older handsets) |
| `hnd_price` | 102.1 | 70.8 | **−31** (cheaper devices) |
| `mou_Mean` | 510.4 | 319.9 | **−190** (lower usage) |
| `change_mou` | −12.3 | −57.8 | **−45** (sharper decline) |
| `months` | 18.9 | 19.6 | +0.8 |
| `custcare_Mean` | 1.75 | 0.50 | −1.25 |

This profile **echoes EDA**: high-risk customers look like aging, cheaper handsets with thinner and declining usage — natural hooks for upgrade offers, plan-fit checks, and win-back campaigns.

> **Do not over-interpret `custcare_Mean` alone:** lower care calls in the top bin can mean disengaged customers (silent churn) rather than “happy” ones. Pair with usage decline in the narrative.

### 7.4 What this is / is not

| This section is | This section is not |
|-----------------|---------------------|
| A lift proof-of-concept for targeting | A full ROI / offer-cost memo |
| Tied to actionable FE features | A guarantee of campaign success |
| Computed on holdout probabilities | In-sample vanity lift |

A full business proposal would still need offer cost, acceptance rate, and margin assumptions — optional next step beyond this ML doc.

---

## 8. Why not other common ML choices?

| Approach | Why not (for this assignment path) |
|----------|-------------------------------------|
| **SMOTE** | Classes already balanced; synthetic samples add complexity without clear benefit |
| **XGBoost / LightGBM first** | Stronger sometimes, but heavier dependency story; RF already beats LR enough to justify trees |
| **Deep learning** | Overkill for tabular ~25 features; weak interpretability for course markers |
| **Accuracy-only optimization** | Ignores ranking quality needed for top-k targeting; we report AUC + lift too |
| **Threshold tuning for F1** | Possible enhancement; 0.5 is the transparent default for the counterpart comparison |

---

## 9. Limitations (state these in the assignment)

1. **Balanced label:** may not reflect true population churn rate → calibrate thresholds / priors before production.
2. **Modest AUC (~0.67):** many stayers and churners overlap; expect false positives in outreach.
3. **Importance ≠ causation:** `eqpdays` predicts churn; upgrading phones may or may not *cause* retention without an experiment.
4. **Geography in LR:** area effects may proxy network quality or competition, not something marketing alone fixes.
5. **Single holdout:** nested CV / time-based split would be stronger if a true observation timeline were available.
6. **Lean feature set:** some weak signal in dropped demographics may exist; we traded that for missingness honesty and clarity.

---

## 10. Reproducibility

| Item | Value |
|------|-------|
| Notebook | [`modeling.ipynb`](modeling.ipynb) |
| Seed | `random_state=42` |
| Split | Stratified 80/20 |
| Primary model | Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=20`) |
| Counterpart | Logistic Regression (`max_iter=1000`) |
| Dependencies | `scikit-learn` et al. in [`requirements.txt`](requirements.txt) |

Re-run the notebook top-to-bottom after installing requirements. Metrics may differ slightly across machines/BLAS but should stay in the same ballpark.

---

## 11. How to write this up (suggested narrative arc)

1. **Problem:** predict churn to target retention, not to chase a leaderboard alone.  
2. **EDA → FE:** lean, actionable features; no ID; train-only clipping ([`feature_engineering.md`](feature_engineering.md)).  
3. **Models:** RF primary because linear signal was weak; LR as interpretability baseline.  
4. **Evidence:** RF AUC/F1 advantage; importances match EDA (`eqpdays`, `change_mou`, …).  
5. **Action:** top 10% risk → ~1.47× lift; profile suggests handset upgrade + usage win-back.  
6. **Limits:** balanced extract, modest separability, correlation ≠ causation.

---

## 12. Summary

| Question | Answer in this project |
|----------|------------------------|
| What model is primary? | Random Forest |
| What is the counterpart? | Logistic regression |
| Did trees help? | Yes — higher AUC/F1 than LR |
| Is the signal actionable? | Yes — aligns with handset age & usage decline |
| Can we target? | Yes — top 10% shows 1.47× lift on holdout |
| Is FE documented separately? | Yes — [`feature_engineering.md`](feature_engineering.md) |

> **One-sentence summary:** We train a leakage-safe Random Forest on a lean, EDA-driven feature set, validate it against logistic regression, and show that ranking customers by predicted risk concentrates churners for retention outreach.
