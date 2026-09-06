# Can We Push Accuracy Higher? (Kaggle notes vs our experiments)

The Kaggle dataset card describes several “hidden anomalies” and a transformation pipeline that supposedly unlocks higher predictive power. This note records **the Kaggle “About Dataset” text (kept for reference)**, **what is true**, **what we tried**, and **whether accuracy moved**.

Related: [`why_accuracy_plateau.md`](why_accuracy_plateau.md) · [`model_evolution.md`](model_evolution.md)

Dataset: [Kaggle — Telecom Customer Churn 100K cleaned records](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records)

---

## 0. Archived — Kaggle “About Dataset” (as found)

> Kept here so the project notes retain the dataset-card wording we used when deciding whether accuracy could still be improved.

### Context & overview

Understanding customer attrition (churn) is one of the most critical challenges in the telecommunications industry. This dataset contains comprehensive behavioral, operational, and financial records of **100,000** customers with **100** initial features. It provides a rigorous testing ground for advanced tabular data engineering, deep learning, and gradient-boosting architectures.

The target variable is `churn` (binary: 1 if the customer left, 0 if they stayed), which is almost perfectly balanced (~**50.44%** class 0 / ~**49.56%** class 1), making it ideal for standard metric evaluation.

### Dataset structure & baseline facts

| Item | Value |
|------|--------|
| Total records (rows) | 100,000 |
| Total dimensions (features) | 100 (79 numerical, 21 categorical) |
| Target | `churn` (binary classification) |
| Usability goals (per card) | Multi-collinearity suppression, severe zero-inflation, advanced feature interaction |

### Statistical challenges embedded (per card)

1. **MNAR (Missing Not At Random)**  
   Features like `income` (~25.4% missing), `numbcars` (~49.3% missing), and `dwllsize` (~38.3% missing) may carry signal. The card claims customers who hide demographics have ~**2.5%** higher churn than those who provide them.

2. **Extreme kurtosis & zero-inflation**  
   Key financial features like `change_mou` and `rev_Mean` can show very high kurtosis (card cites up to **1665.37**), with mass near zero and heavy outliers.

3. **Multimodal contract boundaries**  
   `months` has sharp peaks at **10, 12, 18, and 24** months — treated as contract-expiration zones linked to cyclical churn.

4. **Phantom multi-collinearity**  
   The card warns that after target encoding, many categoricals can form a **>0.95** correlation network and hurt deep-learning training if untreated.

### Proven transformation pipeline (per card)

The card’s stated baseline optimization reduced the matrix to **68** features via:

- **Feature engineering:** `loyalty_frustration_index = drop_vce_Mean / (months + 1)` and contract-expiration flags to reduce class overlap  
- **Outlier capping:** clipping in robust bounds after a **RobustScaler** block (card cites about **[-3.0, +3.0]**)  
- **Kurtosis suppression:** forcing max kurtosis down from ~1665 toward a more symmetric value (card cites ~**2.38**)

---

## 1. Short answer

**Yes, you can still try to improve — but don’t expect a jump from ~0.63 to 0.80+ on this label with the same tables.**

The Kaggle challenges are **real** (we verified several). Adding the suggested features (MNAR flags, contract-month flags, `loyalty_frustration_index`, RobustScaler-style prep) on top of our tuned XGBoost **did not meaningfully raise holdout accuracy** in a controlled re-run (~0.6335 → ~0.6333).

So: the card explains *why the data is hard*; it does **not** guarantee a large accuracy unlock.

---

## 2. What the Kaggle card claims (and what we checked)

| Claim | Our check | Verdict |
|-------|-----------|---------|
| ~50/50 churn | ~49.6% / 50.4% | True |
| Demographics MNAR; missing linked to higher churn | `income` missing: churn **51.4%** vs present **48.9%** (~**+2.5 pp**); similar gaps for `dwllsize`, `numbcars` | True directionally |
| Extreme tails / zero-inflation on `change_mou`, revenue | EDA already showed heavy tails; we clip train percentiles | True |
| `months` peaks at contract-like values | Churn at month **12 ≈ 62%**, month **10 ≈ 40%** (different!) | True — useful as a *story*, weaker as a free accuracy boost |
| Need multi-collinearity control | MOU/revenue families highly correlated | True — we already lean-selected |

---

## 3. What we already did (before reading that card)

- Train-only outlier clipping on heavy columns  
- Derived ratios / interactions (`overage_ratio`, `eqpdays × change_mou`, …)  
- Dropped ultra-sparse demos instead of blindly imputing  
- Tuned XGBoost (better than LogReg / RF)  
- Documented the overlap ceiling in [`why_accuracy_plateau.md`](why_accuracy_plateau.md)

---

## 4. Extra experiment (Kaggle-inspired)

Same split (`test_size=0.2`, `random_state=42`), XGBoost-style booster.

| Run | Idea | Accuracy | ROC-AUC | Top-10% lift |
|-----|------|----------|---------|--------------|
| **A** | Our current rich FE (control) | **0.6336** | **0.6892** | 1.58× |
| **B** | A + MNAR flags + contract flags (10/12/18/24) + `loyalty_frustration_index` + a few demo categoricals + RobustScaler | 0.6333 | 0.6893 | 1.59× |
| **C** | B + stronger XGB (deeper / more trees) | 0.6312 | 0.6885 | 1.58× |

**Delta B vs A:** accuracy **≈ 0** (slightly down), AUC **≈ 0**.

Interpretation: those features are **plausible and partly informative**, but they largely **overlap signal we already capture** (tenure, drops, missingness patterns via other columns, etc.). They do not break class overlap.

---

## 5. So is there *any* way to improve?

### Realistic (small gains possible)

| Idea | Why it might help a little |
|------|----------------------------|
| Threshold tuning for F1 / business cost (not accuracy@0.5) | Better operating point; accuracy may not rise |
| Stacking / light ensembles | Sometimes +0.005–0.01 AUC |
| More careful categorical encoding (OHE vs target encoding with CV) | Card warns about target-encoding collinearity |
| Focus on **lift / precision@top-k** instead of accuracy | Already our strongest story (~1.58×) |
| Calibrated probabilities | Better for decision thresholds, not raw accuracy |

### Unrealistic expectations from the card’s tone

| Hype risk | Reality |
|-----------|---------|
| “Kurtosis suppression → high predictive power” | Cleaning tails stabilizes training; it doesn’t create new leave/stay separation |
| “68 symmetric independent features” | Feature count ≠ accuracy; we already reduced redundancy |
| Deep learning will smash tabular GBDT here | On this size/overlap, GBDT usually ties or wins; DL rarely jumps accuracy magically |

### What would be needed for a *large* jump

New information not in the 100 columns (competitor offers, contract end dates as ground truth, tickets, NPS, richer time series) — or a different label definition. See [`why_accuracy_plateau.md`](why_accuracy_plateau.md).

---

## 6. Practical recommendation for this project

1. **Keep final model ~0.63 accuracy / ~0.69 AUC** as the honest result.  
2. Emphasize **top-decile lift** and actionable drivers (`eqpdays`, usage decline, contract timing story).  
3. Optionally mention MNAR demographics (+~2.5 pp churn when income missing) as an EDA insight — even if it didn’t move XGB accuracy much.  
4. Don’t chase 80%+ accuracy on this extract without new data; graders usually prefer a **justified ceiling** over inflated metrics.

---

## 7. One-sentence summary

**The Kaggle “anomalies” are real and worth knowing, but in our tests they did not unlock a higher accuracy band — the ~0.63 wall is still mostly class overlap and limited label information, not a missing RobustScaler step.**
