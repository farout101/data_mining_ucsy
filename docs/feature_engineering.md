# Feature Engineering — Detailed Documentation

This document explains **what** feature engineering was done for Company A telecom churn prediction, **why** each choice was made, what was **deliberately excluded**, and how the steps connect to EDA and modeling.

Related files:

- Implementation: [`modeling.ipynb`](modeling.ipynb)
- EDA notebook: [`main.ipynb`](main.ipynb)
- EDA takeaways: [`residue.ipynb`](residue.ipynb)
- Column meanings: [`telecom_column_dictionary.md`](telecom_column_dictionary.md)
- Modeling write-up: [`machine_learning.md`](machine_learning.md)

---

## 1. Purpose of this stage

EDA answered: *Can we trust the data, and where is the signal?*

Feature engineering answers: *How do we turn raw columns into a model-ready matrix that is honest, lean, and explainable for a business proposal?*

Goals:

1. **No leakage** — do not use IDs or post-outcome information as predictors.
2. **No train/test contamination** — clip bounds and imputers must be fit on train only (or inside a `Pipeline`).
3. **Actionable features** — prefer variables that map to retention levers (handset age, usage decline, overages, segments).
4. **Avoid redundancy** — many MOU/revenue columns measure the same idea at different windows; keeping all of them inflates importance noise and weakens the narrative.

---

## 2. Starting point (after EDA)

| Fact from EDA | Implication for FE |
|---------------|--------------------|
| `Client` ⋈ `Record` is a clean 1:1 on `Customer_ID` → 100k × 100 | Join first; then drop the key |
| Churn ≈ 50/50 | No need for SMOTE / heavy class rebalancing |
| Demographics often 22–49% missing; usage mostly complete | Prefer usage/handset; drop or carefully encode sparse CRM fields |
| Linear `|r|` with churn max ≈ 0.11 (`eqpdays`) | Keep non-linear-friendly features; do not rely on correlation alone to select |
| MOU/revenue families highly correlated | Keep **one** recent window + **one** short rolling average, not all `tot*` / `avg*` / `*_Mean` |
| Negative `eqpdays`; extreme `change_*` tails | Clip / floor quality issues before modeling |
| Stronger gaps: older handsets, lower MOU, larger usage drop | Prioritize `eqpdays`, `hnd_price`, `mou_Mean`, `change_mou` |

---

## 3. End-to-end FE pipeline

```text
Client.csv + Record.csv
        │
        ▼
  Inner join on Customer_ID     → 100,000 × 100
        │
        ▼
  Quality fixes                 → eqpdays ≥ 0
        │
        ▼
  Derived features              → mou_decline_flag, overage_ratio
        │
        ▼
  Lean column selection         → 17 numeric + 8 categorical = 25
        │
        ▼
  Explicit Missing on cats
        │
        ▼
  Stratified 80/20 split        → train 80k / test 20k (same churn rate)
        │
        ▼
  Train-only winsorization      → change_mou, change_rev, overage_ratio
        │
        ▼
  sklearn ColumnTransformer     → median impute (+ scale for LR)
                                → one-hot categoricals
        │
        ▼
  Model matrix (~63 columns after OHE)
```

Implementation lives in [`modeling.ipynb`](modeling.ipynb) sections 1–3.

---

## 4. Step-by-step detail

### 4.1 Join and target definition

- **Join:** `record.merge(client, on="Customer_ID", how="inner")`.
- **Target `y`:** `churn` (0 = stayed, 1 = churned between 31–60 days after observation).
- **Dropped from `X`:** `Customer_ID` — unique identifier; predictive only via memorization / leakage risk; useless for new customers.

> **Why drop the ID?** Models can overfit arbitrary ID ranges. Retention decisions need *attributes*, not account numbers.

---

### 4.2 Quality fixes (domain constraints)

| Issue | Treatment | Rationale |
|-------|-----------|-----------|
| `eqpdays < 0` (equipment age in days) | `clip(lower=0)` | Negative age is impossible; floor to 0 rather than delete rare rows |
| Slightly negative `rev_Mean` / `totmrc_Mean` | Left as-is (credits / adjustments) | Rare; median impute handles nulls; clipping revenue to ≥0 would erase real billing semantics |

These fixes are applied on the full frame **before** split only when they are deterministic domain rules (not statistics estimated from the sample). Winsorization that uses percentiles is deferred until after the split (see §4.6).

---

### 4.3 Derived features

Two engineered fields were added because they are **simple, explainable, and map to retention actions**:

#### `mou_decline_flag`

```text
mou_decline_flag = 1 if change_mou < 0 else 0
```

- **Meaning:** customer’s recent minutes fell vs the prior 3-month average.
- **Why:** EDA showed churners with more negative `change_mou`. A binary flag makes “usage declining” easy to discuss in a proposal (win-back / plan fit) even when the continuous value is heavy-tailed.
- **Counterpart kept:** raw `change_mou` is still in the model so magnitude is not discarded.

#### `overage_ratio`

```text
overage_ratio = ovrrev_Mean / (|totmrc_Mean| + ε)    with ε = 1e-6
```

- **Meaning:** overage revenue relative to monthly recurring charge — a proxy for “bill shock” vs base plan size.
- **Why:** Absolute overage dollars mix light and heavy plans; a ratio is more comparable across customers.
- **Counterpart kept:** `ovrmou_Mean` and `ovrrev_Mean` remain for absolute scale.

> **What we did *not* derive (on purpose):** dozens of interaction terms, PCA, target encoding. Those can raise leaderboard scores but are harder to defend in a university / business narrative. Prefer a few transparent features first.

---

### 4.4 Lean numeric feature set (17)

| Feature | Family | Why kept |
|---------|--------|----------|
| `eqpdays` | Handset / loyalty | Strongest EDA signal; maps to upgrade offers |
| `hnd_price` | Handset | Churners tend toward cheaper devices |
| `phones`, `models` | Handset inventory | Device churn / multi-device complexity |
| `months` | Tenure | Timing of retention outreach |
| `mou_Mean` | Recent usage | Volume of engagement |
| `rev_Mean` | Recent revenue | Value / billing level |
| `totmrc_Mean` | Plan size | Base recurring charge |
| `change_mou`, `change_rev` | Trajectory | Decline / shock vs recent history |
| `avg3mou` | Short rolling window | One lifetime-adjacent view; **not** all of `avgmou`/`avg6mou`/`totmou` |
| `drop_vce_Mean` | Quality friction | Dropped calls → service pain |
| `custcare_Mean` | Support friction | Care volume as dissatisfaction proxy |
| `ovrmou_Mean`, `ovrrev_Mean` | Overage | Bill-shock candidates |
| `mou_decline_flag` | Derived | Interpretable decline indicator |
| `overage_ratio` | Derived | Plan-relative overage |

#### Why `avg3mou` only (not the full lifetime stack)

Client and Record both contain overlapping usage/revenue windows:

- Lifetime: `totmou`, `avgmou`, …
- Rolling: `avg3mou`, `avg6mou`, …
- Recent means: `mou_Mean`, …

EDA correlation heatmaps showed these are **near-duplicates**. Keeping the whole family:

- makes Random Forest importance split across clones,
- makes logistic coefficients unstable,
- muddies the story (“which minutes matter?”).

**Rule used:** keep **recent mean** (`mou_Mean`) + **one short rolling average** (`avg3mou`) + **change** (`change_mou`). Drop redundant `tot*`, `avgmou`, `avg6*`, peak/off-peak MOU splinters for the baseline model.

---

### 4.5 Lean categorical feature set (8)

| Feature | Why kept |
|---------|----------|
| `asl_flag` | Account spending limit — EDA showed clear churn-rate gap |
| `creditcd` | Credit card on file — segmentable for ops |
| `new_cell` | New vs existing cell user |
| `refurb_new` | Refurbished vs new handset |
| `dualband` | Device capability |
| `hnd_webcap` | Web-capable handset (RF importance later highlighted `WCMB`) |
| `area` | Geographic ops / marketing cut |
| `prizm_social_one` | Social/geo-lifestyle segment |

#### Encoding policy

1. Cast to string.
2. Fill nulls with the explicit level **`"Missing"`** (before and inside the pipeline imputer).
3. `OneHotEncoder(handle_unknown="ignore")` so unseen test levels do not crash scoring.

**Why keep `"Missing"` as a category?** In CRM data, “unknown income / unknown dwelling” is often informative (non-match in append files). Dropping null rows would bias the sample; silent mode imputation as “most frequent” would invent a false label.

#### High-cardinality note on `area`

`area` has ~19–20 levels. One-hot expands the matrix (final transformed width ≈ **63** columns). That is acceptable for RF/LR at n=80k. Frequency encoding was considered as a counterpart if OHE had exploded; it was unnecessary here.

---

### 4.6 What was deliberately dropped

| Dropped group | Examples | Why |
|---------------|----------|-----|
| Identifier | `Customer_ID` | Leakage / non-generalizable |
| Sparse demographics | `numbcars` (~49% NA), `dwllsize`, `HHstatin`, `ownrent`, `lor`, `income`, kids flags, etc. | High missingness + weak actionability for a telecom retention offer; risk of selection-bias stories |
| Redundant usage/revenue | `totmou`, `totrev`, `avgmou`, `avg6mou`, peak/off-peak splinters, many `plcd_*`/`comp_*` | Collinearity / narrative clutter; signal largely covered by lean set |
| Undocumented / noisy | e.g. poorly documented SMS fields | Avoid defending undefined predictors |

This is the **lean curated set** route. The counterpart is “use all ~99 columns + automatic selection.” That can be fine for a pure leaderboard; it is worse for explaining *why* customers leave.

---

### 4.7 Train/test split

```text
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

| Split | Rows | Churn rate |
|-------|------|------------|
| Train | 80,000 | ≈ 0.4956 |
| Test | 20,000 | ≈ 0.4956 |

**Why stratify?** Keeps class balance identical in train and test so accuracy / AUC comparisons are not distorted by sampling noise.

**Why 80/20?** Standard holdout for a 100k-row set; enough test mass for a stable top-10% lift estimate (2,000 customers).

---

### 4.8 Train-only winsorization (clipping)

Heavy tails on change / ratio features can dominate splits and scaled linear models.

| Column | Train 1st–99th pct bounds (example run) |
|--------|----------------------------------------|
| `change_mou` | [−841.30, 748.79] |
| `change_rev` | [−104.83, 121.34] |
| `overage_ratio` | [0.00, 3.91] |

**Critical rule:** percentiles are computed on **`X_train` only**, then applied to both train and test. Using full-data percentiles would leak test extremes into the transform.

Plot-time clipping in [`main.ipynb`](main.ipynb) was **not** modeling preprocessing — it was only for readable histograms.

---

### 4.9 Pipeline preprocessing (inside sklearn)

After the hand-crafted FE above, both models share a `ColumnTransformer`:

#### Numeric branch

| Model | Steps |
|-------|--------|
| Random Forest | `SimpleImputer(strategy="median")` |
| Logistic Regression | median impute → `StandardScaler()` |

- **Median impute:** robust to remaining skew; usage fields are mostly complete.
- **Scaling for LR only:** logistic regression with regularization / gradient steps is sensitive to feature scale. Trees do not need scaling.

#### Categorical branch

```text
SimpleImputer(strategy="constant", fill_value="Missing")
 → OneHotEncoder(handle_unknown="ignore", dense output)
```

#### Output width

After OHE, the design matrix has on the order of **~63** columns (17 numerics + expanded categoricals). That is the matrix the classifiers actually see.

---

## 5. Design decisions — chosen route vs counterpart

| Decision | Chosen | Counterpart | How to judge |
|----------|--------|-------------|--------------|
| Feature count | Lean ~25 raw features | All ~99 columns | Lean wins if importances align with EDA story and AUC stays competitive |
| Missing demos | Drop sparse CRM fields | Impute everything | Drop wins if demos are mostly NA and not actionable |
| Derived feats | 2 transparent flags/ratios | Heavy feature factory | Prefer explainability unless AUC gap is large |
| Clip tails | Train 1–99% on selected cols | No clip / robust scaler only | Clip if extremes distort LR or shallow trees |
| Cat encoding | One-hot + `"Missing"` | Ordinal / target encoding | OHE is safest for LR + RF comparison |
| Class balance | No SMOTE | SMOTE / class_weight | Unnecessary while churn ≈ 50/50 |

---

## 6. Leakage & honesty checklist

| Check | Status in this project |
|-------|------------------------|
| Target not used as a feature | Yes — `churn` only in `y` |
| ID not used as a feature | Yes — `Customer_ID` dropped |
| Imputer / scaler fit in pipeline on train folds | Yes — inside `Pipeline.fit` |
| Winsor bounds from train only | Yes |
| No future information after churn window | Assumed by dataset definition (features observed before 31–60 day churn label) |
| Churn rate caveat documented | Yes — ~50/50 may be a balanced extract, not raw industry incidence |

---

## 7. How this feeds modeling

| FE output | Used by |
|-----------|---------|
| Lean `X_train` / `X_test` | Both RF and LR pipelines |
| Same categorical encoding | Fair head-to-head comparison |
| Scale only in LR branch | Isolates “tree vs linear” rather than “scaled vs unscaled” confusion |
| Interpretable columns | RF importances + LR coefficients + top-10% risk profile |

Full training, metrics, and business lift are documented in [`machine_learning.md`](machine_learning.md).

---

## 8. Reproducibility notes

- Random seed: `RANDOM_STATE = 42`
- Data paths: `./telecom/Client.csv`, `./telecom/Record.csv`
- Library: `scikit-learn` (see [`requirements.txt`](requirements.txt))
- Re-run: execute [`modeling.ipynb`](modeling.ipynb) top to bottom

If you change the lean feature list, update **this document**, the notebook constants `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`, and the modeling write-up so the three stay aligned.

---

## 9. Summary table — approach → purpose

| Step | Approach | Main purpose |
|------|----------|--------------|
| 1 | Join on `Customer_ID` | Single customer table with target + profile |
| 2 | Drop ID | Prevent leakage |
| 3 | Floor `eqpdays` | Enforce domain validity |
| 4 | Derive decline flag + overage ratio | Add interpretable levers |
| 5 | Curate 17 + 8 features | Cut redundancy; keep actionable signal |
| 6 | `"Missing"` + one-hot | Honest categorical nulls |
| 7 | Stratified 80/20 | Fair evaluation |
| 8 | Train-only winsorize | Control heavy tails without leakage |
| 9 | Median impute (± scale for LR) | Model-ready matrix |

> **One-sentence summary:** Feature engineering here is a deliberate funnel from 100 raw columns to a lean, leakage-safe, business-explainable matrix — not an attempt to maximize column count.

---

## Appendix — Richer FE used in the improvement round

After the first lean model, [`modeling.ipynb`](modeling.ipynb) added extra engineered fields for XGBoost. See [`model_evolution.md`](model_evolution.md) Stage 3 for the list (`eqpdays_x_change_mou`, `mou_per_month`, `drop_rate`, `eqpdays_bin`, etc.) and results.

