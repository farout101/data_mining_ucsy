# Telecom Customer Churn Prediction: A Machine Learning Approach Using XGBoost

**Semester IX | IS-212**

**University of Computer Studies, Yangon (UCSY)**

**Student Name:** [Your Name]
**Student ID:** [Your Student ID]
**Supervisor:** [Supervisor Name]
**Submission Date:** [Date]

> **Note:** This manuscript is written as a self-contained project book. Paste it into the UCSY template (Times New Roman 12, 1.5 line spacing) and place each figure where the figure caption appears in the text.

---

## TABLE OF CONTENTS

**ACKNOWLEDGEMENT**
**ABSTRACT**

**CHAPTER 1: INTRODUCTION**
- 1.1 Project Background
- 1.2 Objectives of the Project
- 1.3 Scope and Limitations
- 1.4 Project Significance

**CHAPTER 2: DATA UNDERSTANDING AND EXPLORATORY DATA ANALYSIS**
- 2.1 Dataset Description
- 2.2 Why This EDA Order (Trust Funnel)
- 2.3 Data Join and Structure Audit
- 2.4 Missing Value Analysis
- 2.5 Target Variable Distribution
- 2.6 Univariate Feature Distributions
- 2.7 Bivariate Analysis — Features by Churn
- 2.8 Categorical Segment Analysis
- 2.9 Geographic Analysis
- 2.10 Correlation Analysis and Feature Redundancy
- 2.11 Joint Separability — Pairplot
- 2.12 Client vs Record Sanity Check
- 2.13 EDA Summary and Modeling Implications

**CHAPTER 3: FEATURE ENGINEERING**
- 3.1 Strategy and Goals
- 3.2 Quality Fixes
- 3.3 Derived Features (Lean Stage)
- 3.4 Lean Feature Set (Stages 1 & 2)
- 3.5 Why Not Keep All MOU/Revenue Columns
- 3.6 Rich Feature Set (Stage 3)
- 3.7 Features Deliberately Dropped
- 3.8 Train / Test Split
- 3.9 Preprocessing Pipeline and Leakage Control

**CHAPTER 4: METHODOLOGY — MODEL EVOLUTION**
- 4.1 Overall Training Pipeline and Design Choices
- 4.2 Stage 1 — Logistic Regression (Baseline)
- 4.3 Stage 2 — Random Forest (First Upgrade)
- 4.4 Stage 3 — XGBoost Tuned (Final Model)
- 4.5 Evaluation Metrics and How to Read Them
- 4.6 What We Explicitly Did Not Do

**CHAPTER 5: RESULTS AND DISCUSSION**
- 5.1 Full Model Comparison Table
- 5.2 Stage-by-Stage Analysis
- 5.3 Feature Importance and Coefficient Interpretation
- 5.4 Business View — Top 10% Risk Targeting
- 5.5 Accuracy Plateau Explanation
- 5.6 Kaggle Dataset-Card Notes vs Our Experiments

**CHAPTER 6: SYSTEM IMPLEMENTATION AND DEPLOYMENT**
- 6.1 Project Structure
- 6.2 Saved Models
- 6.3 Model Bundler (`XGBBundle`)
- 6.4 Streamlit Web Application

**CHAPTER 7: CONCLUSION**
- 7.1 Summary of Findings
- 7.2 Advantages of the Project
- 7.3 Disadvantages / Limitations
- 7.4 Future Work

**APPENDICES**
- Appendix A: Project Folder Structure
- Appendix B: Installation Guide
- Appendix C: XGBoost Hyperparameter Search Space
- Appendix D: EDA Figures Index
- Appendix E: Column Dictionary

---

## LIST OF FIGURES

| Figure | Caption |
|--------|---------|
| Figure 1.1 | End-to-end project pipeline |
| Figure 2.1 | Missing values — Client vs Record |
| Figure 2.2 | Churn distribution (target balance) |
| Figure 2.3 | Key feature distributions |
| Figure 2.4 | Feature distributions by churn status |
| Figure 2.5 | Categorical segment churn rates |
| Figure 2.6 | Geographic churn rates by area |
| Figure 2.7 | Correlation heatmap |
| Figure 2.8 | Pairplot — joint view of key metrics |
| Figure 2.9 | Client lifetime vs Record recent metrics |
| Figure 3.1 | Feature engineering pipeline |
| Figure 4.1 | Stage 1 — Logistic Regression pipeline and results |
| Figure 4.2 | Stage 2 — Random Forest pipeline and results |
| Figure 4.3 | Stage 3 — XGBoost tuning and final results |
| Figure 6.1 | System architecture — data to models to web demo |

---

## LIST OF TABLES

| Table | Caption |
|-------|---------|
| Table 2.1 | Dataset overview — Client.csv and Record.csv |
| Table 2.2 | Top features by missing value percentage |
| Table 2.3 | EDA step → purpose summary |
| Table 3.1 | Lean numeric feature set (17 features) |
| Table 3.2 | Lean categorical feature set (8 features) |
| Table 3.3 | Rich feature additions for Stage 3 |
| Table 3.4 | Features deliberately dropped and rationale |
| Table 3.5 | Train-only winsorization bounds |
| Table 4.1 | Stage 1 — Logistic Regression hyperparameters |
| Table 4.2 | Stage 2 — Random Forest hyperparameters |
| Table 4.3 | Stage 3 — XGBoost best hyperparameters |
| Table 4.4 | XGBoost hyperparameter search space |
| Table 5.1 | Full model comparison (all stages, same split) |
| Table 5.2 | Stage 1 detailed metrics |
| Table 5.3 | Stage 2 detailed metrics and improvement over Stage 1 |
| Table 5.4 | Stage 3 detailed metrics and improvement over Stage 2 |
| Table 5.5 | Top 10 feature importances (Random Forest) |
| Table 5.6 | Top coefficients — Logistic Regression |
| Table 5.7 | Top 10% risk segment profile vs all-test mean |
| Table 6.1 | Saved model files and their roles |
| Table 7.1 | Project achievements summary |

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to [Supervisor Name] for their guidance and valuable feedback throughout this project. I also thank the faculty of the University of Computer Studies, Yangon, for providing the learning environment and resources needed to complete this work.

I acknowledge the creators of the Telecom Customer Churn 100K dataset available on Kaggle (Shenouda Safwat), which forms the empirical foundation of this project. Special thanks to the open-source communities behind scikit-learn, XGBoost, and Streamlit, whose tools made this work possible.

Finally, I am grateful to my family and friends for their encouragement and patience during the completion of this project.

---

## ABSTRACT

Customer churn — the loss of existing subscribers — is a critical business problem in the telecommunications industry, where acquiring a new customer costs significantly more than retaining an existing one. This project develops a machine learning pipeline to predict customer churn for a telecom company (referred to as Company A) using an anonymized dataset of 100,000 customers.

The work follows three progressive model-improvement stages. First, a **Logistic Regression** baseline was trained on a lean, EDA-driven feature set of approximately 25 columns, yielding ~58% accuracy and ROC-AUC ~0.61. Second, a **Random Forest** was trained on the same lean features, lifting accuracy to ~61% and AUC to ~0.67, confirming that churn relationships are non-linear. Third, a **tuned XGBoost** model was trained on a richer engineered feature set with hyperparameter search, achieving ~63.3% accuracy, ROC-AUC ~0.69, and a top-10% risk lift of **1.58×** — meaning the highest-risk tenth of customers contains 1.58 times more churners than a random tenth of the same size.

Comprehensive exploratory data analysis (EDA) preceded all modeling, revealing weak linear correlations with churn (max |r| ≈ 0.11), high class balance (~50/50), and heavy collinearity among usage windows. Feature engineering curated a lean, leakage-safe, business-actionable feature set and a richer extended set for the final stage.

The project concludes with a Streamlit web application that demonstrates single-customer and batch churn prediction. Accuracy plateaus near 63% due to inherent class overlap in the available feature space — the model is most valuable as a **risk ranking and retention targeting** tool, not a perfect classifier.

**Keywords:** Customer Churn Prediction, Logistic Regression, Random Forest, XGBoost, Feature Engineering, Telecom, Data Mining, Streamlit

---

# CHAPTER 1

## INTRODUCTION

### 1.1 Project Background

The telecommunications industry is one of the most competitive consumer markets. Winning a new subscriber is expensive; losing an existing one wastes acquisition cost and future margin. Customer churn—the decision to leave the service—is therefore not only a reporting metric but an operational problem: if risk can be anticipated, retention offers can be aimed at the right people before they disconnect.

Machine learning approaches this problem by learning patterns from historical behavior and scoring each customer with a churn probability. That score is most useful when it supports targeting—who should enter a retention queue first—rather than when it is treated as a perfect prophecy for every individual.

This project applies that idea to an anonymized telecom extract (Company A style) with 100,000 customers and about one hundred features covering profile, usage, billing, handset, and demographics. The prediction target is whether the customer churns in the 31–60 day window after the observation date. The work is intentionally end-to-end: understand the tables, explore them in a fixed trust order, engineer a lean then richer feature matrix, train Logistic Regression then Random Forest then tuned XGBoost, evaluate with accuracy, AUC, and top-decile lift, and demonstrate scoring in a Streamlit application. Each upgrade is kept in the record so a reader can see not only the final number but why the previous stage was not enough.

> 📊 **Figure 1.1 — End-to-End Project Pipeline**
>
> *Figure 1.1 is the map of the whole project: raw Client and Record files flow through EDA, feature engineering, three model stages, and finally the saved models consumed by the Streamlit demo. Later chapters zoom into each box without repeating the entire map.*

### 1.2 Objectives of the Project

The project was designed so that each objective produces a readable artifact—not only a score. The first objective is to conduct EDA that establishes data integrity, class balance, and the main behavioral contrasts with churn. The second is to engineer features in a lean-to-rich progression that stays leakage-safe and tied to retention levers. The third is to train and compare Logistic Regression, Random Forest, and XGBoost as a documented evolution rather than a single final model. Related objectives are to tune the final booster with cross-validated search, to evaluate with accuracy, ROC-AUC, F1, and top-10% lift together, to interpret importances and risk segments in business language, and to deploy scoring through a Streamlit interface for single and batch prediction.

### 1.3 Scope and Limitations

The scope covers the anonymized 100,000-customer extract, the three-stage training path, the Streamlit demonstration, and evaluation with accuracy, AUC, and lift. It does not claim a live production CRM integration or a full retention ROI simulator.

Limitations are equally important to state up front. The data are anonymized and some fields are poorly documented. The near 50/50 churn rate is atypical of many live portfolios. Validation is a random stratified holdout rather than a pure time-based split. Accuracy is bounded by class overlap in the available features. Finally, the model scores risk; it does not by itself prescribe the exact offer a call center should make.

### 1.4 Project Significance

The significance of the work is twofold. Academically, it demonstrates a complete machine-learning lifecycle with decisions that can be audited—why the join came first, why lean features beat “all columns,” why trees replaced the linear baseline, and why 63% accuracy is still useful when lift is 1.58×. Practically, the lift analysis shows how a modest classifier can still improve retention efficiency by concentrating outreach on the riskiest customers instead of treating every subscriber the same.

---


# CHAPTER 2

## DATA UNDERSTANDING AND EXPLORATORY DATA ANALYSIS

Exploratory analysis in this project was not an exercise in drawing every possible chart. It was a deliberate check that the data could support an honest churn story. A trained model always returns *some* accuracy and *some* feature importance. Without EDA, those numbers might reflect real signal, a broken join, a mostly empty demographic column, or three near-copies of the same revenue field. The exploratory work therefore follows a **trust funnel**: fix identity first, then quality, then the target, then shapes and contrasts, then structure and sanity. Later plots only matter if earlier questions pass.

> 📊 **Figure 1.1** (Chapter 1) already shows how this EDA block feeds feature engineering and the three model stages. Chapter 2 expands only the left-hand “understand the data” portion of that pipeline.

### 2.1 Dataset Description

The dataset is an anonymized telecom extract (Company A style), published as cleaned 100K records on Kaggle. It arrives as two tables that describe the **same** customers from two angles. `Client.csv` holds who the customer is over a longer horizon: demographics, handset inventory, equipment age and price, geography, and lifetime or rolling usage averages. `Record.csv` holds how they behaved recently: mean monthly minutes and revenue, overages, dropped calls, customer care, tenure in months, and the outcome label `churn`.

**Table 2.1 — Dataset Overview**

| File | Rows | Role |
|------|------|------|
| `Client.csv` | 100,000 × 50 | Profile + lifetime / rolling history |
| `Record.csv` | 100,000 × 51 | Recent usage/billing + **churn** |
| Joined | 100,000 × ~100 | 1:1 on `Customer_ID` |

The label is binary: `churn = 1` if the customer left in the **31–60 day window after** the observation date, otherwise 0. Full column meanings are given in Appendix E (Column Dictionary). The dataset can be obtained from Kaggle: Telecom Customer Churn 100K cleaned records (https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records).

Without joining the two files, recent behavior cannot be related to profile, and a model cannot use both. That is why EDA begins with the join, not with churn charts.

### 2.2 Why This EDA Order (Trust Funnel)

The sequence is as follows. First ask whether Client and Record describe the same people. Then ask what is missing or broken. Only then look at the target balance, because metrics and baselines depend on how common churn is. Distributions come next so skew and wild tails do not silently dominate later comparisons. Contrasts by churn answer the business question—*what differs for leavers?*—while categorical and geographic cuts ask who operations can actually target. Correlation and pairplots test redundancy and separability. A final Client-versus-Record scatter checks that the join still makes sense semantically, not only on keys.

```text
Identity → Quality → Target → Shape → Contrast → Segments → Structure → Sanity
```

Skipping early steps produces confident-looking charts on the wrong population or on empty columns. That risk is why the notebook does not jump straight to modeling.

### 2.3 Data Join and Structure Audit

Both tables were merged with an inner join on `Customer_ID`. Audits showed unique IDs on each side, identical ID sets (no orphans), and a merged frame of exactly 100,000 rows. That result supports treating the extract as a clean one-row-per-customer analysis table.

For modeling, `Customer_ID` is removed from every feature matrix. An identifier can look predictive by memorizing arbitrary ID ranges; it does not explain *why* a new customer would leave and would be a form of leakage for deployment.

### 2.4 Missing Value Analysis

Missingness is strongly asymmetric. Recent usage and billing fields from Record are mostly complete. Many Client demographics—vehicles, dwelling size, household status, owner/renter, income—are missing for roughly one quarter to one half of customers.

> 📊 **Figure 2.1 — Missing Values: Client vs Record**
>
> *Figure 2.1 compares missing-value percentages across columns. The high-missingness tail is dominated by Client demographics, while Record usage fields stay dense. This chart is the evidence behind dropping sparse CRM fields from the lean feature set rather than building a churn narrative on half-blank income or car counts.*

**Table 2.2 — Illustrative Missingness and Lean Decisions**

| Feature | Approx. missing | Lean decision |
|---------|-----------------|---------------|
| `numbcars`, `dwllsize`, `HHstatin`, `ownrent`, `income` | ~25–49% | Dropped from primary lean set |
| `marital`, `ethnic` | lower | Allowed later in the rich set |
| `mou_Mean`, `eqpdays`, most Record means | <1% | Kept |

The practical reading is simple. The company knows how people *use* the phone much more reliably than it knows their household CRM appends. Preference for usage and handset features is therefore both a completeness choice and a business-actionability choice. Separately, missingness can itself correlate slightly with churn (an MNAR-style pattern discussed again in Chapter 5); that is an insight, not a reason to invent values for fields that are mostly empty.

### 2.5 Target Variable Distribution

> 📊 **Figure 2.2 — Churn Distribution**
>
> *Figure 2.2 shows counts for stayed (0) versus churned (1). The nearly even split (~49.6% / 50.4%) is unusual for live monthly telecom churn, which is often much rarer, but it is convenient for a teaching extract: accuracy becomes a meaningful headline because “always predict stay” only scores about half.*

Because classes are balanced, SMOTE and class weighting were not required. The same balance also means flashy 90%+ accuracy scores seen on imbalanced toy problems are not the right expectation here. Results must be read with the caveat that a production portfolio might need different thresholds under a rarer-churn prior.

### 2.6 Univariate Feature Distributions

Before comparing groups, the notebook inspects shapes. Means hide skew; a few extreme customers can dominate correlations.

> 📊 **Figure 2.3 — Key Feature Distributions**
>
> *Figure 2.3 shows histograms of revenue, minutes, MRC, tenure, equipment age, and usage change (often clipped to the 1st–99th percentile only for readability). `change_mou` and `change_rev` display heavy tails; `eqpdays` spans a wide range; revenue fields are moderately skewed. These shapes motivate train-only winsorization and the flooring of impossible negative equipment ages before modeling—not because charts look nicer, but because unclipped tails can distort both linear models and shallow tree splits.*

A few negative `eqpdays` values appear (impossible ages) and are floored at zero. Slightly negative revenue or MRC values are left alone as plausible credits. Plot clipping in EDA is visual only; modeling uses separately fitted training percentiles so the test set cannot set the clip bounds.

### 2.7 Bivariate Analysis — Features by Churn

This step is the business center of EDA. It asks which measurable levers differ between people who left and people who stayed.

> 📊 **Figure 2.4 — Feature Distributions by Churn Status**
>
> *Figure 2.4 overlays stayed versus churned distributions for equipment age, tenure, minutes, revenue, care calls, and usage change. Churners tend toward older handsets, lower recent MOU, and more negative `change_mou`. Tenure distributions largely overlap. Even for the stronger signals, the two classes still sit on top of each other—so a single threshold rule will not separate churn cleanly.*

The same pattern appears in mean gaps: equipment age is the largest single-feature shift and also shows the highest absolute linear correlation with churn (about 0.11). Cheaper handsets and declining usage reinforce a retention story around upgrades and win-back, not around one demographic switch. The overlap also foreshadows the accuracy plateau discussed in Chapter 5: the signal is real, but it is not sharp.

### 2.8 Categorical Segment Analysis

Telecom operations rarely “target a float.” They target groups—credit flags, device types, plan segments.

> 📊 **Figure 2.5 — Churn Rate by Categorical Segment**
>
> *Figure 2.5 bars churn rate by categories such as `asl_flag`, `creditcd`, handset flags, and lifestyle codes, with a dashed overall baseline. Some segments sit clearly above or below the national rate. Explicit “Missing” levels are shown so nulls are not invisible. These gaps justify keeping a small categorical set in feature engineering rather than modeling on numerics alone.*

### 2.9 Geographic Analysis

> 📊 **Figure 2.6 — Geographic Churn Rates by Area**
>
> *Figure 2.6 ranks geographic areas by churn rate. Even when the national rate is near 50%, some regions differ by several percentage points. That is enough to include `area` as a feature for localized outreach or network follow-up, while remaining honest that geography alone will not solve class overlap.*

### 2.10 Correlation Analysis and Feature Redundancy

> 📊 **Figure 2.7 — Correlation Heatmap**
>
> *Figure 2.7 summarizes Pearson correlations among selected numeric fields and churn. Two messages dominate. First, the churn row is weak: the strongest absolute correlation is only about 0.11 for `eqpdays`. Second, within the MOU and revenue families the cells light up—recent means, three-month averages, six-month averages, and lifetime totals move together. Those columns are not four independent stories; they are mostly repeated measurements of “how much they use” or “how much they pay.”*

High correlation does not make a column worthless. It makes it **redundant**. Feeding every window into a model splits importance across clones, destabilizes linear coefficients, and confuses the write-up. Feature engineering therefore keeps a recent level, one short rolling average, and the *change* fields—because trajectory is a different idea from level—rather than the entire stack. That decision is explained again in Chapter 3.

### 2.11 Joint Separability — Pairplot

> 📊 **Figure 2.8 — Pairplot of Key Metrics by Churn**
>
> *Figure 2.8 shows pairwise scatters and marginals for tenure, revenue, minutes, MRC, and equipment age on a sample of customers, colored by churn. In every two-dimensional view the classes mix. There is no neat linear fence. This single figure is why the project treats tree and boosting models as the primary predictive route and why accuracy expectations stay in the low sixties rather than the nineties.*

### 2.12 Client vs Record Sanity Check

> 📊 **Figure 2.9 — Client Lifetime vs Record Recent Metrics**
>
> *Figure 2.9 scatters lifetime or rolling Client averages against recent Record means. The clouds slope positively: people who used more over their life also tend to show higher recent means, but the points are not identical. That pattern supports a coherent join and motivates choosing time windows carefully instead of assuming every average column is unique signal.*

### 2.13 EDA Summary and Modeling Implications

Taken together, EDA says the merge is trustworthy, demographics are too sparse to carry the main story, the label is usable and balanced, usage and handset fields carry modest but interpretable risk signal, MOU/revenue windows are redundant, and churners overlap stayers enough that linear separation will be weak. Those conclusions become the requirements list for Chapter 3 and the justification for the Stage 1→2→3 model path in Chapter 4.

**Table 2.3 — EDA Step → Purpose**

| Step | Purpose for later work |
|------|------------------------|
| Join audit | Same population; drop ID |
| Missingness | Prefer complete, actionable fields |
| Target check | Accuracy OK; no SMOTE |
| Shapes | Winsorize / floor carefully |
| By-churn overlays | Retention levers |
| Segments / area | Operable groups |
| Correlation / pairplot | Lean features; tree models; honest ceiling |
| Sanity scatters | Keep recent + limited history, not everything |

---

# CHAPTER 3

## FEATURE ENGINEERING

EDA answered whether the data could be trusted and where signal lived. Feature engineering turns those answers into a matrix a model can train on without cheating. The goals are no leakage, no train/test contamination, preference for actionable telecom levers, and refusal to keep every redundant usage window.

> 📊 **Figure 3.1 — Feature Engineering Pipeline**
>
> *Figure 3.1 diagrams the path from the joined table through quality fixes, derived fields, lean or rich selection, stratified split, train-only clipping, and a ColumnTransformer into the classifiers. The diagram is the map; the sections below are the reasons for each box.*

### 3.1 Strategy and Goals

Two feature sets were built on purpose. A **lean** set of roughly twenty-five columns supports Stages 1 and 2 so Logistic Regression and Random Forest can be compared fairly on the same inputs. A **rich** set adds a handful of interactions, ratios, and extra windows for Stage 3 XGBoost. The lean set is not “incomplete work”; it is the controlled baseline that proves trees help even before extra engineering.

### 3.2 Quality Fixes

Impossible negative equipment ages are floored at zero. Slightly negative revenue or MRC values are left as billing credits rather than forced non-negative. Those domain rules do not use sample percentiles, so they can run before the split. Extreme tails on change and ratio fields, by contrast, are clipped using **training** quantiles only after the split, so test extremes cannot set the bounds.

### 3.3 Derived Features (Lean Stage)

Two transparent fields were added early. `mou_decline_flag` marks whether recent minutes fell relative to the prior window, turning the continuous `change_mou` story into a plain yes/no engagement decline indicator while still keeping the raw change for magnitude. `overage_ratio` divides overage revenue by absolute MRC so bill shock is comparable across cheap and expensive plans. Both map cleanly to retention language: win-back for declining usage, plan redesign or alerts for relative overage.

### 3.4 Lean Feature Set (Stages 1 & 2)

The lean numeric list keeps equipment age and price, handset counts, tenure, recent MOU and revenue, MRC, usage and revenue change, one three-month MOU average, dropped-call and care means, overage minutes and revenue, plus the two derived fields above. The lean categorical list keeps spending-limit and credit flags, new-cell and handset type flags, web capability, geographic area, and PRIZM social group—segments that showed churn gaps in Figures 2.5 and 2.6. Null categories become the explicit level `"Missing"` before one-hot encoding so “unknown” remains visible.

**Table 3.1 — Lean Numeric Features (17)**

| Feature | Family | Why kept |
|---------|--------|----------|
| `eqpdays` | Handset | Strongest EDA signal; maps to upgrade offers |
| `hnd_price` | Handset | Churners trend toward cheaper devices |
| `phones`, `models` | Handset inventory | Multi-device / upgrade complexity |
| `months` | Tenure | Timing of retention outreach |
| `mou_Mean` | Recent usage | Volume of engagement |
| `rev_Mean` | Recent revenue | Value / billing level |
| `totmrc_Mean` | Plan size | Base recurring charge |
| `change_mou`, `change_rev` | Trajectory | Usage/revenue decline |
| `avg3mou` | Short rolling avg | One lifetime-adjacent view without full stack |
| `drop_vce_Mean` | Quality friction | Dropped calls as service pain |
| `custcare_Mean` | Support friction | Care volume as dissatisfaction proxy |
| `ovrmou_Mean`, `ovrrev_Mean` | Overage | Bill-shock candidates |
| `mou_decline_flag` | Derived | Interpretable decline indicator |
| `overage_ratio` | Derived | Plan-relative overage |

**Table 3.2 — Lean Categorical Features (8)**

| Feature | Why kept |
|---------|----------|
| `asl_flag` | Clear EDA churn-rate gap |
| `creditcd` | Credit card flag — segmentable |
| `new_cell` | New vs existing cell user |
| `refurb_new` | Refurbished vs new handset |
| `dualband` | Device capability |
| `hnd_webcap` | Web-capable handset |
| `area` | Geographic cut for campaigns |
| `prizm_social_one` | Lifestyle / geo segment |

### 3.5 Why Not Keep All MOU/Revenue Columns

Readers sometimes assume that if columns are correlated they should all be kept “so the model can choose.” Correlation means the columns often move together—like two thermometers in the same room. `mou_Mean`, `avg3mou`, `avg6mou`, and lifetime totals largely restate how much the customer uses. Keeping every clone rarely adds new truth; it mostly adds complexity. Logistic regression then shares credit unstably among near-duplicates, trees split importance across clones, and the business chapter can no longer say which minutes matter. The project’s rule is therefore deliberate: keep recent level, one short average, and change. Stage 3 may add a little more history (`avg6mou`, `avgrev`), not the whole family.

### 3.6 Rich Feature Set (Stage 3)

When the Random Forest still left accuracy near 61%, Stage 3 added interactions and rates that stay explainable. These expand the EDA story; they are not a return to using every raw column.

**Table 3.3 — Additional Features for Stage 3**

| Feature | Type | Purpose |
|---------|------|---------|
| `eqpdays_x_change_mou` | Interaction | Aging handset combined with usage change |
| `mou_per_month` | Ratio | Usage intensity relative to tenure |
| `rev_per_mou` | Ratio | Revenue density / value per minute |
| `drop_rate` | Ratio | Dropped calls relative to attempts |
| `care_per_mou` | Ratio | Care intensity relative to usage |
| `eqpdays_bin` | Binned category | Coarse equipment-age segments (`0_6m`, `6_12m`, `1_2y`, `2y_plus`) |
| `avg6mou`, `avgrev` | Extra windows | Limited additional history beyond the lean set |
| `complete_Mean`, `attempt_Mean` | Call completion | Completion and attempt volume |
| `roam_Mean` | Roaming | Roaming friction |
| `marital`, `ethnic` | Demographic | Lower missingness than cars/income fields |

### 3.7 Features Deliberately Dropped

Some columns were excluded on purpose, not by accident.

**Table 3.4 — Features Deliberately Dropped**

| Group | Examples | Reason |
|-------|----------|--------|
| Identifier | `Customer_ID` | Leakage / does not generalize to new customers |
| Sparse CRM demographics | `numbcars`, `dwllsize`, `HHstatin`, `ownrent`, `lor`, `income` | Often 25–49% missing; weak retention actionability for the primary story |
| Redundant usage/revenue | `totmou`, `avgmou`, many peak/off-peak twins | Highly collinear with the lean windows already kept |
| Undocumented fields | Poorly documented SMS-style fields | Cannot defend undefined predictors in the write-up |

Using every column plus automatic selection can be fine for a leaderboard. It was rejected here because the assignment needs an explainable, business-facing feature story.

### 3.8 Train / Test Split

An 80/20 stratified split with `random_state=42` produces 80,000 training rows and 20,000 test rows at the same churn rate. Stratification keeps evaluation comparable across stages. All reported metrics in Chapters 4 and 5 use this recipe unless noted.

### 3.9 Preprocessing Pipeline and Leakage Control

After selection, heavy-tailed columns such as `change_mou`, `change_rev`, and `overage_ratio` are winsorized to the training 1st–99th percentiles.

**Table 3.5 — Train-only Winsorization Bounds (example run)**

| Column | Train 1st pct | Train 99th pct |
|--------|---------------|----------------|
| `change_mou` | −841.30 | 748.79 |
| `change_rev` | −104.83 | 121.34 |
| `overage_ratio` | 0.00 | 3.91 |
 Inside sklearn, numerics are median-imputed; logistic regression also standardizes. Categoricals are imputed to `"Missing"` and one-hot encoded with unknown levels ignored at score time. Trees do not need scaling; the linear baseline does. The leakage checklist is short and non-negotiable: target only in `y`, no ID features, transforms fit on train or inside a Pipeline, and no feature picking that peeks at test labels.

---

# CHAPTER 4

## METHODOLOGY — MODEL EVOLUTION

Modeling was staged so each upgrade has a clear reason. Logistic Regression establishes an interpretable linear baseline. Random Forest tests whether non-linear structure helps on the **same** lean features. Tuned XGBoost then asks whether richer features and boosting extract more ranking power without leakage. The subsections below record that path in full, with diagrams for each stage.

### 4.1 Overall Training Pipeline and Design Choices

SMOTE was rejected because classes are already near half-and-half. Deep learning was not the first choice: on this tabular overlap problem, gradient boosting is the stronger default, and trees remain easier to discuss in a course report. Every stage shares the stratified split so differences reflect model and features, not lucky draws.

> 📊 **Figure 4.1 — Stage 1: Logistic Regression**
>
> *Figure 4.1 summarizes the linear pipeline—scaled numerics, one-hot categoricals, logistic model—and the modest holdout scores that follow from weak linear signal.*

> 📊 **Figure 4.2 — Stage 2: Random Forest**
>
> *Figure 4.2 shows the same lean inputs under a forest. The jump in AUC and lift on identical features is the evidence that interactions and thresholds matter.*

> 📊 **Figure 4.3 — Stage 3: XGBoost**
>
> *Figure 4.3 covers richer features, randomized hyperparameter search, and the final holdout metrics used as the primary result.*

### 4.2 Stage 1 — Logistic Regression (Baseline)

Logistic regression was trained on the lean matrix with median imputation, standard scaling, and one-hot categoricals (`max_iter=1000`, default L2). It is the right first model when the question is not only “what scores best” but “what does a linear story look like?”

On the holdout set it reached about **58.0% accuracy** and **ROC-AUC 0.609**, with top-decile lift near **1.27×**. That is better than chance, yet still close enough to a coin flip that the project could not stop there. EDA had already capped linear correlation with churn near 0.11, and the pairplot had shown heavy overlap; a straight additive boundary was never likely to separate leavers cleanly. Stage 1 remains in the report as the interpretability counterpart and as proof that the later gains are not imaginary.

### 4.3 Stage 2 — Random Forest (First Upgrade)

Random Forest uses the **same lean features**, so the comparison isolates the algorithm. Two hundred trees, depth capped at 12, and minimum leaf size 20 give stable probabilities without needing feature scaling. Holdout accuracy rose to about **61.3%**, AUC to **0.666**, and top-10% lift to **1.47×**—clear gains over Stage 1.

Those gains match the EDA claim that churn depends on combinations (for example older equipment together with falling usage) more than on any single linear effect. Feature importances again highlighted `eqpdays`, tenure, `change_mou`, and usage level, aligning with Figure 2.4 rather than contradicting it. Accuracy near 61% still looked soft as a headline, which motivated Stage 3 rather than declaring victory.

### 4.4 Stage 3 — XGBoost Tuned (Final Model)

Stage 3 changes three things at once, each documented: the rich feature set from Chapter 3, gradient boosting that builds trees sequentially to correct previous errors, and a light `RandomizedSearchCV` (twenty draws, three-fold CV, scoring ROC-AUC) whose winning settings were depth 7, learning rate 0.03, six hundred trees, subsample 0.8, column subsample 0.7, and L2 regularization 2.0.

The held-out result is about **63.3% accuracy**, **AUC 0.689**, and **1.58×** top-decile lift. Relative to Stage 1 that is roughly +5.3 percentage points of accuracy and +0.08 AUC; relative to Stage 2 the accuracy lift is smaller, which is expected when the easy non-linear gain has already been taken. The artifact is saved as `models/final_model.joblib`.

### 4.5 Evaluation Metrics and How to Read Them

Accuracy at threshold 0.5 is a fair headline here only because labels are balanced. ROC-AUC asks a different question: if one customer churns and one stays, how often does the model rank the churner higher? Precision and recall describe campaign trade-offs; F1 balances them. Top-10% lift asks the business question directly—whether the riskiest tenth contains more actual leavers than a random tenth. A model can look “only 63% accurate” and still be useful if it concentrates risk for outreach.

### 4.6 What We Explicitly Did Not Do

The project refused SMOTE, refused ID features, refused picking features with test-label knowledge, and refused to treat impurity importance as causal proof that upgrading a phone will retain a customer. Those refusals are part of the methodology, not footnotes.

---

# CHAPTER 5

## RESULTS AND DISCUSSION

### 5.1 Full Model Comparison Table

**Table 5.1 — Full Model Comparison (same split)**

| Stage | Model | Features | Accuracy | ROC-AUC | Top-10% lift |
|-------|-------|----------|----------|---------|--------------|
| 1 | Logistic Regression | Lean | 0.5802 | 0.6090 | 1.27× |
| 2 | Random Forest | Lean | 0.6133 | 0.6658 | 1.47× |
| — | RF / HistGB / XGB variants | Rich | ~0.62–0.63 | ~0.67–0.69 | ~1.53–1.58× |
| **3** | **XGBoost Tuned** | **Rich** | **0.6327** | **0.6889** | **1.58×** |

The table is the quantitative spine of the project. Stage 1→2 is the large conceptual jump (linear versus trees). Stage 2→3 is incremental polishing (features, boosting, tuning). Both matter; only the first rewrites the modeling story.

### 5.2 Stage-by-Stage Analysis

Stage 1 shows that a transparent linear model finds signal but not enough for a sharp classifier. Stage 2 shows that the same clues, under a forest, rank customers better and catch more structure—evidence that the EDA overlap is not “no signal,” only “no simple line.” Stage 3 shows that richer engineering and XGBoost still help, especially for ranking and top-decile concentration, without claiming a transformed problem. **Table 5.2 — Stage 1 Detailed Metrics**

| Metric | Value |
|--------|-------|
| Accuracy | 0.5802 (58.0%) |
| ROC-AUC | 0.6090 |
| Precision | 0.5762 |
| Recall | 0.5786 |
| F1 | 0.5774 |
| Top-10% actual churn | ~63.1% |
| Lift vs random | ~1.27× |

**Table 5.3 — Stage 2 vs Stage 1**

| Metric | Stage 1 | Stage 2 | Change |
|--------|---------|---------|--------|
| Accuracy | 0.5802 | 0.6133 | +3.3 pp |
| ROC-AUC | 0.6090 | 0.6658 | +0.057 |
| F1 | 0.5774 | 0.6301 | +0.053 |
| Top-10% lift | 1.27× | 1.47× | +0.20× |

**Table 5.4 — Stage 3 vs Stage 2**

| Metric | Stage 2 | Stage 3 | Change |
|--------|---------|---------|--------|
| Accuracy | 0.6133 | 0.6327 | +1.9 pp |
| ROC-AUC | 0.6658 | 0.6889 | +0.023 |
| F1 | 0.6301 | 0.6330 | +0.003 |
| Top-10% lift | 1.47× | 1.58× | +0.11× |

The narrative point is the direction and size of these gaps: the large jump when moving from a linear model to trees, then a smaller but real gain from richer features and boosting.

### 5.3 Feature Importance and Coefficient Interpretation

Random Forest importances on the lean model and logistic coefficients tell a consistent story with the EDA overlays: equipment age and usage matter most.

**Table 5.5 — Top Random Forest Feature Importances**

| Rank | Feature | Importance | Business meaning |
|------|---------|------------|------------------|
| 1 | `eqpdays` | 0.174 | Older handset → higher risk |
| 2 | `months` | 0.144 | Lifecycle timing |
| 3 | `change_mou` | 0.070 | Declining engagement |
| 4 | `mou_Mean` | 0.066 | Low usage / disengagement |
| 5 | `avg3mou` | 0.051 | Short-window usage |
| 6 | `totmrc_Mean` | 0.051 | Plan / MRC level |
| 7 | `hnd_price` | 0.044 | Cheaper devices ↔ churn |
| 8 | `change_rev` | 0.042 | Revenue trajectory |
| 9 | `rev_Mean` | 0.041 | Current revenue level |
| 10 | `overage_ratio` | 0.036 | Bill shock relative to plan size |

**Table 5.6 — Top Logistic Regression Coefficients (signed)**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `eqpdays` | +0.29 | Older equipment → higher churn odds |
| Northwest / Rocky Mountain area | +0.21 | Regionally elevated risk |
| `months` | −0.21 | Longer tenure → lower churn odds |
| `mou_Mean` | −0.20 | Higher usage → lower churn odds |
| `asl_flag` = Y | −0.18 | Spending-limit flag ↔ lower churn |
| `change_mou` | −0.11 | Usage increasing → lower churn odds |

When two different model families and the Chapter 2 overlays agree, the retention story is more credible than if only one ranking were available. Correlation among MOU windows still means importance is shared, and none of these associations prove that changing a handset *causes* retention without an experiment.

### 5.4 Business View — Top 10% Risk Targeting

The operational reading of the final model is ranking, not perfection. On the test set, customers at or above the 90th percentile of predicted risk churned about **78%** of the time against a **49.6%** baseline—about **1.58×** lift. Profiling that slice shows older and cheaper handsets, much lower minutes, and a sharper recent usage drop than the average test customer. Care-call volume is lower in that slice, which is better read as possible disengagement than as satisfaction.

That profile supports concrete proposal language: prioritize upgrade offers for aging equipment, plan-fit or win-back when usage is falling, and spend retention budget on the model’s top risk band first rather than spraying the whole base.

### 5.5 Accuracy Plateau Explanation

Accuracy near 63% is not evidence that the wrong library was used. Figures 2.4 and 2.8 already showed that leavers and stayers share much of the same feature space. Linear links are weak; the label is a future decision only partly determined by the snapshot; the balanced extract prevents fake high accuracy from always predicting the majority; and extra correlated columns do not create new separation. The fair claims are that the model beats chance, ranks usefully, and hits a data-limited ceiling. The unfair claims are that 63% equals a coin flip, or that 90% should be easy on these columns alone. Taken together, these reasons defend the plateau as a property of the problem and the available features.

### 5.6 Kaggle Dataset-Card Notes vs Our Experiments

The Kaggle dataset description correctly flags MNAR-style demographic missingness, heavy tails, contract-like tenure peaks, and multi-collinearity risk. Checks in this project found about +2.5 percentage points higher churn when income is missing, and very different churn at month 12 versus month 10. A controlled add-on of MNAR flags, contract flags, `loyalty_frustration_index`, and RobustScaler-style preparation on top of tuned XGBoost did **not** move holdout accuracy in a meaningful way (~0.6335 to ~0.6333). The card therefore helps explain difficulty; it does not, by itself, unlock a new accuracy band on this split.

---

# CHAPTER 6

## SYSTEM IMPLEMENTATION AND DEPLOYMENT

### 6.1 Project Structure

The software side of the project separates data, analysis notebooks, saved models, and the web demo so each stage of the pipeline can be inspected on its own. Raw customer tables feed the notebooks; the notebooks produce trained model files; the Streamlit application loads those files to score new customers. Appendix A lists the folder layout used for this organization.

> 📊 **Figure 6.1 — System Architecture**
>
> *Figure 6.1 shows the flow from the joined customer tables through training to the saved models and the Streamlit scoring interface.*

### 6.2 Saved Models

After training, Stage 1 and Stage 2 pipelines are stored as sklearn joblib files; Stage 3 is stored as the final XGBoost bundle. `final_model.joblib` is the deployment alias of the tuned rich model. `manifest.json` records feature lists and metadata. Large artifacts may be gitignored; they are regenerated from the modeling notebook when needed.

**Table 6.1 — Saved Model Files**

| File | Role |
|------|------|
| `before_logistic_regression_lean.joblib` | Stage 1 baseline |
| `before_random_forest_lean.joblib` | Stage 2 |
| `final_model.joblib` | Stage 3 primary (app default) |

### 6.3 Model Bundler (`XGBBundle`)

The final model cannot be only a raw booster file. Scoring must repeat train-time clipping, column order, imputation, and one-hot encoding. The `XGBBundle` wrapper packages the preprocessor, model, clip bounds, and feature lists so `predict_proba` on a DataFrame stays consistent with training. Defining that class in a real module—not only in `__main__`—is what makes `joblib.load` work from Streamlit and scripts. Helpers rebuild derived columns for the final model and the lean columns for the earlier pipelines, so the application does not depend on a live notebook session.

### 6.4 Streamlit Web Application

The Streamlit app loads the chosen joblib model, accepts either a single customer form or a batch CSV, and returns churn probability plus a simple risk band. Users can compare the final XGBoost model with the earlier logistic and forest baselines. The app is a demonstration of the trained artifacts, not a claim that a probability alone is a retention strategy. Run it with `streamlit run streamlit_app/app.py` from the project root after installing requirements.

---
# CHAPTER 7


## CONCLUSION

### 7.1 Summary of Findings

This project delivered a complete, documented churn pipeline on 100,000 telecom customers:

1. **EDA** established join integrity, missingness asymmetry, ~50/50 labels, weak linear churn links, MOU/revenue redundancy, and heavy class overlap.
2. **Feature engineering** produced leakage-safe lean and rich matrices biased toward actionable levers (handset age, usage decline, overages, segments).
3. **Model evolution** showed LogReg (~58% / AUC 0.61) → RF (~61% / 0.67) → tuned XGBoost (~63.3% / 0.69) with top-10% lift rising to **1.58×**.
4. **Deployment** packaged models for a Streamlit demo via `XGBBundle`.

**Table 7.1 — Achievements Summary**

| Criterion | Achievement |
|-----------|-------------|
| Models | 3 stages (LogReg → RF → XGBoost) |
| Final accuracy | ~63.3% |
| Final ROC-AUC | ~0.689 |
| Top-10% lift | ~1.58× (~78% churn in top decile) |
| Gain vs LogReg | +5.3 pp accuracy, +0.080 AUC |
| App | Streamlit single + batch prediction |
| Documentation | This project book (self-contained chapters + appendices) |

### 7.2 Advantages of the Project

- End-to-end lifecycle: raw CSVs → EDA → FE → models → UI  
- Decisions are **evidence-linked** (each upgrade justified)  
- Honest multi-metric evaluation including lift and plateau discussion  
- Reproducible seeds, pipelines, and saved artifacts  
- Business-readable retention profile (older/cheaper handsets, falling usage)

### 7.3 Disadvantages / Limitations

- Accuracy ceiling ~63% from class overlap and missing real-world drivers  
- ~50/50 label may be a teaching balance, not live incidence  
- Random holdout ≠ temporal validation  
- Importance ≠ causation; upgrade offers need experiments to prove impact  
- Lift analysis is not a full ROI model (offer cost / acceptance / CLV omitted)

### 7.4 Future Work

| Horizon | Item |
|---------|------|
| Short | Time-based split; threshold tuning for business costs |
| Medium | SHAP explanations; prediction API; monitoring |
| Long | Cost-sensitive learning; A/B tests of retention offers; richer operational features |

---
## APPENDICES

### Appendix A: Project Folder Structure

```
data_mining_ucsy/
├── data/telecom/          # Client.csv, Record.csv
├── notebooks/             # EDA and model-training notebooks
├── models/                # Saved Stage 1–3 model files (final_model.joblib)
├── streamlit_app/         # Web demo for single and batch prediction
├── src/                   # Shared model wrapper used at score time
├── config/                # UI default values for the demo form
├── requirements.txt       # Python dependencies
└── (this project book)    # Self-contained written report
```

### Appendix B: Installation Guide

```bash
# 1. Enter the project directory
cd data_mining_ucsy

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place Client.csv and Record.csv under data/telecom/
#    Dataset: https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records

# 5. Run the EDA and training notebooks from the notebooks folder

# 6. Launch the web demo
streamlit run streamlit_app/app.py
```

### Appendix C: XGBoost Hyperparameter Search Space

| Parameter | Values Searched | Selected |
|-----------|----------------|----------|
| `max_depth` | 4, 5, 6, 7, 8 | **7** |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 | **0.03** |
| `n_estimators` | 300, 400, 500, 600 | **600** |
| `subsample` | 0.7, 0.8, 0.9 | **0.8** |
| `colsample_bytree` | 0.6, 0.7, 0.8 | **0.7** |
| `min_child_weight` | 1, 3, 5 | **1** |
| `reg_lambda` | 1.0, 2.0, 5.0 | **2.0** |
| Search method | `RandomizedSearchCV` | n_iter=20, cv=3 |
| Scoring | `roc_auc` | CV best ≈ 0.686 |

### Appendix D: EDA Figures Index

The following figures are used in Chapter 2 of this book:

| Figure | Description |
|--------|-------------|
| Figure 2.1 | Missing value percentages by column (Client vs Record) |
| Figure 2.2 | Target variable class balance (churn 0 vs 1) |
| Figure 2.3 | Univariate histograms of key usage and billing features |
| Figure 2.4 | Feature distributions overlaid by churn status |
| Figure 2.5 | Churn rate per categorical segment |
| Figure 2.6 | Churn rate by geographic area |
| Figure 2.7 | Pearson correlation heatmap of selected numeric features |
| Figure 2.8 | Joint pairplot of key metrics colored by churn |
| Figure 2.9 | Lifetime (Client) vs recent (Record) usage and revenue scatters |

---

### Appendix E: Column Dictionary

Source: Company A–style historical telecom data (`data/telecom/Client.csv`, `data/telecom/Record.csv`).

---

#### Dataset Overview

| File | Rows × Cols | Grain | Role |
|------|-------------|-------|------|
| `Client.csv` | 100,000 × 50 | One row per customer | Account, lifetime usage/billing, demographics, handset |
| `Record.csv` | 100,000 × 51 | One row per customer | Mean monthly usage/billing, call quality, tenure, **churn** |

Both tables cover the same 100,000 customers. `Customer_ID` is unique in each file and forms a perfect 1:1 match. Merged analysis table: **100,000 × 100** columns.

---

#### How the Two Tables Relate

| Column | Meaning | Relationship |
|--------|---------|--------------|
| `Customer_ID` | Unique customer identifier | Primary / foreign key linking `Client` ↔ `Record` |

- **`Client`** = *who the customer is* and *lifetime account history* (household size, credit, income, handset inventory, life-to-date calls/revenue/minutes, recent 3-/6-month averages).
- **`Record`** = *how they used the service recently* (mean monthly revenue, minutes, overages, dropped/blocked calls) plus the outcome **`churn`** and **`months`** in service.

| Theme | In `Record` | In `Client` |
|-------|-------------|-------------|
| Revenue | `rev_Mean`, `totmrc_Mean`, `ovrrev_Mean`, `change_rev` | `totrev`, `adjrev`, `avgrev`, `avg3rev`, `avg6rev` |
| Usage (minutes) | `mou_Mean`, `change_mou`, peak/off-peak MOU | `totmou`, `adjmou`, `avgmou`, `avg3mou`, `avg6mou` |
| Call volume | `attempt_Mean`, `complete_Mean`, `plcd_*`, `recv_*` | `totcalls`, `adjqty`, `avgqty`, `avg3qty`, `avg6qty` |
| Quality / friction | `drop_*`, `blck_*`, `unan_*`, `custcare_*` | — |
| Tenure / loyalty | `months`, `churn` | `eqpdays`, `phones`, `models` |
| Household / demo | — | `uniqsubs`, `actvsubs`, kids, `income`, `area`, … |

---

#### `Record.csv` — Column Meanings

**Revenue & Overage**

| Column | Meaning |
|--------|---------|
| `rev_Mean` | Mean monthly revenue (charge amount) |
| `totmrc_Mean` | Mean total monthly recurring charge |
| `da_Mean` | Mean number of directory-assisted calls |
| `ovrmou_Mean` | Mean overage minutes of use |
| `ovrrev_Mean` | Mean overage revenue |
| `vceovr_Mean` | Mean revenue of voice overage |
| `datovr_Mean` | Mean revenue of data overage |
| `roam_Mean` | Mean number of roaming calls |
| `change_mou` | % change in monthly minutes vs previous 3-month average |
| `change_rev` | % change in monthly revenue vs previous 3-month average |

**Call Quality & Completion**

| Column | Meaning |
|--------|---------|
| `drop_vce_Mean` | Mean dropped (failed) voice calls |
| `drop_dat_Mean` | Mean dropped (failed) data calls |
| `blck_vce_Mean` | Mean blocked (failed) voice calls |
| `blck_dat_Mean` | Mean blocked (failed) data calls |
| `unan_vce_Mean` | Mean unanswered voice calls |
| `unan_dat_Mean` | Mean unanswered data calls |
| `plcd_vce_Mean` | Mean attempted voice calls placed |
| `plcd_dat_Mean` | Mean attempted data calls placed |
| `recv_vce_Mean` | Mean received voice calls |
| `recv_sms_Mean` | *(not documented by provider)* |
| `comp_vce_Mean` | Mean completed voice calls |
| `comp_dat_Mean` | Mean completed data calls |
| `drop_blk_Mean` | Mean dropped or blocked calls |
| `attempt_Mean` | Mean attempted calls |
| `complete_Mean` | Mean completed calls |

**Customer Care & Special Call Types**

| Column | Meaning |
|--------|---------|
| `custcare_Mean` | Mean customer care calls |
| `ccrndmou_Mean` | Mean rounded MOU of customer care calls |
| `cc_mou_Mean` | Mean unrounded MOU of customer care calls |
| `inonemin_Mean` | Mean inbound calls lasting < 1 minute |
| `threeway_Mean` | Mean three-way calls |
| `callfwdv_Mean` | Mean call-forwarding calls |
| `callwait_Mean` | Mean call-waiting calls |

**Minutes of Use (Completed / Received / Peak)**

| Column | Meaning |
|--------|---------|
| `mou_Mean` | Mean monthly minutes of use |
| `mou_cvce_Mean` | Mean unrounded MOU of completed voice calls |
| `mou_cdat_Mean` | Mean unrounded MOU of completed data calls |
| `mou_rvce_Mean` | Mean unrounded MOU of received voice calls |
| `owylis_vce_Mean` | Mean outbound wireless-to-wireless voice calls |
| `mouowylisv_Mean` | Mean unrounded MOU of outbound wireless-to-wireless voice |
| `iwylis_vce_Mean` | *(not documented; likely inbound wireless-to-wireless voice count)* |
| `mouiwylisv_Mean` | Mean unrounded MOU of inbound wireless-to-wireless voice |
| `peak_vce_Mean` | Mean inbound + outbound peak voice calls |
| `peak_dat_Mean` | Mean peak data calls |
| `mou_peav_Mean` | Mean unrounded MOU of peak voice calls |
| `mou_pead_Mean` | Mean unrounded MOU of peak data calls |
| `opk_vce_Mean` | Mean off-peak voice calls |
| `opk_dat_Mean` | Mean off-peak data calls |
| `mou_opkv_Mean` | Mean unrounded MOU of off-peak voice calls |
| `mou_opkd_Mean` | Mean unrounded MOU of off-peak data calls |

**Outcome & Tenure**

| Column | Meaning |
|--------|---------|
| `churn` | Churn between 31–60 days after observation date (0 = stayed, 1 = churned) |
| `months` | Total months in service |
| `Customer_ID` | Customer key (join to Client) |

---

#### `Client.csv` — Column Meanings

**Household / Account Flags**

| Column | Meaning |
|--------|---------|
| `uniqsubs` | Number of unique subscribers in the household |
| `actvsubs` | Number of active subscribers in the household |
| `new_cell` | New cell phone user |
| `crclscod` | Credit class code |
| `asl_flag` | Account spending limit flag |

**Lifetime & Rolling Billing / Usage**

| Column | Meaning |
|--------|---------|
| `totcalls` | Total calls over customer lifetime |
| `totmou` | Total minutes of use over lifetime |
| `totrev` | Total revenue |
| `adjrev` | Billing-adjusted total revenue over lifetime |
| `adjmou` | Billing-adjusted total minutes over lifetime |
| `adjqty` | Billing-adjusted total number of calls over lifetime |
| `avgrev` | Average monthly revenue over lifetime |
| `avgmou` | Average monthly minutes over lifetime |
| `avgqty` | Average monthly number of calls over lifetime |
| `avg3mou` / `avg3qty` / `avg3rev` | Avg monthly MOU / calls / revenue over previous 3 months |
| `avg6mou` / `avg6qty` / `avg6rev` | Avg monthly MOU / calls / revenue over previous 6 months |

**Geography, Device, Lifestyle**

| Column | Meaning |
|--------|---------|
| `prizm_social_one` | Social group letter |
| `area` | Geographic area |
| `dualband` | Dual-band handset indicator |
| `refurb_new` | Handset refurbished (`R`) or new (`N`) |
| `hnd_price` | Current handset price |
| `phones` | Number of handsets issued |
| `models` | Number of models issued |
| `hnd_webcap` | Handset web capability |
| `truck` | Truck indicator |
| `rv` | RV indicator |
| `ownrent` | Home owner / renter status |
| `lor` | Length of residence |
| `dwlltype` | Dwelling unit type |
| `marital` | Marital status |
| `adults` | Number of adults in household |
| `infobase` | InfoBase match |
| `income` | Estimated income |
| `numbcars` | Known number of vehicles |
| `HHstatin` | Premier household status indicator |
| `dwllsize` | Dwelling size |
| `forgntvl` | Foreign travel dummy |
| `ethnic` | Ethnicity roll-up code |
| `kid0_2` … `kid16_17` | Child present in age band (household) |
| `creditcd` | Credit card indicator |
| `eqpdays` | Age (days) of current equipment |
| `Customer_ID` | Customer key (join to Record) |

---

#### Ambiguous / Undocumented Fields

| Column | Note |
|--------|------|
| `recv_sms_Mean` | Listed as NaN / not explained by provider |
| `iwylis_vce_Mean` | Listed as NaN; naming suggests inbound wireless-to-wireless voice |
| `Customer_ID` | Not explained in provider glossary, but clearly the join key |

---



---

*End of Project Book — Telecom Customer Churn Prediction*
*Semester IX | IS-212 | University of Computer Studies, Yangon*
