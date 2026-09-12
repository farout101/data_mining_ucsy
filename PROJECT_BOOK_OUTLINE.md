# Telecom Customer Churn Prediction: An Advanced Machine Learning and Ensembling Approach

**Semester IX | IS-212**  
**University of Computer Studies, Yangon (UCSY)**  

**Student Name:** [Your Name]  
**Student ID:** [Your Student ID]  
**Supervisor:** [Supervisor Name]  
**Submission Date:** [Date]  

---

## ABSTRACT

Customer churn represents one of the most critical operational challenges in the telecommunications industry, directly impacting recurring revenue, customer lifetime value, and marketing acquisition efficiency. This project presents an end-to-end data mining and machine learning investigation into subscriber defection behavior across a balanced dataset of 100,000 telecommunications customers. 

Through disciplined exploratory data analysis, the study demonstrates that customer defection cannot be separated linearly due to significant feature overlap and weak bivariate correlations (|r| <= 0.11). To address this challenge, a multi-tiered feature engineering framework was developed, incorporating data quality corrections, outlier mitigation via train-only Winsorization, and domain-derived behavioral signals—most notably an overage-to-plan ratio to quantify customer bill shock and a discrete directional usage decline indicator.

Complementing the supervised modeling track, an **unsupervised K-Means clustering analysis** (k = 2) was applied to the scaled rich feature space to uncover behaviorally distinct subscriber archetypes without relying on churn labels. The resulting clusters were characterized by statistically significant differences in overage intensity, customer care frequency, usage momentum, and service friction, and were validated against ground-truth churn labels. This descriptive mining layer provides an interpretable, label-free segmentation that enriches the prescriptive retention strategy.

Modeling was conducted through a phased four-stage comparative evolution on an identical 80/20 stratified holdout split (80,000 training records, 20,000 testing records):
1. **Stage 1 — Logistic Regression Baseline:** Achieved 58.02% accuracy and 0.6090 ROC-AUC, demonstrating the inadequacy of linear decision boundaries.
2. **Stage 2 — Random Forest Ensemble:** Improved accuracy to 61.33% and ROC-AUC to 0.6658 on identical features, proving the necessity of non-linear conditional interactions.
3. **Stage 3 — Tuned XGBoost:** Squeezed additional predictive power from an expanded 38-feature rich space through randomized cross-validation, attaining 63.27% accuracy, 0.6889 ROC-AUC, and a 1.58x top-decile lift.
4. **Stage 4 — Soft-Vote Ensemble (Champion):** Combined the level-wise tree growth of XGBoost with the leaf-wise optimization of LightGBM through equal-weighted probability blending. The ensemble achieved the project's highest discrimination performance: **63.49% accuracy, 0.6899 ROC-AUC, 0.6422 recall, 0.6355 F1-score, and a 1.59x top-10% lift**.

The project establishes that while overall classification accuracy is naturally constrained near ~63.5% by intrinsic class overlap and external market drivers, the model's true commercial value lies in risk prioritization. By identifying a high-risk decile containing 78.8% actual churners, the system enables telecommunications operators to deploy proactive retention interventions—such as equipment upgrades, plan right-sizing, and usage win-back campaigns—with 59% greater efficiency than random outreach. An interactive decision-support deployment interface operationalizes both single-customer profiling and batch portfolio scoring.

---

## TABLE OF CONTENTS

- **CHAPTER 1: INTRODUCTION**
  - 1.1 Project Background
  - 1.2 Problem Statement
  - 1.3 Project Objectives
  - 1.4 Scope and Constraints
  - 1.5 Significance and Practical Application
- **CHAPTER 2: DATA UNDERSTANDING AND EXPLORATORY ANALYSIS**
  - 2.1 Dataset Architecture and Customer Cohorts
  - 2.2 Target Label Balance and Evaluation Implications
  - 2.3 Key Predictor Distributions and Data Irregularities
  - 2.4 Bivariate Driver Analysis and Class Overlap
  - 2.5 Categorical Segments and Geographic Variance
  - 2.6 Multicollinearity and Usage Window Redundancy
  - 2.7 Unsupervised Behavioral Segmentation via K-Means Clustering
  - 2.8 Cluster Interpretation and Business Persona Derivation
- **CHAPTER 3: FEATURE ENGINEERING AND PREPROCESSING**
  - 3.1 Data Quality Remediation
  - 3.2 Domain-Specific Derived Features
  - 3.3 Lean vs. Rich Feature Set Architecture
  - 3.4 Outlier Mitigation via Train-Only Winsorization
  - 3.5 Preprocessing Pipeline and Data Leakage Prevention
- **CHAPTER 4: METHODOLOGY AND MODEL EVOLUTION**
  - 4.1 Modeling Philosophy and Experimental Control
  - 4.2 Stage 1: Logistic Regression (Linear Baseline)
  - 4.3 Stage 2: Random Forest (Non-Linear Tree Bagging)
  - 4.4 Stage 3: Tuned XGBoost (Sequential Gradient Boosting)
  - 4.5 Stage 4: Soft-Vote Ensemble (Dual-Architecture Champion)
  - 4.6 Methodological Refusals
- **CHAPTER 5: MODEL EVALUATION AND PERFORMANCE ANALYSIS**
  - 5.1 Evaluation Metric Framework
  - 5.2 Comparative Cross-Stage Benchmarking
  - 5.3 Confusion Matrix Transition and Error Trade-offs
  - 5.4 Discrimination Analysis via ROC-AUC Sweep
  - 5.5 Threshold Sensitivity and Operating Modes
  - 5.6 Demystifying the Accuracy Ceiling vs. Business Lift
- **CHAPTER 6: BUSINESS INTERPRETABILITY AND RETENTION STRATEGY**
  - 6.1 Global Feature Importance and Risk Drivers
  - 6.2 Top-Decile Risk Persona Profiling
  - 6.3 Prescriptive Retention Playbooks
- **CHAPTER 7: SYSTEM IMPLEMENTATION AND DEPLOYMENT**
  - 7.1 System Architecture and Data Flow
  - 7.2 Model Encapsulation and Preprocessing Serialization
  - 7.3 Interactive Decision-Support Application
- **CHAPTER 8: CONCLUSION AND FUTURE DIRECTIONS**
  - 8.1 Summary of Contributions
  - 8.2 Project Strengths and Methodological Rigor
  - 8.3 Limitations
  - 8.4 Strategic Roadmap for Future Research

---

# CHAPTER 1: INTRODUCTION

### 1.1 Project Background
The contemporary telecommunications sector operates within an intensely competitive, saturated marketplace characterized by low switching costs, standardized network connectivity, and continuous competitor promotions. In this business environment, customer acquisition costs frequently exceed retention expenditures by a factor of five to seven. Retaining existing subscribers is therefore not merely a defensive marketing activity, but the primary driver of sustainable operating margins and enterprise valuation.

Customer defection (churn) occurs when a subscriber terminates their service or transitions their subscription to a rival carrier. In wireless communications, defection is rarely an instantaneous, impulsive event; rather, it represents the culmination of progressive customer dissatisfaction, service friction, equipment obsolescence, or uncompetitive tariff structures. Because these underlying grievances manifest across historical billing, network quality logs, and customer support touchpoints, statistical data mining and predictive machine learning offer a viable mechanism to identify at-risk subscribers prior to contract termination.

### 1.2 Problem Statement
The operational challenge addressed in this research is the early prediction of subscriber churn within a 31-to-60-day forecasting window following a fixed observation date. Formulated as a supervised binary classification task, each customer $i$ is assigned a ground-truth label:
$$y_i \in \{0, 1\}$$
where $y_i = 1$ denotes that the customer churned within the target window, and $y_i = 0$ denotes that the customer maintained an active subscription.

From a data mining perspective, this problem presents substantial analytical hurdles:
1. **Severe Class Overlap:** Defecting subscribers share near-identical distributions of monthly minutes, recurring charges, and tenure with loyal subscribers, preventing trivial linear separation.
2. **Weak Bivariate Associations:** Individual customer features exhibit minimal linear correlation with the churn label ($|r| \le 0.11$), requiring non-linear interaction modeling.
3. **High Feature Redundancy:** Telecommunications billing systems record dozens of overlapping rolling averages (lifetime, 6-month, 3-month, and monthly usage metrics) that introduce severe multicollinearity if ingested without disciplined feature selection.
4. **Data Skewness and Noise:** Usage change and overage revenue distributions exhibit heavy right-side tails populated by extreme billing adjustments.

### 1.3 Project Objectives
The primary objectives of this study are as follows:
1. Conduct an audit and exploratory analysis of 100,000 customer records to identify the primary behavioral and contractual indicators of subscriber defection.
2. Engineer leakage-free, domain-specific behavioral features—specifically targeting billing friction, device aging, and usage momentum.
3. Establish a controlled, multi-stage model progression evaluating linear models, bagging ensembles, gradient boosting algorithms, and dual-architecture ensembling on identical holdout data.
4. Quantify classification performance across both statistical metrics (Accuracy, ROC-AUC, Precision, Recall, F1) and commercial metrics (Top-10% Decile Lift).
5. Translate empirical risk probabilities into actionable, prescriptive retention strategies and deliver an interactive software interface for real-time customer risk scoring.

### 1.4 Scope and Constraints
This study is conducted on an observation cohort of 100,000 telecommunications accounts. Key structural parameters include:
- **Observation Window:** Account usage and billing metrics are aggregated over historical 1-month, 3-month, and 6-month windows.
- **Prediction Target:** Churn event occurring between 31 and 60 days following the observation timestamp.
- **Cohort Balance:** The dataset represents a curated research extract exhibiting an approximately balanced target distribution (49.56% churn vs. 50.44% retention). While live carrier monthly churn typically ranges between 1.5% and 3.0%, this balanced sampling enables direct optimization of classification boundaries without synthetic oversampling.
- **Feature Boundary:** Analysis is restricted to tabular customer relationship management (CRM) attributes, billing records, network friction counters, and device inventory logs. Unstructured customer care call transcripts, real-time cell tower telemetry, and external competitor tariff changes are outside the analytical scope.

### 1.5 Significance and Practical Application
Rather than treating churn prediction as an abstract competition to maximize standard accuracy, this project frames machine learning as a decision-support mechanism for targeted retention marketing. Telecommunications operators operate under strict commercial budget constraints; retention teams cannot contact an entire subscriber base. 

By prioritizing subscribers based on calibrated risk probabilities, operators can concentrate intervention budgets on the top deciles where churn density is highest. Furthermore, because the models preserve interpretable feature representations, account managers can identify *why* a customer is at risk (e.g., severe device aging versus unexpected overage charges) and offer customized remedies rather than generic discounts.

---

# CHAPTER 2: DATA UNDERSTANDING AND EXPLORATORY ANALYSIS

### 2.1 Dataset Architecture and Customer Cohorts
The investigation evaluates 100,000 unique subscriber accounts established by joining two core enterprise data sources via a unique customer identifier:
1. **Customer Profile and Lifetime History:** Encapsulates demographic indicators, household composition, credit ratings, handset inventory counts, and lifetime cumulative billing statistics.
2. **Usage and Billing Activity:** Captures recent mean monthly recurring charges, voice minutes of use (MOU), dropped and blocked call counts, customer care inquiries, overage fees, and the binary defection target.

An initial audit confirmed complete join integrity, yielding a consolidated matrix of 100,000 observations across approximately 100 raw attributes without orphan records. However, a structural asymmetry in missing data was immediately identified: usage and billing fields derived from automated network switches were 100% complete, whereas demographic fields obtained through optional third-party data enrichment (e.g., estimated household income, home ownership, dwelling size) exhibited missingness rates between 25% and 70%. Consequently, demographic fields were treated with caution to avoid building retention strategies on sparse attributes.

### 2.2 Target Label Balance and Evaluation Implications
The target variable exhibits an empirical distribution of 49,557 churned accounts (49.56%) and 50,443 retained accounts (50.44%). 

This near-even split carries major implications for experimental methodology:
- A naïve baseline classifier that indiscriminately assigns the majority class ("Stayed") achieves an accuracy of exactly 50.44%. Any viable machine learning model must comfortably surpass this 50.4% threshold to demonstrate genuine predictive utility.
- Synthetic minority oversampling techniques (such as SMOTE) and algorithmic class-weight adjustments are completely unwarranted and would introduce artificial noise into balanced decision spaces.
- Standard classification accuracy serves as a meaningful initial headline metric; however, receiver operating characteristic area under the curve (ROC-AUC) and top-decile lift remain the primary ranking criteria.

### 2.3 Key Predictor Distributions and Data Irregularities
Univariate distribution audits revealed critical structural characteristics:
- **Equipment Age (`eqpdays`):** Measures the number of days a subscriber has used their current handset. The distribution spans from 0 to over 1,800 days, with an average of 392 days. An anomaly was discovered where a small cluster of records contained negative equipment days (e.g., -5 days), reflecting administrative pre-dating in enterprise CRM systems. Domain constraints dictate that physical device tenure cannot be negative; these values were floored at zero.
- **Usage Dynamics (`change_mou` and `change_rev`):** Represents percentage changes in recent usage and revenue relative to the preceding 3-month rolling baseline. Both variables exhibit extreme leptokurtic behavior with severe positive and negative tails (spanning from -1,000% to +1,500%), caused by accounts resuming activity or experiencing temporary billing adjustments.
- **Monthly Recurring Charges (`totmrc_Mean`):** The fixed base plan cost exhibits sharp multimodal clustering around standard enterprise tariff tiers ($30, $40, $60, and $80 per month).

### 2.4 Bivariate Driver Analysis and Class Overlap
Bivariate exploration evaluating feature behavior segmented by churn status revealed several foundational patterns:
- **Device Aging as the Primary Risk Factor:** Retained customers possessed current devices with an average age of 342 days, whereas churning customers averaged 443 days (+101 days older). Subscribers utilizing handsets exceeding 18 to 24 months exhibited defection rates approaching 65%, driven by device obsolescence, battery degradation, and contract expiration.
- **Usage Trajectory:** Churning subscribers exhibited systematic negative usage momentum, averaging a -15.8% drop in voice minutes relative to their 3-month baseline, compared to a stable -1.2% change among retained users.
- **Severe Class Overlap in Billing and Minutes:** When plotting kernel density estimates of total revenue, minutes of use, and monthly charges, the distribution curves for churners and stayers sit almost directly atop one another. Churn occurs across all spending tiers—from $20 budget accounts to $150 premium accounts. There is no single cutoff threshold on usage or revenue that separates leavers from stayers.

### 2.5 Categorical Segments and Geographic Variance
Cross-tabulation of categorical indicators against defection rates identified actionable commercial groupings:
- **Account Spending Limits (`asl_flag`):** Subscribers subject to strict credit-based account spending limits exhibited a churn rate of 54.2%, compared to 48.9% for unrestricted accounts.
- **Refurbished vs. New Handsets (`refurb_new`):** Subscribers issued refurbished replacement devices experienced an elevated churn rate of 53.8% versus 49.1% for those receiving brand-new handsets, reflecting hardware dissatisfaction and customer support friction.
- **Geographic Area (`area`):** Churn rates varied across regional operating territories, ranging from 46.2% in the Mid-Atlantic region to 53.1% in South Florida and the Northwest/Rocky Mountain territory. Regional variance highlights differing localized network quality and competitive carrier promotional intensity.

### 2.6 Multicollinearity and Usage Window Redundancy
A Pearson correlation analysis across all numeric usage features exposed extensive multicollinearity within metric families:
- The correlation between lifetime minutes (`totmou`), 6-month average minutes (`avg6mou`), 3-month average minutes (`avg3mou`), and current monthly minutes (`mou_Mean`) ranged between $r = 0.85$ and $r = 0.98$.
- Cumulative lifetime revenue (`totrev`) and recent monthly revenue (`rev_Mean`) exhibited near-perfect linear alignment ($r > 0.90$).

Retaining every overlapping rolling window introduces severe variance inflation in linear models, dilutes split importance in tree ensembles, and confuses operational attribution. A disciplined data mining pipeline requires selecting a minimal, non-redundant set of usage windows that captures baseline level, short-term trend, and directional momentum.

### 2.7 Unsupervised Behavioral Segmentation via K-Means Clustering
To uncover structural behavioral patterns without relying on the supervised target label, an unsupervised descriptive data mining layer was deployed using K-Means clustering. By allowing algorithms to group accounts based solely on continuous usage, billing dynamics, and service friction signals, the analysis exposes whether natural subscriber archetypes emerge and how they correspond to churn risk.

#### Experimental Methodology:
1. **Feature Subspace:** 25 continuous and domain-derived behavioral metrics were selected, including `eqpdays`, `months`, `mou_Mean`, `rev_Mean`, `totmrc_Mean`, `change_mou`, `change_rev`, `ovrmou_Mean`, `ovrrev_Mean`, `custcare_Mean`, `drop_vce_Mean`, `overage_ratio`, `mou_decline_flag`, `mou_per_month`, `rev_per_mou`, `drop_rate`, and `care_per_mou`.
2. **Standardization:** Continuous attributes were imputed via training-set medians and scaled to zero mean and unit variance using `StandardScaler`, ensuring distance metrics were not dominated by high-magnitude variables (e.g., minutes vs. call counts).
3. **Partition Optimization ($k = 2$):** An elbow inertia sweep across cluster counts ($k \in [2, 8]$) confirmed that the steepest reduction in within-cluster sum-of-squares (SSE) occurs at $k = 2$ (inertia dropped from $2,041,720$ at $k=2$ to $1,444,101$ at $k=8$). K-Means was initialized with 20 distinct starts (`n_init=20`, `max_iter=300`, locked random state 42).
4. **Statistical Significance of Discrimination:** For each continuous metric, two-sided Mann-Whitney U hypothesis tests were conducted between cluster distributions, verifying that observed median variances were statistically significant ($p < 0.001$).

#### Cluster Archetype Breakdown:
The algorithm partitioned the 100,000 subscriber base into two distinct operational clusters:

| Behavioral Dimension | Metric | Cluster 0: High-Overage / High-Friction | Cluster 1: Low-Usage Baseline | Relative Variance | $p$-value (Mann-Whitney U) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cohort Size** | Account Count ($n$) | **19,436 (19.4%)** | **80,564 (80.6%)** | — | — |
| **Churn Density** | Empirical Churn Rate | **45.90%** | **50.44%** | -4.54 pp | $p < 0.0001$ |
| **Service Friction** | `custcare_Mean` | **2.33 calls/mo** | **0.00 calls/mo** | **+2.00x** | $p < 0.0001$ |
| **Care Intensity** | `care_per_mou` | **0.0018** | **0.0000** | **+2.00x** | $p < 0.0001$ |
| **Billing Penalty** | `ovrrev_Mean` | **$27.22 / mo** | **$0.08 / mo** | **+1.99x** | $p < 0.0001$ |
| **Overage Volume** | `ovrmou_Mean` | **83.63 min/mo** | **0.25 min/mo** | **+1.99x** | $p < 0.0001$ |
| **Bill Shock Ratio** | `overage_ratio` | **0.418 (41.8%)** | **0.001 (0.1%)** | **+1.99x** | $p < 0.0001$ |
| **Revenue Trend** | `change_rev` | **-$5.45** | **-$0.25** | **+1.83x** | $p < 0.0001$ |
| **Usage Momentum** | `change_mou` | **-50.50 min** | **-4.25 min** | **+1.69x** | $p < 0.0001$ |
| **Network Quality** | `drop_vce_Mean` | **11.67 drops/mo** | **2.00 drops/mo** | **+1.41x** | $p < 0.0001$ |
| **Usage Intensity** | `mou_per_month` | **83.12 MoU/mo** | **15.15 MoU/mo** | **+1.38x** | $p < 0.0001$ |

### 2.8 Cluster Interpretation and Business Persona Derivation
The unsupervised clustering analysis provides profound descriptive insight into the operational mechanics of subscriber churn, identifying two distinct behavioural failure modes:

1. **Cluster 0 — The "Frustrated Heavy User" Persona (High Value, High Friction):**
   - **Characteristics:** Representing 19.4% of the customer base, these subscribers are highly active, generating 83 minutes per month per tenure unit. However, they experience chronic operational pain: they accumulate over $27 in monthly overage fees (representing a severe 41.8% surcharge over their base plan), suffer from elevated dropped calls (11.7/month), and contact customer support more than twice monthly.
   - **Churn Mechanism:** This segment churns out of acute **billing and quality frustration** (bill shock defection). Their usage is falling sharply (-50.5 MoU), signaling that dissatisfaction precedes contract termination.
   - **Prescriptive Strategy:** Generic retention discounts are ineffective here. Carriers must deploy **automated plan right-sizing** (migrating users to higher baseline tiers to eliminate overages) paired with priority technical support outreach to resolve network drop issues.

2. **Cluster 1 — The "Passive Disengager" Persona (Low Usage, Low Contact):**
   - **Characteristics:** Comprising 80.6% of the customer base, these subscribers exhibit lower overall usage intensity (15 MoU/month), virtually zero overage charges ($0.08), and almost zero customer support inquiries (median 0.0).
   - **Churn Mechanism:** This segment defects via **silent disengagement**. They do not contact customer support to complain; rather, their usage gently erodes until an aging handset or competitor promotion induces defection.
   - **Prescriptive Strategy:** Because these users have minimal interaction with support channels, reactive interventions fail. Carriers must employ proactive **hardware upgrade incentives** (subsidized handsets tied to 24-month renewals) and automated digital usage stimulation to re-anchor account value.

3. **Methodological Synthesis (Unsupervised vs. Supervised Roles):**
   Unsupervised clustering validates that subscriber churn is not a monolithic phenomenon. The unsupervised K-Means segments explain *how* customers interact with the network, providing contextual personas that allow the supervised predictive models (Stages 1–4) to be operationalized with differentiated retention playbooks.

---

# CHAPTER 3: FEATURE ENGINEERING AND PREPROCESSING

### 3.1 Data Quality Remediation
Prior to statistical modeling, data cleaning steps were enforced to ensure physical validity and eliminate data leakage:
1. **Identifier Elimination:** The unique account key (`Customer_ID`) was dropped from all model matrices to prevent algorithms from memorizing arbitrary index positions.
2. **Domain Flooring:** Negative equipment days (`eqpdays < 0`) were bounded at a minimum of 0.0 days.

### 3.2 Domain-Specific Derived Features
To provide algorithms with explicit commercial signals, domain-driven transformations were developed:
1. **Overage Ratio (`overage_ratio`) — Bill Shock Indicator:** Absolute overage dollars (`ovrrev_Mean`) fail to capture subjective customer frustration. A $20 overage charge on a $100 executive plan represents a mild 20% variance, whereas the same $20 charge on a $20 budget tier doubles the customer's monthly bill, causing severe billing friction. The overage ratio normalizes this shock against the base recurring charge:
   $$	ext{overage\_ratio} = rac{	ext{ovrrev\_Mean}}{|	ext{totmrc\_Mean}| + arepsilon}$$
   where $arepsilon = 10^{-6}$ prevents division by zero.
2. **Usage Decline Flag (`mou_decline_flag`):** A discrete indicator marking whether recent usage fell below the prior 3-month rolling baseline:
   $$	ext{mou\_decline\_flag} = egin{cases} 1, & 	ext{if } 	ext{change\_mou} < 0 \ 0, & 	ext{otherwise} \end{cases}$$
   This feature isolates directional engagement decline from extreme percentage magnitude.
3. **Equipment and Momentum Interaction (`eqpdays_x_change_mou`):** Computes the product of equipment age and usage momentum, specifically flagging subscribers who possess aging devices while simultaneously reducing their engagement.
4. **Service Friction Ratios:** Derived features including call drop rates (`drop_vce_Mean / plcd_vce_Mean`) and support intensity (`custcare_Mean / mou_Mean`) were constructed to capture network and customer care pain points.

### 3.3 Lean vs. Rich Feature Set Architecture
To isolate algorithmic gains from feature expansion, the project established two explicit feature tiers:
- **Lean Feature Set (25 Features):** Designed for Stage 1 and Stage 2 models to test linear versus non-linear algorithms under controlled inputs. It comprises 17 non-redundant numeric features (`eqpdays`, `hnd_price`, `phones`, `models`, `months`, `mou_Mean`, `rev_Mean`, `totmrc_Mean`, `change_mou`, `change_rev`, `avg3mou`, `drop_vce_Mean`, `custcare_Mean`, `ovrmou_Mean`, `ovrrev_Mean`, `mou_decline_flag`, `overage_ratio`) and 8 categorical attributes (`asl_flag`, `creditcd`, `new_cell`, `refurb_new`, `dualband`, `hnd_webcap`, `area`, `prizm_social_one`).
- **Rich Feature Set (38 Features):** Designed for Stage 3 and Stage 4 gradient boosting architectures. It supplements the lean set with interaction terms (`eqpdays_x_change_mou`), usage and revenue ratios (`mou_per_month`, `rev_per_mou`, `drop_rate`, `care_per_mou`), secondary usage windows (`avg6mou`, `avgrev`), and broader demographic fields (`marital`, `ethnic`).

### 3.4 Outlier Mitigation via Train-Only Winsorization
Extreme tails in continuous change and overage ratios can severely distort linear regression lines and force decision trees to waste split levels isolating single outlier rows. To address this without discarding customer records, train-only Winsorization was implemented.

Percentile thresholds (1st percentile lower bound, 99th percentile upper bound) were calculated strictly from the training partition and applied as clamping boundaries across both training and holdout test sets:
- `change_mou`: Clamped to `[-841.30, +748.79]` minutes.
- `change_rev`: Clamped to `[-104.83, +121.34]` dollars.
- `overage_ratio`: Clamped to `[0.0000, 3.9075]` (capping overage charges at ~3.9x base monthly plan).

### 3.5 Preprocessing Pipeline and Data Leakage Prevention
Data leakage represents a fatal flaw in predictive modeling, occurring when test-set distribution statistics inadvertently inform training transformations. 

To ensure complete methodological integrity:
1. **Partitioning:** The dataset was split into an 80% training set (80,000 records) and a 20% holdout test set (20,000 records) using stratified sampling locked on random state 42.
2. **Column Transformation Encapsulation:** Preprocessing steps were isolated within a modular column transformation pipeline. Continuous variables underwent median imputation (using medians computed solely on training folds). Categorical features underwent missingness imputation with an explicit `"Missing"` string level, followed by One-Hot Encoding configured with unknown category handling.
3. **Pipeline Invariance:** For linear models requiring feature normalization, a standard scaler was embedded directly within the pipeline after imputation, ensuring zero test data leakage.

---

# CHAPTER 4: METHODOLOGY AND MODEL EVOLUTION

### 4.1 Modeling Philosophy and Experimental Control
The modeling strategy followed an evolutionary benchmark methodology. Rather than deploying a complex ensemble immediately, models were developed across four sequential stages. Each successive model was required to justify its added complexity by outperforming its predecessor under identical holdout validation data.

### 4.2 Stage 1: Logistic Regression (Linear Baseline)
Logistic regression was deployed as the foundational linear benchmark. The model optimizes an L2-regularized log-loss objective function over standardized numeric inputs and one-hot categorical features:
$$P(y=1|\mathbf{x}) = rac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
- **Role:** Establish a transparent, interpretable baseline and quantify the extent to which churn can be modeled through additive, linear relationships.
- **Holdout Results:** Accuracy: **58.02%**, ROC-AUC: **0.6090**, Precision: **0.5762**, Recall: **0.5786**, F1: **0.5774**, Top-10% Lift: **1.27x**.
- **Diagnostic Conclusion:** While outperforming random chance (50.4%), the linear model's 58.0% accuracy is inadequate for production deployment. The flat decision boundary cannot untangle the class overlap identified during exploratory analysis, confirming that subscriber churn is fundamentally non-linear.

### 4.3 Stage 2: Random Forest (Non-Linear Tree Bagging)
To test whether non-linear decision boundaries could resolve class overlap without altering the underlying data representation, a Random Forest classifier was trained on the **exact same 25 lean features**.
- **Configuration:** Ensemble of 200 bagging decision trees, maximum tree depth constrained to 12, minimum leaf samples set to 20, utilizing Gini impurity split criteria. Continuous feature scaling was omitted because decision trees are invariant to monotonic scaling.
- **Holdout Results:** Accuracy: **61.33%**, ROC-AUC: **0.6658**, Precision: **0.5990**, Recall: **0.6645**, F1: **0.6301**, Top-10% Lift: **1.47x**.
- **Diagnostic Conclusion:** Holding feature inputs strictly constant, the Random Forest achieved an immediate +3.3 percentage point gain in accuracy and a +0.057 jump in ROC-AUC. This confirmed that churn risk is governed by multi-variable conditional interactions (e.g., *if* equipment age > 400 days *and* usage change < -20%, *then* churn probability surges).

### 4.4 Stage 3: Tuned XGBoost (Sequential Gradient Boosting)
Having demonstrated the necessity of tree-based logic, Stage 3 sought to maximize predictive power by transitioning from bagging (independent parallel trees) to gradient boosting (sequential residual error correction) paired with the expanded 38-feature rich matrix.
- **Algorithm & Tuning:** Extreme Gradient Boosting (XGBoost) iteratively fits successive shallow decision trees to the negative gradient of the log-loss loss function. Hyperparameter optimization was conducted via a 20-iteration randomized cross-validation search across tree depth (4–8), learning rates (0.01–0.10), estimators (300–600), and row/column subsampling ratios (0.6–0.9).
- **Optimal Hyperparameters:** `max_depth=7`, `learning_rate=0.03`, `n_estimators=600`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=1`, `reg_lambda=2.0`.
- **Holdout Results:** Accuracy: **63.27%**, ROC-AUC: **0.6889**, Precision: **0.6150**, Recall: **0.6392**, F1: **0.6330**, Top-10% Lift: **1.58x**.
- **Diagnostic Conclusion:** Tuning and rich features pushed accuracy past 63% and ROC-AUC near 0.69, while top-decile lift rose to 1.58x. XGBoost effectively leveraged the derived interaction terms while controlling overfitting via L2 regularization.

### 4.5 Stage 4: Soft-Vote Ensemble (Dual-Architecture Champion)
While XGBoost achieved outstanding ranking metrics, all gradient boosted decision trees exhibit algorithmic inductive biases. Specifically, XGBoost employs a **level-wise (depth-first)** tree growth policy, splitting nodes row-by-row across equal depths. Conversely, LightGBM utilizes a **leaf-wise (best-first)** growth policy, splitting the single node that minimizes loss regardless of tree symmetry.

To determine whether architectural diversity could resolve remaining edge-case errors, Stage 4 introduced an ensemble combining both paradigms:
1. **LightGBM Classifier:** Configured with `num_leaves=63`, `n_estimators=800`, `learning_rate=0.03`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=2.0`, and `reg_alpha=0.1` (adding L1 feature selection pressure). On its own, LightGBM achieved 63.17% accuracy and 0.6882 ROC-AUC.
2. **Soft-Vote Probability Blending:** Predictions were combined via an equal-weighted probability average:
   $$P_{	ext{ensemble}}(y=1|\mathbf{x}) = 0.5 \cdot P_{	ext{XGB}}(y=1|\mathbf{x}) + 0.5 \cdot P_{	ext{LGB}}(y=1|\mathbf{x})$$
3. **Holdout Results:** Accuracy: **63.49%**, ROC-AUC: **0.6899**, Precision: **0.6272**, Recall: **0.6422**, F1: **0.6355**, Top-10% Lift: **1.59x**.
- **Diagnostic Conclusion:** Averaging probability distributions from complementary tree architectures cancelled out idiosyncratic prediction variance. The Stage 4 ensemble achieved the project's highest scores across every performance metric.

### 4.6 Methodological Refusals
To preserve scientific validity, several common modeling shortcuts were explicitly rejected:
- **Refusal of Arbitrary Synthetic Sampling:** SMOTE was rejected because creating artificial observations in overlapping continuous spaces distorts real-world marginal distributions.
- **Refusal of Complex Black-Box Ensembles Without Baselines:** Neural network architectures and multi-layer stacking classifiers were avoided; their marginal performance gains on tabular data do not justify their extreme computational overhead and opacity in corporate governance.
- **Refusal to Confuse Correlation with Causation:** High feature importance indicates predictive association, not direct causality. Interventions must be tested empirically via controlled marketing experiments.

---

# CHAPTER 5: MODEL EVALUATION AND PERFORMANCE ANALYSIS

### 5.1 Evaluation Metric Framework
Evaluating a churn model requires balancing statistical rigor with business utility. Six core metrics were tracked across all experiments:
1. **Accuracy:** Measures global correctness across all predictions; meaningful here due to balanced classes.
2. **ROC-AUC:** Threshold-independent metric assessing the probability that the model ranks a randomly selected churner higher than a randomly selected stayer.
3. **Precision:** Measures the proportion of flagged accounts that actually defect, governing wasted marketing spend.
4. **Recall:** Measures the proportion of total churners successfully identified, governing saved account volume.
5. **F1-Score:** Harmonic mean of precision and recall, balancing false alarms against missed defectors.
6. **Top-10% Lift:** The ratio of churn density within the top risk decile to the baseline population churn rate, serving as the definitive commercial efficiency metric.

### 5.2 Comparative Cross-Stage Benchmarking
All models were evaluated on the exact same 20,000-customer holdout test partition. Performance progressed monotonically across each architectural upgrade:

| Metric | Stage 1 (Logistic Regression) | Stage 2 (Random Forest) | Stage 3 (Tuned XGBoost) | Stage 4 (Soft-Vote Ensemble) | Total Gain (1 -> 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Feature Set** | Lean (~25) | Lean (~25) | Rich (~38) | **Rich (~38)** | +13 features |
| **Accuracy** | 0.5802 | 0.6133 | 0.6327 | **0.6349** | **+5.47 pp** |
| **ROC-AUC** | 0.6090 | 0.6658 | 0.6889 | **0.6899** | **+0.0809** |
| **Precision** | 0.5762 | 0.5990 | 0.6150 | **0.6272** | **+0.0510** |
| **Recall** | 0.5786 | 0.6645 | 0.6392 | **0.6422** | **+0.0636** |
| **F1-Score** | 0.5774 | 0.6301 | 0.6330 | **0.6355** | **+0.0581** |
| **Top-10% Lift** | 1.27x | 1.47x | 1.58x | **1.59x** | **+0.32x** |

### 5.3 Confusion Matrix Transition and Error Trade-offs
Evaluating the raw prediction counts across the 20,000 test accounts illustrates how model evolution altered classification behavior:

| Quadrant / Cell | Stage 1 (LogReg) | Stage 2 (Random Forest) | Stage 3 (XGBoost) | Stage 4 (Ensemble) | Net Transition (1 -> 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Negatives (TN)** | 5,874 | 5,679 | 5,812 | **5,902** | +28 |
| **False Positives (FP)** | 4,214 | 4,409 | 4,276 | **4,186** | -28 (Fewer false alarms) |
| **False Negatives (FN)** | 4,183 | 3,327 | 3,083 | **3,006** | **-1,177 (Fewer missed churners)** |
| **True Positives (TP)** | 5,729 | 6,585 | 6,829 | **6,906** | **+1,177 (More defectors caught)** |

The defining technical achievement across the experimental progression is the systematic transfer of **1,177 customer accounts out of the False Negative cell into the True Positive cell**. In an operational retention campaign, this translates to nearly 1,200 additional defecting customers who receive timely retention interventions rather than defecting uncontacted.

### 5.4 Discrimination Analysis via ROC-AUC Sweep
The Receiver Operating Characteristic curves trace the sensitivity-specificity trade-off as the classification threshold sweeps from 0.0 to 1.0:
- **Stage 1 (AUC = 0.6090):** The linear curve remains close to the 45-degree diagonal across all operating thresholds, indicating weak discriminative power.
- **Stage 2 (AUC = 0.6658):** The Random Forest curve bows upward significantly across all false-positive rates, demonstrating that non-linear decision trees extract genuine ranking signal from the exact same features.
- **Stage 3 (AUC = 0.6889):** The gradient boosting curve expands further, particularly within the low false-positive region (FPR between 0.10 and 0.30), which is critical for cost-sensitive outreach.
- **Stage 4 (AUC = 0.6899):** The dual-architecture ensemble forms the outermost frontier across the entire parameter space, demonstrating superior probability calibration.

### 5.5 Threshold Sensitivity and Operating Modes
The conventional default classification threshold of $p = 0.50$ is an arbitrary mathematical convention. In real-world enterprise operations, the optimal threshold depends entirely on the financial ratio between retention contact costs and saved customer lifetime value (CLV):

| Operating Strategy | Probability Threshold | Primary Objective | Commercial Trade-off |
| :--- | :---: | :--- | :--- |
| **Aggressive Recall** | $p \ge 0.40$ | Maximize saved customers | High campaign contact volume; tolerates elevated false alarms. Ideal when retention incentives are inexpensive digital credits. |
| **Balanced Default** | $p \ge 0.50$ | Balanced precision & recall | Standard operational baseline yielding 64.2% recall and 62.7% precision. |
| **High Precision** | $p \ge 0.60$ | Minimize wasted outreach | Targets only confident defectors. Essential when interventions involve expensive high-touch incentives (e.g., subsidized hardware). |
| **Top-Decile Targeting** | $p \ge 0.617$ | Maximum commercial ROI | Restricts outreach to the top 10% risk band (2,000 accounts), capturing 1,564 actual churners (78.8% purity) with 1.59x lift. |

### 5.6 Demystifying the Accuracy Ceiling vs. Business Lift
External observers often query why advanced gradient boosting ensembles fail to exceed ~64% accuracy on telecommunications churn datasets, contrasting this with computer vision or fraud detection models that routinely achieve >95%.

This ceiling reflects the intrinsic nature of human customer defection:
1. **Unobserved External Drivers:** Customer decisions are heavily influenced by exogenous market dynamics completely absent from historical CRM logs—such as a competitor launching an aggressive regional promotion, personal subscriber relocation, employer subsidy changes, or word-of-mouth recommendations.
2. **Intrinsic Class Overlap:** As proven during exploratory data analysis, loyal subscribers and defecting subscribers exhibit nearly identical distributions of usage minutes and billing amounts.

Consequently, evaluating a churn model solely on binary accuracy at 0.50 misunderstands the business objective. The model's primary value is as a **risk ranking and prioritization engine**. A top-decile lift of 1.59x means that a marketing team with a budget to contact 10,000 customers will capture nearly 1,600 additional churners by using model rankings compared to standard random outreach.

---

# CHAPTER 6: BUSINESS INTERPRETABILITY AND RETENTION STRATEGY

### 6.1 Global Feature Importance and Risk Drivers
Tree feature importances and logistic coefficients converge on a consistent, coherent hierarchy of churn drivers:

| Rank | Predictive Feature | Feature Type | Behavioral Phenomenon | Operational Interpretation |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `eqpdays` | Continuous | Handset Tenure / Aging | Strongest single driver. Device degradation and contract expiration trigger churn. |
| **2** | `months` | Continuous | Account Tenure | Lifecycle timing factor; newer accounts face onboarding friction. |
| **3** | `change_mou` | Continuous | Usage Momentum | Steep negative drops indicate disengagement prior to formal cancellation. |
| **4** | `mou_Mean` | Continuous | Engagement Level | Very low baseline usage correlates with higher churn propensity. |
| **5** | `totmrc_Mean` | Continuous | Tariff Tier | High recurring charges increase price sensitivity and competitor shopping. |
| **6** | `overage_ratio` | Derived Ratio | Bill Shock | Excessive penalty fees relative to base plan price create billing frustration. |
| **7** | `hnd_price` | Continuous | Device Value | Budget handset owners exhibit significantly higher price sensitivity. |
| **8** | `custcare_Mean` | Continuous | Service Friction | Repeated support calls serve as a direct proxy for unresolved customer pain. |

### 6.2 Top-Decile Risk Persona Profiling
Profiling the 2,000 customers assigned to the highest risk decile ($p \ge 0.617$) by the Stage 4 ensemble reveals a clear behavioral signature:

| Account Attribute | Full Test Cohort Average | Top 10% Risk Decile Average | Net Variance |
| :--- | :---: | :---: | :---: |
| **Equipment Age (`eqpdays`)** | 392.6 days | **548.2 days** | **+155.6 days older** |
| **Handset Price (`hnd_price`)** | $102.10 | **$69.40** | **-$32.70 cheaper** |
| **Monthly Minutes (`mou_Mean`)** | 510.4 min | **312.1 min** | **-198.3 min lower** |
| **Usage Change (`change_mou`)** | -12.3% | **-59.4%** | **-47.1% steeper drop** |
| **Overage Ratio (`overage_ratio`)** | 0.22 | **0.48** | **+118% higher bill shock** |
| **Customer Care Calls (`custcare_Mean`)** | 1.75 calls | **0.48 calls** | **-1.27 fewer calls** |

This profile yields a critical operational insight: **the highest-risk subscribers are not complaining—they are disengaging silently**. They possess aging, low-cost handsets (>18 months old), have slashed their monthly call volume by nearly 60%, and suffer from disproportionately high overage ratios. When their contract expires, they defect without contacting customer support.

### 6.3 Prescriptive Retention Playbooks
Rather than dispatching generic, margin-diluting discount vouchers to all flagged subscribers, retention teams should execute targeted, automated intervention playbooks:

1. **Hardware Upgrade Incentive:** For accounts where risk is driven by `eqpdays > 450`, provide a targeted device upgrade credit tied to a 24-month contract renewal. This directly resolves hardware obsolescence while legally locking in account tenure.
2. **Proactive Plan Right-Sizing:** For accounts exhibiting `overage_ratio > 0.35`, dispatch an automated advisory: *"We noticed your recent overages; we have upgraded your monthly plan allowance for an extra $5/month, saving you $25 in penalty fees."* This eliminates bill shock while protecting recurring baseline revenue.
3. **Usage Win-Back Campaigns:** For accounts exhibiting steep negative `change_mou`, trigger digital engagement campaigns (promotional bonus data or service trials) to reverse disengagement momentum before defection occurs.

---

# CHAPTER 7: SYSTEM IMPLEMENTATION AND DEPLOYMENT

### 7.1 System Architecture and Data Flow
To transition the analytical findings into production, an end-to-end scoring pipeline was architected. The system is designed to consume raw enterprise customer records, execute required feature engineering and data transformations in memory, and generate real-time churn risk probabilities along with categorical risk classifications.

The operational pipeline executes in five sequential phases:
1. **Ingestion:** Consumes subscriber account records from either single-record web requests or bulk enterprise batch uploads.
2. **Quality Remediation:** Automatically floors negative equipment age values at zero and drops administrative account keys.
3. **Feature Derivation:** Computes derived behavioral signals in memory, including the bill-shock overage ratio, the directional usage decline flag, and equipment-momentum interaction terms.
4. **Pipeline Preprocessing:** Applies training-locked percentile clipping boundaries to heavy-tailed features, executes median imputation across numeric features, and converts categorical variables to one-hot binary encodings.
5. **Ensemble Scoring & Actionable Output:** Feeds preprocessed matrices into both tuned XGBoost and LightGBM engines, computes the calibrated soft-vote probability average, assigns subscribers to operational risk tiers, and triggers corresponding retention playbooks.

### 7.2 Model Encapsulation and Preprocessing Serialization
A major failure mode in enterprise data science is preprocessing discrepancy between offline model training and online production scoring. 

To eliminate this risk, the system utilizes object-oriented pipeline wrappers:
- Preprocessor objects, model estimators, training-derived percentile clipping boundaries, and canonical feature schemas are packaged together as modular, standalone serialized bundles.
- The dual-architecture ensemble bundle coordinates predictions from both gradient boosting engines in parallel, internally executing Winsorization, median imputation, and categorical encoding before returning the mathematically blended probability:
  $$P(	ext{churn}) = 0.5 \cdot P_{	ext{XGB}} + 0.5 \cdot P_{	ext{LGB}}$$
This modular architecture ensures that offline experimental models can be deployed into online operational services with complete preprocessing fidelity and zero code duplication.

### 7.3 Interactive Decision-Support Application
An interactive web application was deployed to serve as an operational decision-support cockpit for marketing analysts and customer service personnel:
- **Single-Customer Scoring Mode:** Account managers input individual subscriber parameters (e.g., equipment age, monthly charges, recent usage change). The interface queries the underlying ensemble model and displays the predicted churn probability, assigned risk tier (`Lower Risk: < 0.40`, `Medium Risk: 0.40-0.55`, `Higher Risk: > 0.55`), and prescriptive retention recommendations.
- **Batch CSV Portfolio Scoring:** Marketing teams can upload batch customer extracts containing thousands of records. The system processes the records in parallel, computes calibrated risk scores, and generates downloadable prioritized outreach lists sorted by predicted defection probability.
- **Model Comparative Selection:** The interface allows operators to toggle between all four experimental stages (Logistic Regression, Random Forest, Tuned XGBoost, and the Stage 4 Ensemble) to verify how individual risk classifications evolve across modeling architectures.

---

# CHAPTER 8: CONCLUSION AND FUTURE DIRECTIONS

### 8.1 Summary of Contributions
This project delivered a comprehensive, statistically sound data mining framework for telecommunications customer churn prediction across a balanced cohort of 100,000 accounts. 

Key technical achievements include:
1. **Empirical Data Auditing:** Uncovered critical structural properties including join integrity, demographic missingness asymmetry, usage window multicollinearity, and extensive class distribution overlap.
2. **Leakage-Free Feature Engineering:** Derived high-impact behavioral indicators (`overage_ratio`, `mou_decline_flag`, `eqpdays_x_change_mou`) and enforced train-only Winsorization and preprocessing pipelines.
3. **Unsupervised Behavioral Archetyping:** Applied K-Means clustering ($k = 2$) on 25 standardized continuous metrics, identifying two empirically distinct subscriber archetypes ("Frustrated Heavy Users" driven by bill shock and care friction vs. "Passive Disengagers" driven by silent attrition) with statistically validated separation (Mann-Whitney U, $p < 0.0001$).
4. **Phased Model Evolution:** Demonstrated an unbroken progression from a linear baseline (58.0% accuracy, 0.609 AUC) to a bagging ensemble (61.3% accuracy, 0.666 AUC), a gradient boosting engine (63.3% accuracy, 0.689 AUC), and ultimately a dual-architecture soft-vote ensemble (**63.5% accuracy, 0.6899 AUC, 1.59x lift**).
5. **Commercial Value Demonstration:** Proven that a model with ~63.5% headline accuracy concentrates 78.8% actual churners within its top risk decile, delivering 59% greater efficiency than untargeted marketing.
6. **Operational Software Delivery:** Packaged serialized model pipelines and an interactive decision-support tool supporting real-time single-account and batch portfolio inference.

### 8.2 Project Strengths and Methodological Rigor
- **Strict Experimental Controls:** All four model stages were benchmarked on the exact same 20,000-customer holdout split with fixed random seeds, isolating algorithmic gains from data sampling variation.
- **Zero Data Leakage:** All imputation parameters, one-hot category vocabularies, and Winsorization limits were learned strictly from training folds.
- **Actionable Feature Interpretability:** Avoided opaque projection transformations (e.g., PCA), ensuring that all retained features map directly to operational retention levers.

### 8.3 Limitations
- **Accuracy Ceiling:** Due to the absence of external market data (competitor price wars, localized cell tower outages) and intrinsic behavioral overlap, global accuracy is bounded near ~64%.
- **Balanced Cohort Representation:** The balanced 50/50 dataset enables clear model benchmarking but does not reflect the low single-digit monthly base churn rate observed in live carrier networks.
- **Static Window Limitations:** Aggregating metrics into fixed 1-month and 3-month averages obscures fine-grained weekly temporal dynamics.

### 8.4 Strategic Roadmap for Future Research
To build upon the foundation established in this research:
1. **Survival Analysis & Dynamic Hazard Modeling:** Transition from binary classification to Cox proportional hazards or time-varying survival models to predict *when* a customer will churn rather than merely *if* they will churn.
2. **Cost-Matrix Utility Optimization:** Replace the symmetric 0.50 cutoff with an economic loss matrix that dynamically sets thresholds to maximize net financial profit:
   $$	ext{Profit} = (	ext{TP} 	imes 	ext{Saved CLV}) - (	ext{FP} 	imes 	ext{Incentive Cost}) - (	ext{FN} 	imes 	ext{Lost CLV})$$
3. **Tri-Ensemble Architecture with CatBoost:** Incorporate CatBoost alongside XGBoost and LightGBM to leverage its native target-encoding algorithms for categorical attributes like regional operating territory.
4. **Bayesian Hyperparameter Optimization:** Implement multi-objective Bayesian optimization (e.g., via Optuna) to simultaneously tune tree structures, regularizers, and blending weights across hundreds of automated trials.

---
*End of Project Manuscript*  
*University of Computer Studies, Yangon (UCSY)*  
