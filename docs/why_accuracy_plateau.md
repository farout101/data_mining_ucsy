# Why Accuracy Stays Around ~0.60–0.63

This note explains **why** telecom churn prediction in this project tops out near **60–63% accuracy** (and ~0.69 ROC-AUC), even after feature engineering and moving from logistic regression → Random Forest → tuned XGBoost.

It is **not** mainly because we “forgot a better algorithm.” The limit is mostly in the **data and the problem**, not in a missing library.

Related evidence:

- EDA: [`../main.ipynb`](../main.ipynb), [`residue.ipynb`](../residue.ipynb)
- Modeling: [`../modeling.ipynb`](../modeling.ipynb), [`model_evolution.md`](model_evolution.md)
- FE choices: [`feature_engineering.md`](feature_engineering.md)
- Kaggle card notes + our re-test: [`kaggle_notes_vs_accuracy.md`](kaggle_notes_vs_accuracy.md)

---

## 1. Short answer

| Claim | Reality in this project |
|-------|-------------------------|
| “Accuracy should be 90%+ if ML is done right” | Unrealistic here |
| “~63% means the model is useless” | Too harsh — ranking/lift still helps |
| “We can push accuracy much higher with the same labels & features” | Unlikely without new information or leakage |

**Accuracy ~0.63 means:** for a random held-out customer, the model’s yes/no guess is right about **63 times out of 100**.  
Many stayers and leavers **look alike** on the features we have, so perfect separation is impossible.

---

## 2. What we actually achieved (so the ceiling is visible)

| Stage | Model | Test accuracy | ROC-AUC | Top-10% lift |
|-------|--------|---------------|---------|--------------|
| 1 | Logistic Regression (lean) | ~0.58 | ~0.61 | ~1.27× |
| 2 | Random Forest (lean) | ~0.61 | ~0.67 | ~1.47× |
| 3 | XGBoost tuned (rich FE) | ~**0.63** | ~**0.69** | ~**1.58×** |

Going from a linear model to strong tree boosting **did** help — but gains **shrank**. That pattern usually means: *you are approaching the information limit of the features*, not that one more algorithm will jump to 85%.

---

## 3. Reason 1 — Classes overlap (the main reason)

EDA pairplots and churn overlays showed **stayers and churners occupy much of the same space** on revenue, minutes, tenure, etc.

- Some leavers have **high** usage.  
- Some stayers have **old** phones and **falling** minutes.  
- Signals like `eqpdays` help on average, but they do not draw a clean fence.

When two groups overlap heavily, even a perfect algorithm has a high **Bayes error** (unavoidable mistakes). Accuracy then sits in a “modest” band.

**Picture:** if red and blue dots are mixed together, no curve separates them cleanly. Trees and XGBoost draw better fences than a straight line — but they cannot invent separation that isn’t there.

---

## 4. Reason 2 — Weak single-feature links to churn

Linear correlation with `churn` peaked around **|r| ≈ 0.11** (`eqpdays`). Most usage/revenue correlations were weaker.

That means:

- No “magic column” that almost determines leave/stay.  
- Logistic regression (~58%) was expected to struggle.  
- Trees help by combining weak clues, but **weak clues in, modest accuracy out**.

---

## 5. Reason 3 — The label is hard by design

`churn` here = left in a **31–60 day window after** the observation snapshot.

So the model sees **today’s** profile/usage and must guess a **future** leave decision driven by:

- competitor offers  
- life events  
- price changes  
- network issues not fully in the table  
- random timing of when someone finally cancels  

Many of those drivers are **not in Client/Record**. Features explain *propensity*, not destiny → accuracy cannot hit near-certainty.

---

## 6. Reason 4 — ~50/50 churn is a special (hard) setup

In this extract, churn is about **half** leave / half stay.

| Effect | Why it matters |
|--------|----------------|
| Accuracy is “honest” | You can’t get 90% by always saying “stay” (that only scores ~50%) |
| Harder than rare-churn tasks | On rare churn, high accuracy is often fake (always predict stay) |
| Still overlapping | Balance doesn’t create separability — it only removes a trivial baseline |

So ~63% is a **real** lift over ~50% chance/majority, but it will never look like a flashy 95% accuracy score people see on unbalanced toy datasets.

> Note: real carriers often have much lower monthly churn. This file may be **balanced for teaching**, which makes accuracy a fairer headline metric and also makes “high accuracy” harder to fake.

---

## 7. Reason 5 — Features repeat each other (limited new information)

Many MOU/revenue columns are **highly correlated** (recent mean vs 3-/6-month vs lifetime).

Adding all of them rarely adds new signal; it mostly repeats “how much they use / pay.”  
We already tried **richer FE + boosting**; accuracy only moved from ~0.61 → ~0.63. That small bump supports the idea that **extra columns ≠ extra truth**.

---

## 8. Reason 6 — Accuracy is a blunt metric for this use case

Even with AUC ~0.69, a hard cutoff at 0.5 forces every customer into stay/churn and counts every mistake equally.

The model is better as a **ranker**:

- Top 10% highest risk → ~**78%** actually churned (**~1.58×** vs random)

So “stuck at 0.63 accuracy” understates value if the goal is **who to call first**, not **perfect individual prophecy**.

---

## 9. What would be needed to go *much* higher

To push accuracy a lot (e.g. toward 75–85%+), you typically need **new information**, not another sklearn model:

| Possible addition | Why it might help |
|-------------------|-------------------|
| Competitor / port-out signals | Direct leave intent |
| Ticket / network quality detail | Service pain |
| Contract end dates / price plan changes | Timing of leave |
| App engagement, NPS, complaint text | Soft dissatisfaction |
| True time-series of many months | Stronger trajectory |
| Cleaner, less overlapping label definition | Easier target |

**Dangerous “improvements” (don’t):**

- Using `Customer_ID` as a feature  
- Leaking future columns  
- Tuning on the test set until accuracy looks great  

Those raise the number without raising real skill.

---

## 10. How to say this in the assignment (suggested wording)

> Churn prediction accuracy plateaus around 60–63% because leavers and stayers overlap substantially in the available usage, billing, and handset features; linear associations with churn are weak (|r| ≲ 0.11); and the label is a future event only partly determined by the snapshot we observe. Stronger models (RF, XGBoost) improve ranking (AUC ~0.69, top-decile lift ~1.58×) but cannot create separation that the data do not contain. Therefore we treat the model as a targeting tool, not as a near-perfect classifier.

---

## 11. One-sentence summary

**Accuracy sits near ~0.6 because many customers who stay and leave look similar in this dataset — ML can rank risk better than chance, but it cannot invent a sharp boundary that isn’t in the features.**
