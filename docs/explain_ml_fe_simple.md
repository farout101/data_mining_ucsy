# Feature Engineering & Machine Learning — Explained Simply

*Plain language. This is what we did after looking at the data (EDA).*

Related: [`modeling.ipynb`](modeling.ipynb) · [`feature_engineering.md`](feature_engineering.md) · [`machine_learning.md`](machine_learning.md) · [`model_evolution.md`](model_evolution.md)

---

## Big picture in one story

1. We **looked** at the data (EDA).  
2. We **cleaned and chose** smart columns (**feature engineering**).  
3. We taught computer models to guess who might leave (**machine learning**).  
4. We started simple, saw the score was only okay, then **upgraded** the model.

---

## Part A — Feature engineering (making good inputs)

### What is a “feature”?

A **feature** is a column the model is allowed to use as a clue.

Example features:

- How old is the phone?
- How many minutes lately?
- Did minutes go down?
- Which area do they live in?

The **target** is the answer we want: **churn** (leave or stay).  
The model must **not** peek at the answer while learning from clues — and must not use the customer ID as a “magic name tag.”

---

### Why we didn’t use all ~100 columns

Using everything sounds smart. Often it isn’t.

Reasons we kept a **lean** set first:

1. **Many columns say the same thing** (correlated minutes/revenue windows).  
2. Some columns are **mostly empty** (cars, big house details).  
3. Too many columns make the story **hard to explain** to a teacher or a boss.  
4. Extra copies can confuse linear models and split importance in tree models.

**Rule we used for minutes/money:**

```text
Keep:
  • recent level   (mou_Mean, rev_Mean, …)
  • one short average (avg3mou)
  • change         (change_mou, change_rev)
  • overages

Skip (at first):
  • lifetime totals + every other average window
  • lots of peak/off-peak twin columns
```

Later, for a stronger model, we carefully added a **few** more (like `avg6mou`, `avgrev`) — still not “every column on Earth.”

---

### Simple fixes we did

| Problem | What we did | Kid reason |
|---------|-------------|------------|
| Phone age negative | Set to 0 minimum | Age can’t be negative |
| Wild huge “change” numbers | Clip extreme tails using **training** data only | Don’t let a few weird rows boss the model |
| Empty category boxes | Call them `"Missing"` | “Unknown” is information too |
| Empty numbers | Fill with the **median** (middle value) | Better than inventing zero for everything |
| Customer ID | **Dropped** | Names/IDs aren’t real “why they left” clues |

**Important:** clip limits and “middle values” are learned from the **training** half only, then applied to the test half — so we don’t cheat.

---

### New features we invented (derived)

| New column | Plain meaning |
|------------|----------------|
| `mou_decline_flag` | “Did minutes go down?” yes/no |
| `overage_ratio` | Extra charges compared to the normal monthly fee |
| (later) `eqpdays × change_mou` | Old phone **and** usage change together |
| (later) `mou_per_month`, `drop_rate`, … | Simple rates that are easier to compare |

These are like making a “study score” from homework grades — still honest, just clearer.

---

### Train / test split (the quiz rule)

We hide 20% of customers as a **test quiz**.

- Train on 80%  
- Test on 20% the model has never graded before  
- Keep the leave/stay mix the same (**stratify**)

If you only check score on the homework you already studied, you can feel smart and still fail the real quiz.

---

## Part B — Machine learning (teaching the computer)

### What the computer is trying to do

Given the features, guess:

- Will this person **leave**? (yes/no)  
- And: how **risky** are they? (a probability from 0 to 1)

---

### Model 1 — Logistic Regression (the straight-line student)

**Idea:** Draw a mostly straight, additive recipe:  
“a bit of this column + a bit of that column → chance of leaving.”

**Why we tried it first:**

- Easy to explain  
- Good school baseline  

**What happened:**

| Score | About |
|-------|--------|
| Accuracy | **58%** |
| ROC-AUC | **0.61** |
| Top 10% lift | **~1.27×** |

**Why we moved on:**  
EDA said leave/stay isn’t a simple straight pattern. 58% is only a little better than a coin (~50%). The linear student was trying hard but the puzzle is curvy.

---

### Model 2 — Random Forest (a crowd of “if-then” trees)

**Idea:** Many decision trees vote.

Example tree thoughts:

- If phone is very old **and** minutes fell → more risk  
- If usage is high and phone is new → less risk  

**Why we switched from logistic regression:**

| Logistic problem | Forest help |
|------------------|-------------|
| Likes straight relationships | Learns zig-zag rules and combos |
| Weak linear clues in EDA | Matches what we saw in plots |

**What happened (same lean features):**

| Score | About |
|-------|--------|
| Accuracy | **61%** |
| ROC-AUC | **0.67** |
| Top 10% lift | **~1.47×** |

Better! Still not “amazing,” but clearly smarter than the linear baseline **on the same clues**.

---

### Model 3 — XGBoost (the improved champion)

**Idea:** Build trees in a sequence; each new tree tries to fix the last one’s mistakes (boosting). Then we lightly **tuned** settings.

**What we also did:** richer features (a few more good clues, still not all 100 columns).

**What happened:**

| Score | About |
|-------|--------|
| Accuracy | **~63%** |
| ROC-AUC | **~0.69** |
| Top 10% lift | **~1.58×** |

This is the **current primary model**.

Full play-by-play: [`model_evolution.md`](model_evolution.md).

---

## “Is 63% just a coin flip?”

**Short answer: No — but don’t brag like it’s 95%.**

| Comparison | Meaning |
|------------|---------|
| Coin / always guess “stay” | ~50% |
| Our best accuracy | ~63% |
| Ranking quality (AUC) | ~0.69 (0.5 = random order) |
| Top 10% riskiest people | ~78% actually left (**1.58×** better than picking randomly) |

So:

- As a perfect yes/no for every person → still makes mistakes.  
- As a **priority list** for who to call with a keep-you offer → useful.

That’s the real business point.

---

## How we score models (simple)

| Metric | Plain meaning |
|--------|----------------|
| **Accuracy** | % of quiz answers right |
| **Precision** | When we say “will leave,” how often true |
| **Recall** | Of everyone who left, how many we caught |
| **F1** | Balance of precision & recall |
| **ROC-AUC** | How good is the **risk ranking**? |
| **Lift (top 10%)** | In the riskiest tenth, how many leavers vs average |

For retention, **AUC + lift** often matter more than accuracy alone.

---

## What the models say matters (same story as EDA)

The strong clues kept showing up:

1. Older equipment (`eqpdays`)  
2. Tenure / months  
3. Minutes and **falling** minutes  
4. Plan / overage style signals  
5. Sometimes area or handset web type  

When the model’s “important columns” match what we saw by eye, we trust the project more.

---

## Things we refused to do (good manners)

| Temptation | Why no |
|------------|--------|
| Use `Customer_ID` as a feature | Cheating with name tags |
| SMOTE (fake extra leavers) | Already ~50/50 leave/stay |
| Peek at the test quiz to pick features | Also cheating |
| Promise “phone upgrade causes stay” | Model shows **clues**, not proof of cause |

---

## Tiny glossary

| Word | Kid-friendly meaning |
|------|----------------------|
| **Feature** | A clue column |
| **Label / target** | The answer (churn) |
| **Train / test** | Study set / hidden quiz |
| **Pipeline** | Assembly line: clean → encode → model |
| **One-hot** | Turn “New York / Chicago” into yes/no columns |
| **Impute** | Fill empty cells carefully |
| **Tree model** | Learns if-then rules |
| **Boosting** | New trees fix old mistakes |
| **Baseline** | Simple first model to beat |

---

## One-sentence takeaway

We cleaned and chose non-repeating clues, started with a simple straight model (~58%), switched to tree models because leaving isn’t linear (~61%), then improved with better features and XGBoost (~63% accuracy, stronger targeting lift) — useful for finding risky customers, not for perfect crystal-ball guesses.
