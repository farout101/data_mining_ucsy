# Why This EDA Order? — Reasoning Behind the Approach

This document explains the **logic behind** the exploratory analysis in `main.ipynb` — not the steps themselves, but **why that sequence and those choices**.

Companion write-up with figures and fill-in placeholders: [`eda_approach_rationale.md`](eda_approach_rationale.md).

---

## The core idea

EDA here is not “plot everything.” It is a **funnel of trust**:

> Rule out a bad analysis before you commit to a model or a business story.

A model will always produce *some* accuracy and feature importance. Without this path, you cannot tell whether that importance is real signal, leakage, a mostly-missing column, or three copies of the same revenue variable.

Each step answers a different risk.

---

## The funnel (why this order)

```text
1. Identity (join)        → Are we talking about the same customers?
2. Quality (missing)      → What data can we actually trust?
3. Target (churn)         → What problem are we solving, and how do we score it?
4. Shape (distributions)  → Do averages lie? Are there wild tails?
5. Contrast (by churn)    → What differs for people who leave?
6. Segments (cats / area) → Who can we target operationally?
7. Structure (corr/pairs) → What’s redundant? Is churn linearly separable?
8. Sanity (Client↔Record) → Does the joined world still make sense?
```

Later charts only matter if earlier questions pass. If the join is wrong or a column is 40% empty, pretty plots become misleading evidence.

---

## Why each stage exists

### 1. Join on `Customer_ID` first

**Risk if skipped:** You compare usage in one table to demographics in another without knowing they line up — or you silently drop / duplicate people.

**Why first:** Everything else assumes Client and Record describe the **same** population. Identity must be fixed before interpretation.

`Record` carries recent behavior + `churn`. `Client` carries profile + lifetime history. Without the join, you cannot relate behavior to profile or train a model that uses both.

---

### 2. Missingness before “interesting” plots

**Risk if skipped:** You build a narrative around income, dwelling, or cars when half the rows are blank. That is not evidence; it is selection bias.

**Why second:** Missingness decides drop / impute / encode-as-“Missing.” In this dataset, demographics are sparse; usage fields are mostly complete. That asymmetry should drive preprocessing — and honesty in the proposal.

---

### 3. Target distribution before feature hunting

**Risk if skipped:** You optimize accuracy when churn is 5%, or you invent feature stories before knowing the outcome is usable.

**Why third:** `churn` frames the whole problem:

- class balance → which metrics are honest
- baseline → what “good” means (~0.50 if you always predict one class here)
- business framing → retention vs something else

Looking at features first tempts a story that may not match the outcome.

---

### 4. Univariate shape (histograms) before group comparisons

**Risk if skipped:** Means and correlations get dominated by skew and extreme tails (`change_mou`, revenue outliers).

**Why fourth:** You need to know whether the “average customer” is a fiction. Shape tells you what to clip, log, or treat carefully before comparing stayers vs churners.

---

### 5. Stayed vs churned contrasts (the business heart)

**Risk if skipped:** You get model feature importance with no interpretable lever for a proposal.

**Why fifth:** The assignment cares about a **business recommendation**, not only a score. Overlaying distributions and mean gaps asks: *what differs for leavers?* Those gaps become candidates for action (handset age, usage drop, plan fit) — not only model inputs.

This step is biased toward **actionable** telecom signals on purpose.

---

### 6. Segments (categoricals and geography)

**Risk if skipped:** You only think in continuous scores; ops and marketing cannot “target a float.”

**Why sixth:** Companies act on **groups** — credit flag, device type, region. Even when national churn is ~50%, a segment or area above baseline can justify a focused campaign. Labeling missing as its own group avoids pretending nulls don’t exist.

---

### 7. Correlations and pairplots after you know the target and shapes

**Risk if skipped:** You feed ~100 collinear columns into a model and then over-interpret importance among near-duplicates.

**Why seventh:**

- **Correlation heatmap** — spot redundant families (lifetime vs 3-/6-month vs mean MOU/revenue) and see that linear `|r|` with churn can be weak even when real signal exists.
- **Pairplot** — check joint space. If churners and stayers heavily overlap, a single rule or linear story won’t separate them — which is a reason trees / non-linear models can still help.

These are structure checks, not the first place you look for “the answer.”

---

### 8. Client vs Record sanity last among EDA checks

**Risk if skipped:** You trust a join that is technically keyed but semantically wrong (definitions / windows don’t line up).

**Why eighth:** Lifetime averages (`avgmou`, `avgrev`) should relate directionally to recent means (`mou_Mean`, `rev_Mean`), but not be identical. Agreement builds confidence; weird scatter clouds would force you back to definitions and join logic.

---

## Why not jump straight to modeling?

Because modeling **always** returns numbers. EDA’s job is to make those numbers **defensible**:

| Without this EDA | What can go wrong |
|------------------|-------------------|
| No join audit | Wrong population |
| No missingness check | Stories on empty columns |
| No target check | Wrong metric / wrong problem |
| No shape check | Outliers drive “insights” |
| No churn contrast | Importance without action |
| No segments | No operable recommendation |
| No redundancy check | Inflated / unstable narratives |

---

## Why these features (not random columns)?

The contrast step prefers fields that map to **retention levers** in telecom:

| Signal family | Example columns | Why it matters for a proposal |
|---------------|-----------------|-------------------------------|
| Equipment ageing | `eqpdays`, `hnd_price` | Upgrade / replacement offers |
| Usage decline | `change_mou`, `mou_Mean` | Win-back, plan fit, engagement |
| Bill shock | overages, MRC / revenue | Plan redesign, alerts |
| Service friction | drops, care calls | Network / support quality |
| Tenure / loyalty | `months` | Timing of retention outreach |

So the EDA is intentionally biased toward features you could **act on**, matching a proposal-first assignment rather than a pure leaderboard mindset.

---

## One-sentence summary

**Do EDA in this order so every chart answers a prior risk — identity → quality → problem framing → honest distributions → actionable contrasts → operable segments → model-ready structure — instead of producing plots that look complete but cannot support a recommendation.**

---

## Your notes

> **What I want my proposal to argue, based on this logic:**
> - …
> - …

> **Which stage of the funnel matters most for *my* story, and why:**
> - …
