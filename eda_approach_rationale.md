# Why These EDA Approaches?

This note explains **why** each exploratory step in `main.ipynb` was done, embeds the diagrams produced by the notebook, and leaves **placeholders** so you can write your own reasoning for the assignment / business proposal.

Related files:

- Notebook: [`main.ipynb`](main.ipynb)
- Column meanings: [`telecom_column_dictionary.md`](telecom_column_dictionary.md)
- Figures folder: [`eda_figures/`](eda_figures/)

---

## How to use the placeholders

Under each section you will see a block like:

```text
> **My reason / interpretation:**
> - …
> - …
```

Replace the `…` lines with your own words (what the plot suggests for Company A, what you would recommend, what you would check next).

---

## 0. Overall EDA strategy

### Why this sequence?

EDA was ordered to answer three constraints before modeling:

1. **Structure** — Can we trust the join? What columns exist?
2. **Quality** — Where is data missing or broken?
3. **Target & signal** — What does `churn` look like, and which features differ by churn?

That order matters: if the join is wrong or missingness is extreme, later charts are misleading.

> **My reason / interpretation:**
>
> - …
> - …

---

## 1. Load both tables and join on `Customer_ID`

### Why this approach?

`Client.csv` and `Record.csv` are designed as a **1:1 customer split**:

- `Record` holds recent usage / billing and the **`churn`** label
- `Client` holds lifetime metrics, demographics, and handset info

Without joining on `Customer_ID`, you cannot relate behavior to profile — and you cannot train a churn model that uses both.

We also checked uniqueness and that both sides share the same ID set, so the merge is not silently dropping or duplicating customers.

> **My reason / interpretation:**
>
> - …
> - …

---

## 2. Missing-value profile

### Why this approach?

Missingness decides whether a column is usable, needs imputation, or should be dropped. Telecom demographics often have high null rates; usage fields are usually denser. A side-by-side Client vs Record missing chart makes that asymmetry obvious.

![Missing values — Client vs Record](eda_figures/01_missing_values.png)

**What the figure is for:** rank columns by % missing so preprocessing choices are evidence-based (not guesswork).

> **My reason / interpretation:**
>
> - …
> - …
> - Decision I would take (drop / impute / keep as “Missing” category): …

---

## 3. Target check — churn distribution

### Why this approach?

Before choosing metrics or models, you need class balance. If churn were ~5%, accuracy would be a bad headline metric. Here the plot confirms near 50/50 balance, so accuracy is at least interpretable as a starting point.

![Churn distribution](eda_figures/02_churn_distribution.png)

**What the figure is for:** confirm the outcome is usable and set expectations for baselines (~0.50 if you always predict one class).

> **My reason / interpretation:**
>
> - …
> - …
> - Metric I would prioritize for the business case: …

---

## 4. Univariate distributions of key features

### Why this approach?

Raw means hide skew and outliers. Histograms of revenue, minutes, tenure, equipment age, and usage change show:

- which features are heavily skewed
- whether clipping / log transforms are needed
- whether “average customer” is a misleading summary

![Key feature distributions](eda_figures/03_feature_distributions.png)

**What the figure is for:** understand shape and tails before comparing groups or correlating.

> **My reason / interpretation:**
>
> - …
> - …
> - Features that look skewed / need transform: …

---

## 5. Compare features by churn (distribution overlay)

### Why this approach?

Business proposals need **interpretable levers** (e.g. equipment age, usage decline), not only black-box importance. Overlaying stayed vs churned distributions is a direct, visual A/B-style check for candidate drivers.

![Feature distributions by churn](eda_figures/04_features_by_churn.png)

**What the figure is for:** spot shifts between stayers and churners (location, spread, tails).

> **My reason / interpretation:**
>
> - …
> - …
> - Features that look most different by churn: …
> - Business action this might support: …

---

## 6. Categorical / segment churn rates

### Why this approach?

Many Client fields are categorical (`asl_flag`, `creditcd`, handset type, etc.). Segment bars answer: “Does churn concentrate in a product / credit / device subgroup we can target?” A dashed overall rate makes it easy to see who is above/below baseline. Missing values are labeled explicitly so nulls are not invisible.

![Churn rate by categorical segment](eda_figures/05_categorical_churn_rates.png)

**What the figure is for:** find actionable segments, not only continuous predictors.

> **My reason / interpretation:**
>
> - …
> - …
> - Segments with above-average churn I would prioritize: …

---

## 7. Geographic churn rates

### Why this approach?

`area` is a natural operations / marketing cut. Even if national churn is ~50%, regional gaps can justify localized retention campaigns or network-quality follow-ups.

![Churn rate by geographic area](eda_figures/06_geographic_churn_rates.png)

**What the figure is for:** check whether geography is a useful segmentation axis for the proposal.

> **My reason / interpretation:**
>
> - …
> - …
> - Regions of interest and why: …

---

## 8. Correlation heatmap (selected numeric features)

### Why this approach?

With ~100 columns, many revenue/MOU fields are near-duplicates. A correlation heatmap helps:

1. see redundancy (lifetime vs 3-/6-month vs mean monthly)
2. see which features have any linear association with `churn`
3. avoid stuffing highly collinear features into a fragile model narrative

![Correlation heatmap](eda_figures/07_correlation_heatmap.png)

**What the figure is for:** feature-family awareness and a first pass at churn association (knowing linear `|r|` can be weak even when trees find signal).

> **My reason / interpretation:**
>
> - …
> - …
> - Feature groups I would keep vs drop as redundant: …

---

## 9. Pairplot — joint view of business metrics

### Why this approach?

Single histograms miss interactions. A pairplot of tenure, revenue, minutes, MRC, and equipment age (sampled for speed) checks whether churners occupy a different region of the joint space.

![Pairplot of key metrics by churn](eda_figures/08_pairplot.png)

**What the figure is for:** see joint patterns / overlaps; if classes heavily overlap, a simple rule won’t separate churn alone.

> **My reason / interpretation:**
>
> - …
> - …
> - Does churn look linearly separable here? What does that imply for modeling?: …

---

## 10. Client lifetime vs Record recent metrics (sanity check)

### Why this approach?

After a join, you should ask: “Do the two tables agree directionally?” Plotting Client lifetime averages (`avgmou`, `avgrev`, `avg3mou`) against Record mean monthly fields checks that the merge is coherent and that short-term vs long-term views are related but not identical.

![Client vs Record usage/revenue sanity checks](eda_figures/09_client_vs_record_sanity.png)

**What the figure is for:** validate the join and motivate keeping both recent and lifetime features (or choosing one window carefully).

> **My reason / interpretation:**
>
> - …
> - …
> - Would I prefer recent (`*_Mean`) or lifetime (`avg*`) features in the model, and why?: …

---

## Summary table — approach → purpose

| Step | Approach                   | Main purpose                                   |
| ---- | -------------------------- | ---------------------------------------------- |
| 1    | 1:1 join on`Customer_ID` | Build one analysis table with target + profile |
| 2    | Missingness bars           | Guide drop / impute / encode-as-missing        |
| 3    | Churn bar chart            | Choose evaluation metric & baseline            |
| 4    | Univariate histograms      | Spot skew, outliers, transform needs           |
| 5    | Churn overlays             | Find interpretable retention levers            |
| 6    | Categorical churn rates    | Find targetable segments                       |
| 7    | Area churn rates           | Check geographic opportunity                   |
| 8    | Correlation heatmap        | Reduce redundancy; scan linear churn links     |
| 9    | Pairplot                   | Check joint separability                       |
| 10   | Client vs Record scatters  | Sanity-check join & time windows               |

> **My overall narrative for the business proposal:**
>
> - Problem I am solving: …
> - Evidence from EDA I will cite: …
> - Recommendation direction (retention offer / handset upgrade / usage win-back / …): …
> - What I will model next and why: …

---

## Figure checklist

All images below live in `eda_figures/` (exported from `main.ipynb`):

1. `01_missing_values.png`
2. `02_churn_distribution.png`
3. `03_feature_distributions.png`
4. `04_features_by_churn.png`
5. `05_categorical_churn_rates.png`
6. `06_geographic_churn_rates.png`
7. `07_correlation_heatmap.png`
8. `08_pairplot.png`
9. `09_client_vs_record_sanity.png`

If you re-run the notebook and plots change, re-export figures (or ask to refresh this markdown) so the document stays in sync.
