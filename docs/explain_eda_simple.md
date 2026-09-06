# Telecom Data — Explained Simply (EDA & How Columns Relate)

*Written in plain language so anyone can follow. This is about looking at the data before any machine learning.*

Related: [`main.ipynb`](main.ipynb) · [`telecom_column_dictionary.md`](telecom_column_dictionary.md) · [`eda_approach_rationale.md`](eda_approach_rationale.md)

---

## What is this project about?

Imagine a phone company (“Company A”) with **100,000 customers**.

Some customers **keep** their plan.
Some customers **leave** (that’s called **churn**).

We have two big spreadsheets about the same people:

| File           | Like…data_mining_ucsy
Private
￼
Watch
0
 (0)
Fork0 (0)
￼
￼
Star0 (0)
￼
￼
main
1 Branch
0 Tags
￼
t
T
Add file￼
Add file
￼
￼
Code
Latest commit
￼
farout101
update
5b9465c
 ·
last month
History
4 Commits
Folders and files | What’s inside                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `Client.csv` | A customer**profile card**                                                                                                                                                                                                       | Who they are, old usage history, phone age, area |
| `Record.csv` | A recent**report card**                                                                                                                                                                                                          | How they used the phone lately + did they leave? |

We glue them together with `Customer_ID` (like a student ID). One ID → one person. No doubles. Nice and clean.

---

## What does “churn” mean here?

**Churn = 1** means: they left about 1–2 months after we looked at their data.
**Churn = 0** means: they stayed.

In this homework dataset, about **half** leave and **half** stay.
(Real phone companies usually have fewer people leaving. This set may be balanced on purpose for learning.)

---

## Why do EDA first? (looking before modeling)

EDA means **Exploratory Data Analysis** — just looking carefully.

Think of cooking:

1. Check if the ingredients are real and not spoiled (**data quality**)
2. See what’s missing (**holes in the data**)
3. See what “leaving” looks like (**the target**)
4. See what is different about people who leave (**clues**)
5. *Then* try recipes (machine learning)

If you skip looking, your “smart model” might learn nonsense.

---

## The two tables — how they fit

```text
   Client (who you are + lifetime history)
              │
              │  same Customer_ID
              ▼
   Record (recent usage + churn yes/no)
```

- **Client** answers: *Who is this person over a long time?*
- **Record** answers: *How did they behave recently, and did they leave?*

You need **both** to tell a full story.

---

## Missing values — holes in the notebook

Some boxes are empty.

- Recent call/usage numbers → usually **filled in**
- Home / income / cars → often **empty** (about 1 in 4 to 1 in 2 people)

**Simple idea:**
The phone company knows how you *use* the phone.
It doesn’t always know your *house and salary*.

So for stories and models, we trust usage more than “number of cars.”

---

## What “correlated” means (5th-grade version)

**Correlated** means: two columns often move together.

### Easy picture

Imagine two thermometers in the same room:

- When one says “hot,” the other usually says “hot” too.
- They are **highly correlated**.
- You don’t need *both* to know the room is hot.

### In our phone data

“How many minutes someone talks” is written in **many** columns:

| Name style              | What it roughly means           |
| ----------------------- | ------------------------------- |
| `mou_Mean`            | Minutes lately (recent average) |
| `avg3mou`             | Minutes over the last ~3 months |
| `avg6mou`             | Minutes over the last ~6 months |
| `avgmou` / `totmou` | Long-time / lifetime minutes    |

If someone talks a lot lately, they often talked a lot in the 3-month and 6-month views too.
Those numbers are **friends** — they say almost the same thing.

**Revenue** works the same way (`rev_Mean`, `avgrev`, `totrev`, …).

### High correlation ≠ “bad”

It just means **repeat information**.

Keeping *every* minutes column is like putting five identical apples in a fruit basket and saying you have five kinds of fruit.

---

## What is *not* the same as “minutes level”?

Some columns are related but tell a **different** story:

| Column            | Simple meaning                               | Why it’s special                   |
| ----------------- | -------------------------------------------- | ----------------------------------- |
| `change_mou`    | Did minutes go **up or down** lately? | Trajectory — not just “how much” |
| `change_rev`    | Did the bill go up or down?                  | Same idea for money                 |
| `eqpdays`       | How**old** is the phone?               | Not minutes at all                  |
| `drop_vce_Mean` | Calls that failed / dropped                  | Quality problems                    |
| `custcare_Mean` | Calls to customer care                       | Frustration / help seeking          |

So:

- `mou_Mean` and `avg3mou` → often **redundant** (same family)
- `mou_Mean` and `change_mou` → **both useful** (level vs change)

---

## What we saw when we compared leavers vs stayers

People who **left** more often looked like:

1. **Older phones** (`eqpdays` bigger)
2. **Cheaper phones** (`hnd_price` smaller)
3. **Fewer minutes** lately
4. **Minutes dropping** (`change_mou` more negative)

These are clues for a company:

- Maybe offer a **phone upgrade**
- Maybe fix a **bad plan fit** / win them back when usage falls

---

## Segments (groups of people)

We also looked at groups:

- Credit card on file? Spending limit flag? New phone user?
- Which **city/area**?

Some groups churn a bit more than others.
That helps marketing pick *who* to call — not only a score.

---

## Geographic areas

Places like New York, Chicago, Florida, etc.
National churn might be ~50%, but some areas sit a bit higher or lower.
Useful for local campaigns or network checks.

---

## Why charts were done in a certain order

We followed a **trust funnel**:

```text
1. Are the IDs the same people?     (join)
2. What’s missing or broken?       (quality)
3. What is churn?                  (target)
4. What do numbers look like?      (shapes / weird tails)
5. What differs for leavers?       (clues)
6. Which groups / areas?           (segments)
7. Which columns repeat each other?(correlation)
8. Do Client & Record agree?       (sanity check)
```

Later charts only matter if earlier checks pass.

---

## Sanity check — Client vs Record

Lifetime average minutes should **point the same way** as recent minutes — not be identical, but not random either.

If they matched in spirit, the join is believable. Good.

---

## Tiny glossary

| Word                           | Kid-friendly meaning            |
| ------------------------------ | ------------------------------- |
| **Churn**                | Customer leaves                 |
| **MOU**                  | Minutes of use (talk/data time) |
| **Revenue / MRC**        | Money the company charges       |
| **Correlation**          | Two things rise/fall together   |
| **Redundant**            | Extra copy of the same idea     |
| **EDA**                  | Looking at data before modeling |
| **Missing**              | Empty cell                      |
| **Outlier / heavy tail** | A few wild huge/small numbers   |

---

## One-sentence takeaway

The data is about real-looking phone customers; many minute/money columns **repeat the same story**, while phone age and **falling usage** are some of the clearest clues that someone might leave — and we learned that by looking carefully before building any smart model.
