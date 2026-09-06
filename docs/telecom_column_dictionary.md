# Telecom Dataset — Column Dictionary & Relationships

Source: Company A–style historical telecom data (`telecom/Client.csv`, `telecom/Record.csv`).  
Download: [Kaggle — Telecom Customer Churn 100K cleaned records](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records)

---

## 1. Dataset overview

| File           | Rows × cols  | Grain                | Role                                                             |
| -------------- | ------------- | -------------------- | ---------------------------------------------------------------- |
| `Client.csv` | 100,000 × 50 | One row per customer | Account, lifetime usage/billing, demographics, handset           |
| `Record.csv` | 100,000 × 51 | One row per customer | Mean monthly usage/billing, call quality, tenure,**churn** |

Both tables cover the **same 100,000 customers**. `Customer_ID` is unique in each file and forms a perfect 1:1 match (no orphans on either side).

```text
Record (usage + churn) ──1:1── Client (profile + lifetime)
              │
              └── Customer_ID
```

Merged analysis table: **100,000 × 100** columns (50 + 51 − 1 shared key).

---

## 2. How the two tables relate

### Join key

| Column          | Meaning                    | Relationship                                                     |
| --------------- | -------------------------- | ---------------------------------------------------------------- |
| `Customer_ID` | Unique customer identifier | **Primary / foreign key** linking `Client` ↔ `Record` |

### Conceptual split

- **`Client`** = *who the customer is* and *lifetime account history* (household size, credit, income, handset inventory, life-to-date calls/revenue/minutes, recent 3-/6-month averages).
- **`Record`** = *how they used the service recently* (mean monthly revenue, minutes, overages, dropped/blocked calls) plus the outcome **`churn`** and **`months`** in service.

### Logical feature groups that connect across tables

| Theme              | In`Record`                                                   | In`Client`                                                   | How they relate                                              |
| ------------------ | -------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------ |
| Revenue            | `rev_Mean`, `totmrc_Mean`, `ovrrev_Mean`, `change_rev` | `totrev`, `adjrev`, `avgrev`, `avg3rev`, `avg6rev`   | Short-term monthly revenue vs lifetime / rolling averages    |
| Usage (minutes)    | `mou_Mean`, `change_mou`, peak/off-peak MOU                | `totmou`, `adjmou`, `avgmou`, `avg3mou`, `avg6mou`   | Recent mean MOU vs life-to-date and 3-/6-month averages      |
| Call volume        | `attempt_Mean`, `complete_Mean`, `plcd_*`, `recv_*`    | `totcalls`, `adjqty`, `avgqty`, `avg3qty`, `avg6qty` | Recent call attempts/completions vs lifetime call counts     |
| Quality / friction | `drop_*`, `blck_*`, `unan_*`, `custcare_*`             | —                                                             | Service problems that may drive churn; no direct Client twin |
| Tenure / loyalty   | `months`, `churn`                                          | `eqpdays`, `phones`, `models`                            | Tenure and churn vs equipment age and handset churn          |
| Household / demo   | —                                                             | `uniqsubs`, `actvsubs`, kids, `income`, `area`, …     | Segmentation context for usage and churn patterns            |

### Target for modeling (typical)

- **`churn`** lives only in `Record`: churn between 31–60 days after the observation date (`0` = stayed, `1` = churned).
- To predict churn you **must join** Client features onto Record (or vice versa) on `Customer_ID`.

---

## 3. `Record.csv` — column meanings

Usage/billing fields named `*_Mean` are **averages over the observation window** (typically recent months).

### Revenue & overage

| Column          | Meaning                                                 |
| --------------- | ------------------------------------------------------- |
| `rev_Mean`    | Mean monthly revenue (charge amount)                    |
| `totmrc_Mean` | Mean total monthly recurring charge                     |
| `da_Mean`     | Mean number of directory-assisted calls                 |
| `ovrmou_Mean` | Mean overage minutes of use                             |
| `ovrrev_Mean` | Mean overage revenue                                    |
| `vceovr_Mean` | Mean revenue of voice overage                           |
| `datovr_Mean` | Mean revenue of data overage                            |
| `roam_Mean`   | Mean number of roaming calls                            |
| `change_mou`  | % change in monthly minutes vs previous 3-month average |
| `change_rev`  | % change in monthly revenue vs previous 3-month average |

### Call quality & completion

| Column            | Meaning                           |
| ----------------- | --------------------------------- |
| `drop_vce_Mean` | Mean dropped (failed) voice calls |
| `drop_dat_Mean` | Mean dropped (failed) data calls  |
| `blck_vce_Mean` | Mean blocked (failed) voice calls |
| `blck_dat_Mean` | Mean blocked (failed) data calls  |
| `unan_vce_Mean` | Mean unanswered voice calls       |
| `unan_dat_Mean` | Mean unanswered data calls        |
| `plcd_vce_Mean` | Mean attempted voice calls placed |
| `plcd_dat_Mean` | Mean attempted data calls placed  |
| `recv_vce_Mean` | Mean received voice calls         |
| `recv_sms_Mean` | *(not documented by provider)*  |
| `comp_vce_Mean` | Mean completed voice calls        |
| `comp_dat_Mean` | Mean completed data calls         |
| `drop_blk_Mean` | Mean dropped or blocked calls     |
| `attempt_Mean`  | Mean attempted calls              |
| `complete_Mean` | Mean completed calls              |

### Customer care & special call types

| Column            | Meaning                                   |
| ----------------- | ----------------------------------------- |
| `custcare_Mean` | Mean customer care calls                  |
| `ccrndmou_Mean` | Mean rounded MOU of customer care calls   |
| `cc_mou_Mean`   | Mean unrounded MOU of customer care calls |
| `inonemin_Mean` | Mean inbound calls lasting&lt; 1 minute   |
| `threeway_Mean` | Mean three-way calls                      |
| `callfwdv_Mean` | Mean call-forwarding calls                |
| `callwait_Mean` | Mean call-waiting calls                   |

### Minutes of use (completed / received / peak)

| Column              | Meaning                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `mou_Mean`        | Mean monthly minutes of use                                           |
| `mou_cvce_Mean`   | Mean unrounded MOU of completed voice calls                           |
| `mou_cdat_Mean`   | Mean unrounded MOU of completed data calls                            |
| `mou_rvce_Mean`   | Mean unrounded MOU of received voice calls                            |
| `owylis_vce_Mean` | Mean outbound wireless-to-wireless voice calls                        |
| `mouowylisv_Mean` | Mean unrounded MOU of outbound wireless-to-wireless voice             |
| `iwylis_vce_Mean` | *(not documented; likely inbound wireless-to-wireless voice count)* |
| `mouiwylisv_Mean` | Mean unrounded MOU of inbound wireless-to-wireless voice              |
| `peak_vce_Mean`   | Mean inbound + outbound peak voice calls                              |
| `peak_dat_Mean`   | Mean peak data calls                                                  |
| `mou_peav_Mean`   | Mean unrounded MOU of peak voice calls                                |
| `mou_pead_Mean`   | Mean unrounded MOU of peak data calls                                 |
| `opk_vce_Mean`    | Mean off-peak voice calls                                             |
| `opk_dat_Mean`    | Mean off-peak data calls                                              |
| `mou_opkv_Mean`   | Mean unrounded MOU of off-peak voice calls                            |
| `mou_opkd_Mean`   | Mean unrounded MOU of off-peak data calls                             |

### Outcome & tenure

| Column          | Meaning                                          |
| --------------- | ------------------------------------------------ |
| `churn`       | Churn between 31–60 days after observation date |
| `months`      | Total months in service                          |
| `Customer_ID` | Customer key (join to Client)                    |

---

## 4. `Client.csv` — column meanings

### Household / account flags

| Column       | Meaning                                       |
| ------------ | --------------------------------------------- |
| `uniqsubs` | Number of unique subscribers in the household |
| `actvsubs` | Number of active subscribers in the household |
| `new_cell` | New cell phone user                           |
| `crclscod` | Credit class code                             |
| `asl_flag` | Account spending limit flag                   |

### Lifetime & rolling billing / usage

| Column                                  | Meaning                                                           |
| --------------------------------------- | ----------------------------------------------------------------- |
| `totcalls`                            | Total calls over customer lifetime                                |
| `totmou`                              | Total minutes of use over lifetime                                |
| `totrev`                              | Total revenue                                                     |
| `adjrev`                              | Billing-adjusted total revenue over lifetime                      |
| `adjmou`                              | Billing-adjusted total minutes over lifetime                      |
| `adjqty`                              | Billing-adjusted total number of calls over lifetime              |
| `avgrev`                              | Average monthly revenue over lifetime                             |
| `avgmou`                              | Average monthly minutes over lifetime                             |
| `avgqty`                              | Average monthly number of calls over lifetime                     |
| `avg3mou` / `avg3qty` / `avg3rev` | Avg monthly MOU / calls / revenue over previous**3** months |
| `avg6mou` / `avg6qty` / `avg6rev` | Avg monthly MOU / calls / revenue over previous**6** months |

### Geography, device, lifestyle

| Column                     | Meaning                                    |
| -------------------------- | ------------------------------------------ |
| `prizm_social_one`       | Social group letter                        |
| `area`                   | Geographic area                            |
| `dualband`               | Dual-band handset indicator                |
| `refurb_new`             | Handset refurbished (`R`) or new (`N`) |
| `hnd_price`              | Current handset price                      |
| `phones`                 | Number of handsets issued                  |
| `models`                 | Number of models issued                    |
| `hnd_webcap`             | Handset web capability                     |
| `truck`                  | Truck indicator                            |
| `rv`                     | RV indicator                               |
| `ownrent`                | Home owner / renter status                 |
| `lor`                    | Length of residence                        |
| `dwlltype`               | Dwelling unit type                         |
| `marital`                | Marital status                             |
| `adults`                 | Number of adults in household              |
| `infobase`               | InfoBase match                             |
| `income`                 | Estimated income                           |
| `numbcars`               | Known number of vehicles                   |
| `HHstatin`               | Premier household status indicator         |
| `dwllsize`               | Dwelling size                              |
| `forgntvl`               | Foreign travel dummy                       |
| `ethnic`                 | Ethnicity roll-up code                     |
| `kid0_2` … `kid16_17` | Child present in age band (household)      |
| `creditcd`               | Credit card indicator                      |
| `eqpdays`                | Age (days) of current equipment            |
| `Customer_ID`            | Customer key (join to Record)              |

---

## 5. Ambiguous / undocumented fields

Per the provider overview, some fields have incomplete documentation:

| Column              | Note                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `recv_sms_Mean`   | Listed as NaN / not explained                                     |
| `iwylis_vce_Mean` | Listed as NaN; naming suggests inbound wireless-to-wireless voice |
| `Customer_ID`     | Not explained in the glossary, but clearly the join key           |

Treat these carefully in modeling; prefer documented peers when building business narratives.

---

## 6. Practical notes for analysis

1. **Always join on `Customer_ID`** before relating usage (`Record`) to demographics or lifetime metrics (`Client`).
2. **Near-duplicates:** many revenue/MOU/call columns measure the same idea at different windows (mean monthly vs 3-/6-month vs lifetime). Prefer a small non-redundant set for models and storytelling.
3. **Missingness is concentrated in Client demographics** (`numbcars`, `dwllsize`, `HHstatin`, `ownrent`, `lor`, `income`, …). Record usage fields are mostly complete (&lt;1% missing).
4. **`churn` is nearly balanced** (~50/50) in this dataset — unusual vs real-world telecom, but convenient for accuracy-based evaluation.
5. Strong candidate relationships to explore in EDA: declining usage (`change_mou` / `change_rev`), equipment age (`eqpdays`), customer care volume, overages, and tenure (`months`).
