# Diagram: Stage 1 — Logistic Regression (Baseline)

```mermaid
flowchart TD
    A["Lean Feature Matrix\n~25 features\nTrain: 80,000 rows"]

    subgraph PIPE1 ["LR Pipeline"]
        direction TB
        P1["SimpleImputer\nstrategy=median (numerics)\nfill='Missing' (categoricals)"]
        P2["StandardScaler\n(numerics only)"]
        P3["OneHotEncoder\nhandle_unknown=ignore"]
        P4["LogisticRegression\nmax_iter=1000\nL2 penalty (default)\nrandom_state=42"]
        P1 --> P2 --> P3 --> P4
    end

    B["Test Set\n20,000 rows"]
    C["Predictions\nproba + hard label @0.5"]

    subgraph RESULTS1 ["Stage 1 Results"]
        R1["Accuracy: 58.0%"]
        R2["ROC-AUC: 0.609"]
        R3["Precision: 0.576"]
        R4["Recall: 0.579"]
        R5["F1: 0.577"]
        R6["Top-10% churn: 63.1%\nLift: 1.27×"]
    end

    subgraph INTERP ["Top Coefficients (|coef|)"]
        I1["eqpdays +0.29 → older device = higher risk"]
        I2["months −0.21 → longer tenure = lower risk"]
        I3["mou_Mean −0.20 → more usage = lower risk"]
        I4["change_mou −0.11 → usage up = lower risk"]
    end

    VERDICT["❌ Not sufficient as primary model\nLinear boundary too weak for non-linear churn signal\n→ Proceed to Stage 2"]

    A --> PIPE1
    B --> PIPE1
    PIPE1 --> C
    C --> RESULTS1
    C --> INTERP
    RESULTS1 --> VERDICT
```

**Why this stage was not enough:**
- Accuracy is only ~8 points above a random 50/50 guess.
- EDA showed max linear `|r|` with churn ≈ 0.11 — linear models are fundamentally limited here.
- Churn relationships involve interactions (e.g. old handset **and** usage drop together), which logistic regression cannot easily capture.
- AUC 0.609 means the model ranks churners only slightly better than random.
