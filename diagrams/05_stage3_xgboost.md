# Diagram: Stage 3 — XGBoost Tuned (Final Model)

```mermaid
flowchart TD
    A["Rich Feature Matrix\n~35+ features\nTrain: 80,000 rows"]

    subgraph FE_RICH ["Additional Rich Features (vs Lean)"]
        direction LR
        N1["eqpdays_x_change_mou\n(interaction term)"]
        N2["mou_per_month\ndrop_rate\ncare_per_mou"]
        N3["rev_per_mou\neqpdays_bin\navg6mou · avgrev"]
        N4["complete_Mean\nattempt_Mean\nroam_Mean"]
        N5["marital\nethnic"]
    end

    subgraph SEARCH ["Hyperparameter Search\nRandomizedSearchCV\nn_iter=20, cv=3, scoring=roc_auc"]
        direction TB
        S1["max_depth: [4,5,6,7,8]"]
        S2["learning_rate: [0.01,0.03,0.05,0.1]"]
        S3["n_estimators: [300,400,500,600]"]
        S4["subsample: [0.7,0.8,0.9]"]
        S5["colsample_bytree: [0.6,0.7,0.8]"]
        S6["CV Best AUC ≈ 0.686"]
        S1 & S2 & S3 & S4 & S5 --> S6
    end

    subgraph BEST ["Best XGBoost Config"]
        B1["max_depth=7"]
        B2["learning_rate=0.03"]
        B3["n_estimators=600"]
        B4["subsample=0.8"]
        B5["colsample_bytree=0.7"]
        B6["min_child_weight=1"]
        B7["reg_lambda=2.0"]
    end

    subgraph RESULTS3 ["Stage 3 Final Results"]
        R1["Accuracy:  0.6327  (~63.3%) 🏆"]
        R2["ROC-AUC:   0.6889          🏆"]
        R3["F1:        0.6330          🏆"]
        R4["Top-10% churn: 78.2%       🏆"]
        R5["Lift vs random: 1.58×      🏆"]
    end

    subgraph ALLCOMP ["Full Model Comparison (same 80/20 split)"]
        direction TB
        C1["LogReg  Lean  → Acc 0.580 · AUC 0.609 · Lift 1.27×"]
        C2["RF      Lean  → Acc 0.613 · AUC 0.666 · Lift 1.47×"]
        C3["RF      Rich  → Acc 0.618 · AUC 0.675 · Lift 1.53×"]
        C4["HistGB  Rich  → Acc 0.630 · AUC 0.684 · Lift 1.58×"]
        C5["XGB     Rich  → Acc 0.631 · AUC 0.689 · Lift 1.58×"]
        C6["XGB Tuned Rich→ Acc 0.633 · AUC 0.689 · Lift 1.58× ✅ FINAL"]
        C1 --- C2 --- C3 --- C4 --- C5 --- C6
    end

    SAVE[("💾 Saved as\nmodels/final_model.joblib\nmodels/after_xgboost_tuned_rich.joblib")]

    A --> FE_RICH
    FE_RICH --> SEARCH
    SEARCH --> BEST
    BEST --> RESULTS3
    RESULTS3 --> ALLCOMP
    RESULTS3 --> SAVE

    style RESULTS3 fill:#2d6a4f,color:#fff
    style C6 fill:#2d6a4f,color:#fff
    style SAVE fill:#e76f51,color:#fff
    style SEARCH fill:#457b9d,color:#fff
```

**Total gain from Stage 1 → Stage 3:**

| Metric | Stage 1 (LogReg) | Stage 3 (XGB Tuned) | Gain |
|--------|-----------------|---------------------|------|
| Accuracy | 0.580 | **0.633** | **+5.3 pp** |
| ROC-AUC | 0.609 | **0.689** | **+0.080** |
| Top-10% Lift | 1.27× | **1.58×** | **+0.31×** |
