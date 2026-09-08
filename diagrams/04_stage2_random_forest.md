# Diagram: Stage 2 — Random Forest (First Upgrade)

```mermaid
flowchart TD
    A["Same Lean Feature Matrix\n~25 features — identical to Stage 1\nTrain: 80,000 rows"]

    subgraph PIPE2 ["RF Pipeline"]
        direction TB
        P1["SimpleImputer\nstrategy=median (numerics)\nfill='Missing' (categoricals)"]
        P2["OneHotEncoder\nhandle_unknown=ignore\n(NO scaling needed for trees)"]
        P3["RandomForestClassifier\nn_estimators=200\nmax_depth=12\nmin_samples_leaf=20\nrandom_state=42\nn_jobs=-1"]
        P1 --> P2 --> P3
    end

    B["Test Set\n20,000 rows"]
    C["Predictions\nproba + hard label @0.5"]

    subgraph COMPARE ["Stage 1 vs Stage 2 Comparison (same features)"]
        direction LR
        T1["Metric"]
        T2["Logistic Reg"]
        T3["Random Forest"]
        T4["Change"]
        T5["Accuracy"]
        T6["0.5802"]
        T7["0.6133"]
        T8["+3.3 pp ✅"]
        T9["ROC-AUC"]
        T10["0.6090"]
        T11["0.6658"]
        T12["+0.057 ✅"]
        T13["F1"]
        T14["0.5774"]
        T15["0.6301"]
        T16["+0.053 ✅"]
        T17["Top-10% Lift"]
        T18["1.27×"]
        T19["1.47×"]
        T20["better ✅"]
    end

    subgraph IMPORTANCE ["Top Feature Importances (MDI)"]
        I1["1. eqpdays       0.174"]
        I2["2. months        0.144"]
        I3["3. change_mou    0.070"]
        I4["4. mou_Mean      0.066"]
        I5["5. avg3mou       0.051"]
        I6["6. totmrc_Mean   0.051"]
        I7["7. hnd_price     0.044"]
        I8["8. change_rev    0.042"]
        I9["9. rev_Mean      0.041"]
        I10["10. overage_ratio 0.036"]
    end

    VERDICT["✅ RF becomes first serious primary model\nProves non-linearity matters\n→ Still ask: can we do better with richer features + boosting?"]

    A --> PIPE2
    B --> PIPE2
    PIPE2 --> C
    C --> COMPARE
    C --> IMPORTANCE
    COMPARE --> VERDICT

    style VERDICT fill:#2d6a4f,color:#fff
    style COMPARE fill:#457b9d,color:#fff
    style IMPORTANCE fill:#f1faee
```

**Key insight from Stage 2:**
- RF beat LR on **every metric** using the **exact same features** — this proves the gain comes from the algorithm (non-linear trees), not from adding new data.
- Feature importances aligned perfectly with EDA findings: `eqpdays`, `months`, `change_mou` are the top drivers.
- ~61% accuracy still felt low for a headline score → motivated Stage 3 with richer features + gradient boosting.
