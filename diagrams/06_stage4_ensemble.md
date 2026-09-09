# Diagram: Stage 4 — Soft-Vote Ensemble (XGBoost + LightGBM)

This diagram shows the architecture and workflow of the Stage 4 Soft-Vote Ensemble, combining level-wise XGBoost and leaf-wise LightGBM into the project's highest-performing churn prediction model.

```mermaid
flowchart TD
    A["Rich Feature Matrix\n38 features (27 numeric, 11 categorical)\nTrain: 80,000 | Test: 20,000"]

    subgraph PREP ["Preprocessing Pipeline"]
        P1["Clip 1st-99th percentile (train only)\nMedian Imputer (numerics)\nOneHotEncoder (categoricals)\n→ 102 transformed features"]
    end

    subgraph MODELS ["Two Complementary Boosting Architectures"]
        direction TB
        subgraph M1 ["Model 1: Tuned XGBoost (Stage 3)"]
            X1["Level-wise Tree Growth\n(splits by horizontal depth)"]
            X2["max_depth=7 · lr=0.03\n600 trees · subsample=0.8\ncolsample=0.7 · L2=2.0"]
            X3["P_xgb(churn)"]
            X1 --> X2 --> X3
        end

        subgraph M2 ["Model 2: Tuned LightGBM (Stage 4a)"]
            L1["Leaf-wise Tree Growth\n(best-first loss reduction)"]
            L2["num_leaves=63 · lr=0.03\n800 trees · subsample=0.8\ncolsample=0.7 · L1=0.1 · L2=2.0"]
            L3["P_lgb(churn)"]
            L1 --> L2 --> L3
        end
    end

    subgraph BLEND ["Soft-Vote Ensemble Blending"]
        direction TB
        E1["Equal Weighting (50% / 50%)\nP_ens = 0.5 × P_xgb + 0.5 × P_lgb"]
        E2["Error Variance Reduction\n(XGB and LGB errors decorrelate on uncertain boundaries)"]
        E1 --> E2
    end

    subgraph RESULTS4 ["Stage 4 Champion Results"]
        R1["Accuracy:  0.6349  (~63.5%) 🏆"]
        R2["ROC-AUC:   0.6899  (~0.690) 🏆"]
        R3["Recall:    0.6422  (Caught 6,366 churners) 🏆"]
        R4["F1-Score:  0.6355          🏆"]
        R5["Top-10% Lift: 1.59×        🏆"]
    end

    SAVE[("💾 Saved as\nmodels/stage4_ensemble.joblib\n(EnsembleBundle)")]
    APP["🖥️ Streamlit App\nSelectable model option in UI"]

    A --> PREP
    PREP --> M1
    PREP --> M2
    X3 --> BLEND
    L3 --> BLEND
    BLEND --> RESULTS4
    RESULTS4 --> SAVE
    SAVE --> APP
```

### Full Evolution Summary Across All 4 Stages

| Metric | Stage 1 (LogReg) | Stage 2 (Random Forest) | Stage 3 (XGBoost) | Stage 4 (Ensemble) | Total Gain (1 → 4) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Features Used** | Lean (~25) | Lean (~25) | Rich (~38) | **Rich (~38)** | — |
| **Accuracy** | 0.5802 | 0.6133 | 0.6327 | **0.6349** | **+5.5 pp** |
| **ROC-AUC** | 0.6090 | 0.6658 | 0.6889 | **0.6899** | **+0.081** |
| **Recall (Churners)** | 0.5786 | 0.6645 | 0.6392 | **0.6422** | **+0.064** |
| **F1-Score** | 0.5774 | 0.6301 | 0.6330 | **0.6355** | **+0.058** |
| **Top-10% Lift** | 1.27× | 1.47× | 1.58× | **1.59×** | **+0.32×** |

### Why This Stage Was Added
1. **Architectural Diversity**: XGBoost splits level-by-level; LightGBM splits leaf-by-leaf. They learn slightly different decision boundaries on the same features.
2. **Variance Reduction**: Blending two high-performing probability distributions reduces noise and smooths probability calibration.
3. **Seamless Deployment**: Wrapped in `EnsembleBundle` inside [`models/stage4_ensemble.joblib`](../models/stage4_ensemble.joblib), making it instantly runnable in the Streamlit UI.
