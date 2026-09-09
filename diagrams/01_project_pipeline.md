# Diagram: Overall Project Pipeline

This diagram shows the end-to-end flow of the entire project, from raw data to deployment.

```mermaid
flowchart TD
    A[("📂 Client.csv\n100k rows")]
    B[("📂 Record.csv\n100k rows")]
    C["Inner Join\non Customer_ID\n→ 100,000 × ~100 cols"]
    D["EDA\nnnotebooks/main.ipynb"]
    E["Feature Engineering\nLean Set ~25 cols"]
    F["Stratified 80/20 Split\nTrain 80k / Test 20k"]
    G1["Stage 1\nLogistic Regression\nLean features"]
    G2["Stage 2\nRandom Forest\nLean features"]
    G3["Stage 3\nXGBoost Tuned\nRich features"]
    G4["Stage 4\nEnsemble (XGB+LGB)\nRich features"]
    H["Model Evaluation\nAccuracy · AUC · Lift"]
    I[("💾 Save Models\nmodels/*.joblib")]
    J["Streamlit UI\nstreamlit_app/app.py"]
    K["Single / Batch\nChurn Prediction"]

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G1
    F --> G2
    F --> G3
    F --> G4
    G1 --> H
    G2 --> H
    G3 --> H
    G4 --> H
    H --> I
    I --> J
    J --> K
```

**Key points:**
- Both CSV files are joined on `Customer_ID` before any analysis.
- EDA informs which features to engineer and which to drop.
- Four model stages are trained on the same 80/20 split for fair comparison.
- All models (Stages 1 through 4) are selectable in the Streamlit UI, with Stage 4 Ensemble as the top-performing champion.
