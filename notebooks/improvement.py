"""
Stage 4 — LightGBM + XGBoost Soft-Vote Ensemble
=================================================
What we are doing:
  1. Load & join data exactly as in modeling.ipynb.
  2. Apply the same rich feature engineering (engineer_rich_features from src/).
  3. Train LightGBM (already installed, fast, handles categoricals natively).
  4. Average its probability scores with the existing Stage-3 XGBoost probabilities
     → soft-vote ensemble (Stage 4).
  5. Evaluate and compare against Stage 3.
  6. If the ensemble beats Stage 3, save it as improved_ensemble.joblib.

Why this works:
  - LightGBM uses a leaf-wise tree growth strategy (vs XGBoost's level-wise),
    so it discovers different split patterns on the same data.
  - Averaging two models that disagree in different parts of the feature space
    reduces prediction variance → higher ROC-AUC and lift.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model_bundle import engineer_rich_features, XGBBundle, EnsembleBundle  # type: ignore # noqa: E402

DATA_DIR = ROOT / "data" / "telecom"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

# ─── 1. Load & Join ───────────────────────────────────────────────────────────
print("Loading data ...")
client = pd.read_csv(DATA_DIR / "Client.csv")
record = pd.read_csv(DATA_DIR / "Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")
print(f"  Joined shape: {df.shape}")

# ─── 2. Rich Feature Engineering ─────────────────────────────────────────────
print("Engineering rich features ...")
work = engineer_rich_features(df)

RICH_NUMERIC = [
    "eqpdays", "hnd_price", "phones", "models", "months",
    "mou_Mean", "rev_Mean", "totmrc_Mean",
    "change_mou", "change_rev",
    "avg3mou", "avg6mou", "avgrev",
    "drop_vce_Mean", "custcare_Mean",
    "ovrmou_Mean", "ovrrev_Mean",
    "complete_Mean", "attempt_Mean", "roam_Mean",
    "mou_decline_flag", "overage_ratio",
    "eqpdays_x_change_mou", "mou_per_month", "rev_per_mou",
    "drop_rate", "care_per_mou",
]
RICH_CATEGORICAL = [
    "asl_flag", "creditcd", "new_cell", "refurb_new",
    "dualband", "hnd_webcap", "area", "prizm_social_one",
    "eqpdays_bin", "marital", "ethnic",
]

# Keep only columns that actually exist
RICH_NUMERIC = [c for c in RICH_NUMERIC if c in work.columns]
RICH_CATEGORICAL = [c for c in RICH_CATEGORICAL if c in work.columns]
FEATURE_COLS = RICH_NUMERIC + RICH_CATEGORICAL

print(f"  Features: {len(FEATURE_COLS)} ({len(RICH_NUMERIC)} numeric, {len(RICH_CATEGORICAL)} categorical)")

X = work[FEATURE_COLS].copy()
y = work["churn"].astype(int)

for col in RICH_CATEGORICAL:
    X[col] = X[col].astype("string").fillna("Missing").astype(str)

# ─── 3. Train/Test Split (same as Stage 3) ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ─── 4. Winsorization (same clip bounds as Stage 3) ──────────────────────────
CLIP_COLS = ["change_mou", "change_rev", "overage_ratio",
             "eqpdays_x_change_mou", "rev_per_mou", "drop_rate", "care_per_mou"]
CLIP_COLS = [c for c in CLIP_COLS if c in X_train.columns]

clip_bounds = {}
for col in CLIP_COLS:
    lo, hi = X_train[col].quantile([0.01, 0.99])
    clip_bounds[col] = (lo, hi)
    X_train[col] = X_train[col].clip(lo, hi)
    X_test[col] = X_test[col].clip(lo, hi)

# ─── 5. Preprocessor ─────────────────────────────────────────────────────────
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, RICH_NUMERIC),
    ("cat", cat_pipe, RICH_CATEGORICAL),
], remainder="drop")

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)
print(f"  Transformed shape: {X_train_t.shape}")

# ─── 6. Load Stage-3 XGBoost model & get its test probabilities ───────────────
print("\nLoading Stage-3 XGBoost model ...")
xgb_bundle = joblib.load(MODEL_DIR / "final_model.joblib")
xgb_proba = xgb_bundle.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"  Stage 3 XGB — Accuracy: {xgb_acc:.4f}  AUC: {xgb_auc:.4f}")

# ─── 7. Train LightGBM ────────────────────────────────────────────────────────
print("\nTraining LightGBM (Stage 4a) ...")
lgb_clf = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=63,          # 2^max_depth - 1 → more expressive than XGB default
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=0.1,          # L1 regularization (extra vs XGB config)
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
lgb_clf.fit(X_train_t, y_train)
lgb_proba = lgb_clf.predict_proba(X_test_t)[:, 1] # type: ignore

lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
lgb_auc = roc_auc_score(y_test, lgb_proba)
print(f"  LightGBM — Accuracy: {lgb_acc:.4f}  AUC: {lgb_auc:.4f}")

# ─── 8. Soft-Vote Ensemble (Stage 4) ─────────────────────────────────────────
print("\nBuilding Soft-Vote Ensemble (XGBoost 50% + LightGBM 50%) ...")
ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
ens_preds = (ensemble_proba >= 0.5).astype(int)

ens_acc = accuracy_score(y_test, ens_preds)
ens_auc = roc_auc_score(y_test, ensemble_proba)
ens_prec = precision_score(y_test, ens_preds)
ens_rec = recall_score(y_test, ens_preds)
ens_f1 = f1_score(y_test, ens_preds)

# ─── 9. Top-10% Lift ─────────────────────────────────────────────────────────
def top10_lift(y_true, proba):
    df_r = pd.DataFrame({"y": y_true.values, "p": proba})
    cutoff = df_r["p"].quantile(0.90)
    top = df_r[df_r["p"] >= cutoff]
    baseline = df_r["y"].mean()
    top_rate = top["y"].mean()
    return top_rate, top_rate / baseline if baseline > 0 else float("nan")

xgb_top_rate, xgb_lift = top10_lift(y_test, xgb_proba)
lgb_top_rate, lgb_lift = top10_lift(y_test, lgb_proba)
ens_top_rate, ens_lift = top10_lift(y_test, ensemble_proba)

# ─── 10. Results Table ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STAGE COMPARISON")
print("=" * 65)
header = f"{'Model':<30} {'Acc':>7} {'AUC':>7} {'Recall':>8} {'F1':>7} {'Lift':>7}"
print(header)
print("-" * 65)
rows = [
    ("Stage 3 — XGBoost (baseline)", xgb_acc, xgb_auc,
     recall_score(y_test, (xgb_proba >= 0.5).astype(int)), f1_score(y_test, (xgb_proba >= 0.5).astype(int)), xgb_lift),
    ("Stage 4a — LightGBM", lgb_acc, lgb_auc,
     recall_score(y_test, (lgb_proba >= 0.5).astype(int)), f1_score(y_test, (lgb_proba >= 0.5).astype(int)), lgb_lift),
    ("Stage 4b — XGB+LGB Ensemble", ens_acc, ens_auc, ens_rec, ens_f1, ens_lift),
]
for name, acc, auc, rec, f1, lift in rows:
    print(f"  {name:<28} {acc:>7.4f} {auc:>7.4f} {rec:>8.4f} {f1:>7.4f} {lift:>7.2f}x")
print("=" * 65)

print("\nClassification Report — Stage 4 Ensemble:")
print(classification_report(y_test, ens_preds, target_names=["Stayed", "Churned"]))

# ─── 11. Save improved model if it beats Stage 3 AUC ─────────────────────────
if ens_auc > xgb_auc:
    save_path = MODEL_DIR / "stage4_ensemble.joblib"
    ensemble_bundle = EnsembleBundle(
        xgb_bundle=xgb_bundle,
        lgb_preprocessor=preprocessor,
        lgb_model=lgb_clf,
        lgb_clip_bounds=clip_bounds,
        numeric=RICH_NUMERIC,
        categorical=RICH_CATEGORICAL,
        weights=[0.5, 0.5],
        threshold=0.5,
    )
    joblib.dump(ensemble_bundle, save_path)
    print(f"\n  Model SAVED → {save_path}")
    print(f"  AUC improvement over Stage 3: +{ens_auc - xgb_auc:.4f}")
else:
    print("\n  Ensemble did not beat Stage 3 AUC — no file saved.")
    print(f"  AUC delta: {ens_auc - xgb_auc:+.4f}")

print("\nDone.")
