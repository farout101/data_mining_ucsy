"""Serializable model wrappers for churn prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd


class XGBBundle:
    """Preprocessor + XGBoost model. Load with joblib; call predict_proba(X_df)."""

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        clip_bounds: dict,
        numeric: list[str],
        categorical: list[str],
        threshold: float = 0.5,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.clip_bounds = clip_bounds
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.threshold = threshold

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, bounds in self.clip_bounds.items():
            lo, hi = bounds
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        for col in self.categorical:
            if col in X.columns:
                X[col] = X[col].astype("string").fillna("Missing").astype(str)
        return X[self.numeric + self.categorical]

    def predict_proba(self, X: pd.DataFrame):
        Xp = self._prepare(X)
        return self.model.predict_proba(self.preprocessor.transform(Xp))

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def engineer_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rich feature columns from raw Client+Record style fields when possible.

    Accepts either already-engineered columns or a subset of raw columns.
    Derived fields are recomputed when their inputs exist.
    """
    out = df.copy()
    eps = 1e-6

    if "eqpdays" in out.columns:
        out["eqpdays"] = out["eqpdays"].clip(lower=0)

    if "change_mou" in out.columns and "mou_decline_flag" not in out.columns:
        out["mou_decline_flag"] = (out["change_mou"] < 0).astype(float)

    if (
        "ovrrev_Mean" in out.columns
        and "totmrc_Mean" in out.columns
        and "overage_ratio" not in out.columns
    ):
        out["overage_ratio"] = out["ovrrev_Mean"] / (out["totmrc_Mean"].abs() + eps)

    if "eqpdays" in out.columns and "change_mou" in out.columns:
        out["eqpdays_x_change_mou"] = out["eqpdays"] * out["change_mou"].fillna(0)

    if "mou_Mean" in out.columns and "months" in out.columns:
        out["mou_per_month"] = out["mou_Mean"] / (out["months"] + eps)

    if "rev_Mean" in out.columns and "mou_Mean" in out.columns:
        out["rev_per_mou"] = out["rev_Mean"] / (out["mou_Mean"].abs() + eps)

    if "drop_vce_Mean" in out.columns and "plcd_vce_Mean" in out.columns:
        out["drop_rate"] = out["drop_vce_Mean"] / (out["plcd_vce_Mean"].abs() + eps)
    elif "drop_rate" not in out.columns and "drop_vce_Mean" in out.columns:
        # fallback if attempts missing
        out["drop_rate"] = 0.0

    if "custcare_Mean" in out.columns and "mou_Mean" in out.columns:
        out["care_per_mou"] = out["custcare_Mean"] / (out["mou_Mean"].abs() + eps)

    if "eqpdays" in out.columns:
        out["eqpdays_bin"] = pd.cut(
            out["eqpdays"],
            bins=[-0.1, 180, 365, 730, 1e9],
            labels=["0_6m", "6_12m", "1_2y", "2y_plus"],
        ).astype(str)

    return out


def lean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lean columns exist (for before-improvement pipelines)."""
    out = engineer_rich_features(df)
    lean_num = [
        "eqpdays",
        "hnd_price",
        "phones",
        "models",
        "months",
        "mou_Mean",
        "rev_Mean",
        "totmrc_Mean",
        "change_mou",
        "change_rev",
        "avg3mou",
        "drop_vce_Mean",
        "custcare_Mean",
        "ovrmou_Mean",
        "ovrrev_Mean",
        "mou_decline_flag",
        "overage_ratio",
    ]
    lean_cat = [
        "asl_flag",
        "creditcd",
        "new_cell",
        "refurb_new",
        "dualband",
        "hnd_webcap",
        "area",
        "prizm_social_one",
    ]
    for col in lean_cat:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Missing").astype(str)
    return out[lean_num + lean_cat]
