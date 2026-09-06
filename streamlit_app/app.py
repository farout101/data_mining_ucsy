"""
Telecom churn prediction — Streamlit UI

Run from project root:
  streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from model_bundle import engineer_rich_features, lean_feature_frame  # noqa: E402

MODELS_DIR = ROOT / "models"
DEFAULTS_PATH = APP_DIR / "ui_defaults.json"

MODEL_CHOICES = {
    "Final — XGBoost (after improvement)": MODELS_DIR / "final_model.joblib",
    "Before — Random Forest (lean)": MODELS_DIR / "before_random_forest_lean.joblib",
    "Before — Logistic Regression (lean)": MODELS_DIR / "before_logistic_regression_lean.joblib",
}


@st.cache_resource
def load_model(path_str: str):
    return joblib.load(path_str)


@st.cache_data
def load_defaults():
    return json.loads(DEFAULTS_PATH.read_text())


def is_final_model(label: str) -> bool:
    return label.startswith("Final")


def predict_frame(model, label: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with churn_proba and churn_pred."""
    engineered = engineer_rich_features(raw_df)
    if is_final_model(label):
        # Ensure required columns exist; fill from defaults if missing
        defaults = load_defaults()
        for col, val in defaults["numeric_defaults"].items():
            if col not in engineered.columns:
                engineered[col] = val
        for col, val in defaults["categorical_defaults"].items():
            if col not in engineered.columns:
                engineered[col] = val
        # recompute derived after fills
        engineered = engineer_rich_features(engineered)
        # eqpdays_bin always from eqpdays
        needed = list(model.numeric) + list(model.categorical)
        missing = [c for c in needed if c not in engineered.columns]
        if missing:
            raise ValueError(f"Missing columns for final model: {missing}")
        X = engineered[needed]
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= getattr(model, "threshold", 0.5)).astype(int)
    else:
        X = lean_feature_frame(engineered)
        # clip change cols lightly if present (pipeline also imputes)
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    out = raw_df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    out["risk_band"] = pd.cut(
        proba,
        bins=[-0.01, 0.4, 0.55, 1.01],
        labels=["Lower", "Medium", "Higher"],
    )
    return out


def build_single_row(defaults: dict) -> dict:
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)

    nd = defaults["numeric_defaults"]
    cd = defaults["categorical_defaults"]
    opts = defaults["categorical_options"]

    with c1:
        months = st.number_input("Months in service", 6, 61, int(nd["months"]))
        eqpdays = st.number_input("Equipment age (days)", 0, 2000, int(nd["eqpdays"]))
        hnd_price = st.number_input("Handset price", 0.0, 600.0, float(nd["hnd_price"]))
        phones = st.number_input("Phones issued", 1, 20, int(nd["phones"]))
        models = st.number_input("Models issued", 1, 20, int(nd["models"]))

    with c2:
        mou_Mean = st.number_input("Mean monthly minutes (MOU)", 0.0, 5000.0, float(nd["mou_Mean"]))
        rev_Mean = st.number_input("Mean monthly revenue", -10.0, 500.0, float(nd["rev_Mean"]))
        totmrc_Mean = st.number_input("Monthly recurring charge (MRC)", -30.0, 200.0, float(nd["totmrc_Mean"]))
        change_mou = st.number_input("Change in MOU", -2000.0, 2000.0, float(nd["change_mou"]))
        change_rev = st.number_input("Change in revenue", -500.0, 500.0, float(nd["change_rev"]))

    with c3:
        avg3mou = st.number_input("3-month avg MOU", 0.0, 5000.0, float(nd["avg3mou"]))
        avg6mou = st.number_input("6-month avg MOU", 0.0, 5000.0, float(nd.get("avg6mou", nd["avg3mou"])))
        avgrev = st.number_input("Lifetime avg revenue", 0.0, 500.0, float(nd["avgrev"]))
        drop_vce_Mean = st.number_input("Dropped voice calls (mean)", 0.0, 100.0, float(nd["drop_vce_Mean"]))
        custcare_Mean = st.number_input("Customer care calls (mean)", 0.0, 50.0, float(nd["custcare_Mean"]))

    st.subheader("Usage extras")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        ovrmou_Mean = st.number_input("Overage minutes", 0.0, 1000.0, float(nd["ovrmou_Mean"]))
    with e2:
        ovrrev_Mean = st.number_input("Overage revenue", 0.0, 200.0, float(nd["ovrrev_Mean"]))
    with e3:
        attempt_Mean = st.number_input("Attempted calls", 0.0, 500.0, float(nd["attempt_Mean"]))
        # use as plcd_vce proxy for drop_rate
        plcd_vce_Mean = attempt_Mean
    with e4:
        complete_Mean = st.number_input("Completed calls", 0.0, 500.0, float(nd["complete_Mean"]))
        roam_Mean = st.number_input("Roaming calls", 0.0, 100.0, float(nd["roam_Mean"]))

    st.subheader("Segments")
    s1, s2, s3 = st.columns(3)
    with s1:
        asl_flag = st.selectbox("Account spending limit", opts["asl_flag"], index=opts["asl_flag"].index(cd["asl_flag"]) if cd["asl_flag"] in opts["asl_flag"] else 0)
        creditcd = st.selectbox("Credit card", opts["creditcd"], index=opts["creditcd"].index(str(cd["creditcd"])) if str(cd["creditcd"]) in opts["creditcd"] else 0)
        new_cell = st.selectbox("New cell user", opts["new_cell"], index=opts["new_cell"].index(cd["new_cell"]) if cd["new_cell"] in opts["new_cell"] else 0)
        refurb_new = st.selectbox("Handset new/refurb", opts["refurb_new"], index=opts["refurb_new"].index(cd["refurb_new"]) if cd["refurb_new"] in opts["refurb_new"] else 0)
    with s2:
        dualband = st.selectbox("Dualband", opts["dualband"], index=opts["dualband"].index(cd["dualband"]) if cd["dualband"] in opts["dualband"] else 0)
        hnd_webcap = st.selectbox("Handset web cap", opts["hnd_webcap"], index=opts["hnd_webcap"].index(cd["hnd_webcap"]) if cd["hnd_webcap"] in opts["hnd_webcap"] else 0)
        area = st.selectbox("Area", opts["area"], index=opts["area"].index(cd["area"]) if cd["area"] in opts["area"] else 0)
        prizm = st.selectbox("PRIZM social", opts["prizm_social_one"], index=opts["prizm_social_one"].index(cd["prizm_social_one"]) if cd["prizm_social_one"] in opts["prizm_social_one"] else 0)
    with s3:
        marital = st.selectbox("Marital", opts["marital"], index=opts["marital"].index(str(cd["marital"])) if str(cd["marital"]) in opts["marital"] else 0)
        ethnic = st.selectbox("Ethnic", opts["ethnic"], index=opts["ethnic"].index(str(cd["ethnic"])) if str(cd["ethnic"]) in opts["ethnic"] else 0)

    row = {
        "eqpdays": eqpdays,
        "hnd_price": hnd_price,
        "phones": phones,
        "models": models,
        "months": months,
        "mou_Mean": mou_Mean,
        "rev_Mean": rev_Mean,
        "totmrc_Mean": totmrc_Mean,
        "change_mou": change_mou,
        "change_rev": change_rev,
        "avg3mou": avg3mou,
        "avg6mou": avg6mou,
        "avgrev": avgrev,
        "drop_vce_Mean": drop_vce_Mean,
        "custcare_Mean": custcare_Mean,
        "ovrmou_Mean": ovrmou_Mean,
        "ovrrev_Mean": ovrrev_Mean,
        "complete_Mean": complete_Mean,
        "attempt_Mean": attempt_Mean,
        "plcd_vce_Mean": plcd_vce_Mean,
        "roam_Mean": roam_Mean,
        "asl_flag": asl_flag,
        "creditcd": creditcd,
        "new_cell": new_cell,
        "refurb_new": refurb_new,
        "dualband": dualband,
        "hnd_webcap": hnd_webcap,
        "area": area,
        "prizm_social_one": prizm,
        "marital": marital,
        "ethnic": ethnic,
    }
    return row


def render_prediction(result: pd.DataFrame):
    row = result.iloc[0]
    proba = float(row["churn_proba"])
    pred = int(row["churn_pred"])
    band = str(row["risk_band"])

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{proba:.1%}")
    m2.metric("Predicted class", "Churn" if pred == 1 else "Stay")
    m3.metric("Risk band", band)

    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= 0.55:
        st.warning(
            "Higher risk — consider retention actions (handset upgrade, plan fit, win-back if usage is falling)."
        )
    elif proba >= 0.4:
        st.info("Medium risk — monitor usage change and equipment age.")
    else:
        st.success("Lower risk relative to the model’s threshold — still watch for sudden usage drops.")


def main():
    st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")
    st.title("Telecom churn predictor")
    st.caption(
        "Uses saved models from this project. "
        "**Final** = tuned XGBoost (after improvement). "
        "Before models = lean Logistic Regression / Random Forest."
    )

    if not (MODELS_DIR / "final_model.joblib").exists():
        st.error("Model files not found in `models/`. Train/export models first.")
        st.stop()

    label = st.sidebar.selectbox("Model", list(MODEL_CHOICES.keys()))
    model_path = MODEL_CHOICES[label]
    model = load_model(str(model_path))
    defaults = load_defaults()

    st.sidebar.markdown("### Holdout reference scores")
    if is_final_model(label):
        st.sidebar.write("Accuracy ≈ **63.3%** · AUC ≈ **0.69** · Top-10% lift ≈ **1.58×**")
    elif "Forest" in label:
        st.sidebar.write("Accuracy ≈ **61.3%** · AUC ≈ **0.67**")
    else:
        st.sidebar.write("Accuracy ≈ **58.0%** · AUC ≈ **0.61**")

    tab1, tab2 = st.tabs(["Single customer", "Batch CSV"])

    with tab1:
        row = build_single_row(defaults)
        if st.button("Predict churn", type="primary"):
            raw = pd.DataFrame([row])
            try:
                result = predict_frame(model, label, raw)
                render_prediction(result)
                with st.expander("Feature row sent to model"):
                    st.dataframe(result)
            except Exception as e:
                st.exception(e)

    with tab2:
        st.write(
            "Upload a CSV with customer feature columns (Client/Record style or already engineered). "
            "Missing derived fields are computed when possible; other gaps use training defaults for the final model."
        )
        sample_cols = [
            "eqpdays", "hnd_price", "months", "mou_Mean", "rev_Mean", "totmrc_Mean",
            "change_mou", "change_rev", "avg3mou", "drop_vce_Mean", "custcare_Mean",
            "asl_flag", "creditcd", "area",
        ]
        st.code(", ".join(sample_cols) + ", ...")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df):,}** rows × {df.shape[1]} columns")
            if st.button("Run batch prediction", type="primary"):
                try:
                    result = predict_frame(model, label, df)
                    st.dataframe(result.head(50))
                    st.download_button(
                        "Download predictions CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )
                    st.bar_chart(result["risk_band"].value_counts())
                except Exception as e:
                    st.exception(e)

    st.markdown("---")
    st.caption(
        "Project models: `models/final_model.joblib` · docs in `docs/model_evolution.md` · "
        "This UI is a demo; scores are probabilistic, not guarantees."
    )


if __name__ == "__main__":
    main()
