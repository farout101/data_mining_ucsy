"""
Telecom Churn Prediction & Risk Intelligence Platform — Streamlit UI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
EDA_DIR = ROOT / "eda_figures"
DIAG_DIR = ROOT / "diagrams"

sys.path.insert(0, str(SRC_DIR))
from model_bundle import engineer_rich_features, lean_feature_frame  # noqa: E402

MODELS_DIR = ROOT / "models"
DEFAULTS_PATH = ROOT / "config" / "ui_defaults.json"

MODEL_CHOICES = {
    "Stage 4 — Ensemble (XGBoost + LightGBM)": MODELS_DIR / "stage4_ensemble.joblib",
    "Stage 3 — Tuned XGBoost (Rich Features)": MODELS_DIR / "final_model.joblib",
    "Stage 2 — Random Forest (Lean Features)": MODELS_DIR / "before_random_forest_lean.joblib",
    "Stage 1 — Logistic Regression (Baseline)": MODELS_DIR / "before_logistic_regression_lean.joblib",
}

MODEL_METRICS = {
    "Stage 4 — Ensemble (XGBoost + LightGBM)": {
        "accuracy": "63.49%",
        "roc_auc": "0.6899",
        "recall": "64.22%",
        "f1": "0.6355",
        "lift": "1.59×",
        "features": "38 Rich Features (Interactions, Ratios, Segments)",
        "badge": "Champion Model (Recommended)",
        "desc": "Soft-vote ensemble blending level-wise XGBoost (depth-7) with leaf-wise LightGBM (num_leaves=63). Achieves highest probability discrimination and lift."
    },
    "Stage 3 — Tuned XGBoost (Rich Features)": {
        "accuracy": "63.27%",
        "roc_auc": "0.6889",
        "recall": "63.92%",
        "f1": "0.6330",
        "lift": "1.58×",
        "features": "38 Rich Features",
        "badge": "Stage 3 Primary Model",
        "desc": "Sequential gradient boosted decision trees tuned via 20-draw randomized cross-validation with L2 regularization."
    },
    "Stage 2 — Random Forest (Lean Features)": {
        "accuracy": "61.33%",
        "roc_auc": "0.6658",
        "recall": "66.45%",
        "f1": "0.6301",
        "lift": "1.47×",
        "features": "25 Lean Features",
        "badge": "Stage 2 Tree Ensemble",
        "desc": "Bagging ensemble of 200 trees (max depth 12). Proved that non-linear interaction logic outperforms linear boundaries."
    },
    "Stage 1 — Logistic Regression (Baseline)": {
        "accuracy": "58.02%",
        "roc_auc": "0.6090",
        "recall": "57.86%",
        "f1": "0.5774",
        "lift": "1.27×",
        "features": "25 Lean Features (Standardized)",
        "badge": "Stage 1 Linear Baseline",
        "desc": "L2-regularized linear model on standardized inputs. Establishes the baseline and proves linear separation is insufficient."
    }
}


@st.cache_resource
def load_model(path_str: str):
    return joblib.load(path_str)


@st.cache_data
def load_defaults():
    return json.loads(DEFAULTS_PATH.read_text())


def is_rich_model(label: str) -> bool:
    return "Ensemble" in label or "XGBoost" in label or "Stage 4" in label or "Stage 3" in label or label.startswith("Final")


def get_prescriptive_recommendation(row: pd.Series | dict) -> tuple[str, str, str]:
    """Return (badge_type, title, description) for business retention."""
    proba = float(row.get("churn_proba", 0.5))
    eqp = float(row.get("eqpdays", 0))
    cmou = float(row.get("change_mou", 0))
    ovr_ratio = float(row.get("overage_ratio", 0)) if "overage_ratio" in row else (
        float(row.get("ovrrev_Mean", 0)) / (abs(float(row.get("totmrc_Mean", 45))) + 1e-6)
    )
    care = float(row.get("custcare_Mean", 0))

    if proba >= 0.55:
        if eqp >= 450:
            return (
                "warning",
                "Hardware Upgrade Renewal Playbook",
                f"Customer handset is **{int(eqp)} days old** (15+ months). Primary churn driver is device obsolescence and contract expiry. **Action:** Offer a subsidized 5G handset upgrade credit tied to a mandatory 24-month contract renewal commitment."
            )
        elif ovr_ratio >= 0.35:
            return (
                "error",
                "Plan Right-Sizing Playbook (Bill Shock Mitigation)",
                f"Overage fees represent **{ovr_ratio:.1%} of base monthly plan**. High risk of bill-shock defection. **Action:** Proactively migrate customer to the next higher tariff tier for an extra $5/month, waiving accumulated penalty overages."
            )
        elif cmou <= -25.0:
            return (
                "warning",
                "Usage Win-Back Outreach Playbook",
                f"Customer monthly usage dropped by **{cmou:.1f}%**. Customer is actively disengaging. **Action:** Dispatch a proactive satisfaction survey, award 10GB bonus loyalty data, and follow up with a customer success outreach call."
            )
        else:
            return (
                "error",
                "VIP Retention Intervention",
                f"High predicted defection probability (**{proba:.1%}**). **Action:** Route account to specialized retention queue with executive authorization to offer a 20% billing credit for 3 billing cycles."
            )
    elif proba >= 0.40:
        if care >= 2:
            return (
                "info",
                "Service Quality & Support Follow-Up",
                f"Customer logged **{int(care)} customer care inquiries**. **Action:** Escalate ticket to technical support supervisor to resolve recurring friction before frustration triggers defection."
            )
        else:
            return (
                "info",
                "Active Monitoring & Engagement Survey",
                "Moderate churn probability. **Action:** Monitor next month usage trajectory and send an automated digital customer satisfaction check-in."
            )
    else:
        return (
            "success",
            "Healthy Account — Loyalty & Cross-Sell Opportunity",
            "Customer exhibits low churn probability and stable engagement. **Action:** Maintain standard service; suitable for secondary device or family plan cross-sell campaigns."
        )


def predict_frame(model, label: str, raw_df: pd.DataFrame, custom_threshold: float = 0.50) -> pd.DataFrame:
    """Return dataframe enriched with churn_proba, churn_pred, risk_band, and recommendation."""
    engineered = engineer_rich_features(raw_df)
    if is_rich_model(label):
        defaults = load_defaults()
        for col, val in defaults["numeric_defaults"].items():
            if col not in engineered.columns:
                engineered[col] = val
        for col, val in defaults["categorical_defaults"].items():
            if col not in engineered.columns:
                engineered[col] = val
        engineered = engineer_rich_features(engineered)
        needed = list(model.numeric) + list(model.categorical)
        missing = [c for c in needed if c not in engineered.columns]
        if missing:
            raise ValueError(f"Missing columns for model: {missing}")
        X = engineered[needed]
        proba = model.predict_proba(X)[:, 1]
    else:
        X = lean_feature_frame(engineered)
        proba = model.predict_proba(X)[:, 1]

    pred = (proba >= custom_threshold).astype(int)

    out = raw_df.copy()
    out["churn_proba"] = np.round(proba, 4)
    out["churn_pred"] = pred
    out["risk_band"] = pd.cut(
        proba,
        bins=[-0.01, 0.40, 0.55, 1.01],
        labels=["Lower Risk", "Medium Risk", "Higher Risk"],
    )

    # Add recommendations
    recs = []
    for _, r in out.iterrows():
        _, title, _ = get_prescriptive_recommendation(r)
        recs.append(title)
    out["recommended_action"] = recs

    return out


def generate_sample_csv() -> bytes:
    """Create sample CSV template for testing batch prediction."""
    sample_df = pd.DataFrame([
        {
            "eqpdays": 580, "hnd_price": 29.99, "phones": 2, "models": 2, "months": 24,
            "mou_Mean": 210.0, "rev_Mean": 55.0, "totmrc_Mean": 35.0, "change_mou": -45.0, "change_rev": 12.0,
            "avg3mou": 380.0, "avg6mou": 410.0, "avgrev": 45.0, "drop_vce_Mean": 6.0, "custcare_Mean": 3.0,
            "ovrmou_Mean": 35.0, "ovrrev_Mean": 18.0, "complete_Mean": 45.0, "attempt_Mean": 70.0, "roam_Mean": 2.0,
            "asl_flag": "Y", "creditcd": "N", "new_cell": "N", "refurb_new": "R", "dualband": "Y", "hnd_webcap": "WC",
            "area": "SOUTH FLORIDA AREA", "prizm_social_one": "S", "marital": "M", "ethnic": "N"
        },
        {
            "eqpdays": 75, "hnd_price": 249.99, "phones": 1, "models": 1, "months": 10,
            "mou_Mean": 750.0, "rev_Mean": 65.0, "totmrc_Mean": 65.0, "change_mou": 15.0, "change_rev": 0.0,
            "avg3mou": 680.0, "avg6mou": 650.0, "avgrev": 65.0, "drop_vce_Mean": 1.0, "custcare_Mean": 0.0,
            "ovrmou_Mean": 0.0, "ovrrev_Mean": 0.0, "complete_Mean": 120.0, "attempt_Mean": 130.0, "roam_Mean": 0.0,
            "asl_flag": "N", "creditcd": "Y", "new_cell": "Y", "refurb_new": "N", "dualband": "Y", "hnd_webcap": "WCMB",
            "area": "NEW YORK CITY AREA", "prizm_social_one": "C", "marital": "S", "ethnic": "C"
        },
        {
            "eqpdays": 342, "hnd_price": 99.99, "phones": 1, "models": 1, "months": 16,
            "mou_Mean": 355.0, "rev_Mean": 48.0, "totmrc_Mean": 45.0, "change_mou": -6.0, "change_rev": -0.5,
            "avg3mou": 358.0, "avg6mou": 363.0, "avgrev": 50.0, "drop_vce_Mean": 3.0, "custcare_Mean": 0.0,
            "ovrmou_Mean": 2.0, "ovrrev_Mean": 1.0, "complete_Mean": 76.0, "attempt_Mean": 101.0, "roam_Mean": 0.0,
            "asl_flag": "N", "creditcd": "Y", "new_cell": "N", "refurb_new": "N", "dualband": "Y", "hnd_webcap": "WCMB",
            "area": "CALIFORNIA NORTH AREA", "prizm_social_one": "S", "marital": "M", "ethnic": "N"
        },
        {
            "eqpdays": 490, "hnd_price": 49.99, "phones": 3, "models": 2, "months": 22,
            "mou_Mean": 180.0, "rev_Mean": 70.0, "totmrc_Mean": 40.0, "change_mou": -30.0, "change_rev": 15.0,
            "avg3mou": 260.0, "avg6mou": 290.0, "avgrev": 52.0, "drop_vce_Mean": 4.0, "custcare_Mean": 2.0,
            "ovrmou_Mean": 40.0, "ovrrev_Mean": 25.0, "complete_Mean": 40.0, "attempt_Mean": 65.0, "roam_Mean": 1.0,
            "asl_flag": "Y", "creditcd": "N", "new_cell": "N", "refurb_new": "R", "dualband": "N", "hnd_webcap": "WC",
            "area": "NORTHWEST/ROCKY MOUNTAIN AREA", "prizm_social_one": "U", "marital": "U", "ethnic": "H"
        },
        {
            "eqpdays": 120, "hnd_price": 199.99, "phones": 1, "models": 1, "months": 8,
            "mou_Mean": 520.0, "rev_Mean": 50.0, "totmrc_Mean": 50.0, "change_mou": 5.0, "change_rev": 0.0,
            "avg3mou": 510.0, "avg6mou": 495.0, "avgrev": 49.0, "drop_vce_Mean": 2.0, "custcare_Mean": 1.0,
            "ovrmou_Mean": 0.0, "ovrrev_Mean": 0.0, "complete_Mean": 90.0, "attempt_Mean": 105.0, "roam_Mean": 0.0,
            "asl_flag": "N", "creditcd": "Y", "new_cell": "Y", "refurb_new": "N", "dualband": "Y", "hnd_webcap": "WCMB",
            "area": "DALLAS AREA", "prizm_social_one": "T", "marital": "M", "ethnic": "B"
        }
    ])
    return sample_df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(
        page_title="Telecom Churn Intelligence Platform",
        page_icon="telecom",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for crisp UI layout
    st.markdown("""
        <style>
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border-left: 5px solid #1f77b4;
            margin-bottom: 10px;
        }
        .risk-high {
            background-color: #ffebee;
            color: #c62828;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-med {
            background-color: #fff8e1;
            color: #f57f17;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-low {
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1e3d59;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 6px;
            margin-top: 15px;
            margin-bottom: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Top Header Banner
    st.title("Telecom Churn Prediction & Risk Intelligence Platform")
    # st.caption(
    #     "A Production-Grade Decision Support System for Customer Retention | "
    #     "Trained on 100,000 Verified Accounts | 4-Stage Progressive Machine Learning Evolution"
    # )

    # Sidebar: Model Selection and Configuration
    st.sidebar.title("Model Control Panel")

    selected_model_label = st.sidebar.selectbox("Select Predictive Model", list(MODEL_CHOICES.keys()), index=0)
    model_path = MODEL_CHOICES[selected_model_label]

    if not model_path.exists():
        st.sidebar.error(f"Artifact not found: {model_path.name}")
        st.stop()

    model = load_model(str(model_path))
    defaults = load_defaults()
    model_info = MODEL_METRICS.get(selected_model_label, {})

    # Model Performance Scorecard in Sidebar
    st.sidebar.markdown(f"### {model_info.get('badge', 'Active Model')}")
    st.sidebar.write(model_info.get("desc", ""))

    c_sb1, c_sb2 = st.sidebar.columns(2)
    c_sb1.metric("ROC-AUC", model_info.get("roc_auc", "N/A"))
    c_sb2.metric("Accuracy", model_info.get("accuracy", "N/A"))

    c_sb3, c_sb4 = st.sidebar.columns(2)
    c_sb3.metric("Recall", model_info.get("recall", "N/A"))
    c_sb4.metric("Top-10% Lift", model_info.get("lift", "N/A"))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Decision Threshold Tuning")
    custom_threshold = st.sidebar.slider(
        "Classification Cutoff (Threshold)",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Lowering threshold (e.g. 0.40) catches more churners (higher recall) at cost of more false alarms. Raising threshold (e.g. 0.60) catches only highly confident churners."
    )
    if custom_threshold < 0.45:
        st.sidebar.info("Mode: **Aggressive Recall** (Maximizes saved customers)")
    elif custom_threshold > 0.55:
        st.sidebar.info("Mode: **High Precision** (Minimizes wasted contact spend)")
    else:
        st.sidebar.info("Mode: **Balanced Operating Point** (Default standard)")

    # Main Tabs
    tab_single, tab_batch, tab_eda, tab_arch, tab_dict = st.tabs([
        "Single Customer Predictor",
        "Batch CSV Scoring",
        "Descriptive Data Mining (EDA)",
        "Model Architectures & Pipelines",
        "Data Dictionary & Feature Guide"
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: SINGLE CUSTOMER PREDICTOR
    # ═════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown("### Single Subscriber Real-Time Churn Scoring")
        st.write(
            "Input subscriber contractual, financial, and behavioral metrics below to predict 31–60 day defection risk and generate an actionable retention playbook."
        )

        # Presets selector
        st.markdown("##### Quick Preset Profiles")
        p1, p2, p3, p_blank = st.columns([1.5, 1.5, 1.5, 3.5])
        
        preset = "Default"
        if "profile_preset" not in st.session_state:
            st.session_state.profile_preset = "Default"

        if p1.button("Load High-Risk Profile", use_container_width=True):
            st.session_state.profile_preset = "HighRisk"
        if p2.button("Load Low-Risk Profile", use_container_width=True):
            st.session_state.profile_preset = "LowRisk"
        if p3.button("Load Baseline Average", use_container_width=True):
            st.session_state.profile_preset = "Default"

        current_preset = st.session_state.profile_preset
        nd = defaults["numeric_defaults"].copy()
        cd = defaults["categorical_defaults"].copy()
        opts = defaults["categorical_options"]

        if current_preset == "HighRisk":
            nd["eqpdays"] = 580.0
            nd["months"] = 24.0
            nd["hnd_price"] = 29.99
            nd["mou_Mean"] = 210.0
            nd["change_mou"] = -45.0
            nd["totmrc_Mean"] = 35.0
            nd["ovrrev_Mean"] = 18.0
            nd["ovrmou_Mean"] = 35.0
            nd["custcare_Mean"] = 3.0
            cd["asl_flag"] = "Y"
            cd["refurb_new"] = "R"
            st.caption("Currently loaded: **High Risk Defector** (Old phone, steep usage drop, high overages, spending limit active)")
        elif current_preset == "LowRisk":
            nd["eqpdays"] = 75.0
            nd["months"] = 10.0
            nd["hnd_price"] = 249.99
            nd["mou_Mean"] = 750.0
            nd["change_mou"] = 15.0
            nd["totmrc_Mean"] = 65.0
            nd["ovrrev_Mean"] = 0.0
            nd["ovrmou_Mean"] = 0.0
            nd["custcare_Mean"] = 0.0
            cd["asl_flag"] = "N"
            cd["refurb_new"] = "N"
            st.caption("Currently loaded: **Low Risk Loyal** (Recent 5G phone, growing usage, zero overages, no friction)")
        else:
            st.caption("Currently loaded: **Average Baseline Customer** (Cohort median values)")

        # Form Layout
        st.markdown("<div class='section-header'>Inputs: Customer Contract, Usage & Demographic Features</div>", unsafe_allow_html=True)
        
        with st.expander("1. Handset & Equipment Tenure (Top Risk Driver)", expanded=True):
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                eqpdays = st.number_input("Equipment Age (days)", 0, 2000, int(nd["eqpdays"]), help="Days customer has used current phone. #1 predictive driver.")
            with col_d2:
                hnd_price = st.number_input("Handset Value ($)", 0.0, 800.0, float(nd["hnd_price"]), help="Retail purchase price of current phone.")
            with col_d3:
                months = st.number_input("Account Tenure (months)", 1, 72, int(nd["months"]), help="Total months customer has maintained service.")
            with col_d4:
                phones = st.number_input("Total Handsets Issued", 1, 20, int(nd["phones"]))

        with st.expander("2. Usage & Billing Dynamics", expanded=True):
            col_u1, col_u2, col_u3, col_u4 = st.columns(4)
            with col_u1:
                mou_Mean = st.number_input("Monthly Minutes (MOU)", 0.0, 6000.0, float(nd["mou_Mean"]))
                change_mou = st.number_input("Change in Minutes (% vs 3-mo)", -2000.0, 2000.0, float(nd["change_mou"]), help="Negative values indicate falling usage engagement.")
            with col_u2:
                rev_Mean = st.number_input("Total Monthly Revenue ($)", -50.0, 800.0, float(nd["rev_Mean"]))
                change_rev = st.number_input("Change in Revenue ($)", -500.0, 500.0, float(nd["change_rev"]))
            with col_u3:
                totmrc_Mean = st.number_input("Base Monthly Recurring Charge ($)", -30.0, 300.0, float(nd["totmrc_Mean"]))
                avg3mou = st.number_input("3-Month Average Minutes", 0.0, 6000.0, float(nd["avg3mou"]))
            with col_u4:
                avg6mou = st.number_input("6-Month Average Minutes", 0.0, 6000.0, float(nd.get("avg6mou", nd["avg3mou"])))
                avgrev = st.number_input("Lifetime Average Revenue ($)", 0.0, 600.0, float(nd["avgrev"]))

        with st.expander("3. Overage Fees & Service Quality Friction", expanded=True):
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                ovrrev_Mean = st.number_input("Overage Revenue ($)", 0.0, 300.0, float(nd["ovrrev_Mean"]), help="Unexpected overage fees driving bill shock.")
                ovrmou_Mean = st.number_input("Overage Minutes", 0.0, 1500.0, float(nd["ovrmou_Mean"]))
            with col_f2:
                drop_vce_Mean = st.number_input("Dropped Voice Calls (Mean)", 0.0, 100.0, float(nd["drop_vce_Mean"]))
                custcare_Mean = st.number_input("Customer Care Calls", 0.0, 50.0, float(nd["custcare_Mean"]), help="Inquiries to support indicating customer pain.")
            with col_f3:
                attempt_Mean = st.number_input("Total Attempted Calls", 0.0, 600.0, float(nd["attempt_Mean"]))
                complete_Mean = st.number_input("Total Completed Calls", 0.0, 600.0, float(nd["complete_Mean"]))
            with col_f4:
                roam_Mean = st.number_input("Roaming Calls", 0.0, 100.0, float(nd["roam_Mean"]))
                models = st.number_input("Unique Models Issued", 1, 20, int(nd["models"]))

        with st.expander("4. Customer Segments & Geographic Area", expanded=False):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                asl_flag = st.selectbox("Account Spending Limit", opts["asl_flag"], index=opts["asl_flag"].index(cd["asl_flag"]) if cd["asl_flag"] in opts["asl_flag"] else 0)
                creditcd = st.selectbox("Credit Card on File", opts["creditcd"], index=opts["creditcd"].index(str(cd["creditcd"])) if str(cd["creditcd"]) in opts["creditcd"] else 0)
                refurb_new = st.selectbox("Handset Condition", opts["refurb_new"], index=opts["refurb_new"].index(cd["refurb_new"]) if cd["refurb_new"] in opts["refurb_new"] else 0)
                new_cell = st.selectbox("New Cellular User", opts["new_cell"], index=opts["new_cell"].index(cd["new_cell"]) if cd["new_cell"] in opts["new_cell"] else 0)
            with col_s2:
                area = st.selectbox("Regional Operating Territory", opts["area"], index=opts["area"].index(cd["area"]) if cd["area"] in opts["area"] else 0)
                hnd_webcap = st.selectbox("Web Capability", opts["hnd_webcap"], index=opts["hnd_webcap"].index(cd["hnd_webcap"]) if cd["hnd_webcap"] in opts["hnd_webcap"] else 0)
                dualband = st.selectbox("Dualband Device", opts["dualband"], index=opts["dualband"].index(cd["dualband"]) if cd["dualband"] in opts["dualband"] else 0)
            with col_s3:
                prizm_social_one = st.selectbox("PRIZM Lifestyle Cluster", opts["prizm_social_one"], index=opts["prizm_social_one"].index(cd["prizm_social_one"]) if cd["prizm_social_one"] in opts["prizm_social_one"] else 0)
                marital = st.selectbox("Marital Status", opts["marital"], index=opts["marital"].index(str(cd["marital"])) if str(cd["marital"]) in opts["marital"] else 0)
                ethnic = st.selectbox("Ethnicity Rollup", opts["ethnic"], index=opts["ethnic"].index(str(cd["ethnic"])) if str(cd["ethnic"]) in opts["ethnic"] else 0)

        # Run Prediction
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict Customer Churn Risk", type="primary", use_container_width=True)

        if predict_button:
            input_dict = {
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
                "plcd_vce_Mean": attempt_Mean,
                "roam_Mean": roam_Mean,
                "asl_flag": asl_flag,
                "creditcd": creditcd,
                "new_cell": new_cell,
                "refurb_new": refurb_new,
                "dualband": dualband,
                "hnd_webcap": hnd_webcap,
                "area": area,
                "prizm_social_one": prizm_social_one,
                "marital": marital,
                "ethnic": ethnic,
            }

            raw_df = pd.DataFrame([input_dict])
            try:
                scored_df = predict_frame(model, selected_model_label, raw_df, custom_threshold)
                res = scored_df.iloc[0]
                proba = float(res["churn_proba"])
                pred = int(res["churn_pred"])
                band = str(res["risk_band"])

                st.markdown("<div class='section-header'>Outputs: Churn Probability & Strategic Retention Diagnosis</div>", unsafe_allow_html=True)
                
                # Output KPI Cards
                out_col1, out_col2, out_col3 = st.columns(3)
                
                with out_col1:
                    st.metric("Predicted Churn Probability", f"{proba:.1%}")
                with out_col2:
                    status_text = "Churn (High Risk)" if pred == 1 else "Stay (Low Risk)"
                    st.metric("Model Binary Classification", status_text)
                with out_col3:
                    st.metric("Assigned Risk Band", band)

                # Progress bar risk meter
                st.write("**Visual Risk Exposure Gauge:**")
                st.progress(min(max(proba, 0.0), 1.0))

                # Prescriptive Playbook Container
                badge_type, rec_title, rec_body = get_prescriptive_recommendation(res)
                rec_msg = f"### {rec_title}\n\n{rec_body}"
                if badge_type == "error":
                    st.error(rec_msg)
                elif badge_type == "warning":
                    st.warning(rec_msg)
                elif badge_type == "info":
                    st.info(rec_msg)
                else:
                    st.success(rec_msg)

                # Factor contribution diagnosis
                with st.expander("Key Behavioral Risk Drivers for this Subscriber"):
                    diag_cols = st.columns(3)
                    with diag_cols[0]:
                        if eqpdays >= 400:
                            st.write(f"• **Device Aging:** Current handset is {int(eqpdays)} days old (elevates defection probability).")
                        else:
                            st.write(f"• **Device Recent:** Handset is {int(eqpdays)} days old (low hardware friction).")
                    with diag_cols[1]:
                        if change_mou < -15:
                            st.write(f"• **Usage Momentum:** Call volume fell by {change_mou:.1f}% (steep engagement drop).")
                        else:
                            st.write(f"• **Usage Momentum:** Call volume change is {change_mou:.1f}% (stable trajectory).")
                    with diag_cols[2]:
                        ovr_p = ovrrev_Mean / (abs(totmrc_Mean) + 1e-6)
                        if ovr_p >= 0.25:
                            st.write(f"• **Bill Shock:** Overage fees are {ovr_p:.1%} of monthly base plan.")
                        else:
                            st.write(f"• **Bill Shock:** Overage ratio is negligible ({ovr_p:.1%}).")

                # Transformed row inspector
                with st.expander("Inspect Full Transformed Feature Vector Sent to Engine"):
                    st.dataframe(scored_df)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: BATCH CSV SCORING
    # ═════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("### Batch Portfolio Churn Scoring")
        st.write(
            "Upload a customer extract CSV to score thousands of accounts in parallel. "
            "The system calculates individual defection probabilities, assigns risk tiers, and generates actionable retention playbooks."
        )

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            uploaded_file = st.file_uploader("Upload Customer Portfolio CSV", type=["csv"])
        with b_c2:
            st.markdown("**Need a sample to test?**")
            st.download_button(
                "Download Sample Test CSV",
                data=generate_sample_csv(),
                file_name="sample_telecom_customers.csv",
                mime="text/csv",
                help="Download a ready-to-test CSV with 5 diverse customer records."
            )

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded portfolio: **{len(batch_df):,} accounts** across **{batch_df.shape[1]} attributes**.")
            st.dataframe(batch_df.head(5))

            if st.button("Run Batch Churn Prediction Pipeline", type="primary"):
                try:
                    with st.spinner("Executing feature engineering and ensemble scoring..."):
                        scored_batch = predict_frame(model, selected_model_label, batch_df, custom_threshold)

                    st.markdown("<div class='section-header'>Batch Scoring Summary & Portfolio Diagnostics</div>", unsafe_allow_html=True)
                    
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    total_n = len(scored_batch)
                    high_n = (scored_batch["risk_band"] == "Higher Risk").sum()
                    med_n = (scored_batch["risk_band"] == "Medium Risk").sum()
                    low_n = (scored_batch["risk_band"] == "Lower Risk").sum()
                    avg_p = scored_batch["churn_proba"].mean()

                    kpi1.metric("Total Scored", f"{total_n:,}")
                    kpi2.metric("High Risk Defectors", f"{high_n:,}", f"{high_n/total_n:.1%}")
                    kpi3.metric("Medium Risk Accounts", f"{med_n:,}", f"{med_n/total_n:.1%}")
                    kpi4.metric("Average Churn Probability", f"{avg_p:.1%}")

                    # Chart & Table
                    chart_col, dl_col = st.columns([2, 1])
                    with chart_col:
                        st.write("**Portfolio Risk Tier Distribution:**")
                        st.bar_chart(scored_batch["risk_band"].value_counts())
                    with dl_col:
                        st.write("**Download Scored Portfolio:**")
                        csv_data = scored_batch.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Enriched Predictions CSV",
                            data=csv_data,
                            file_name="scored_churn_portfolio.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )

                    st.markdown("#### Scored Customer Preview (Sorted by Highest Risk)")
                    st.dataframe(scored_batch.sort_values(by="churn_proba", ascending=False).head(50), use_container_width=True)

                except Exception as e:
                    st.error(f"Batch Scoring Error: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: DESCRIPTIVE DATA MINING (EDA FIGURES)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_eda:
        st.markdown("### Descriptive Data Mining Visualizations")
        st.write(
            "Empirical findings and exploratory charts generated across the 100,000-subscriber dataset. "
            "These data mining discoveries directly informed feature engineering and algorithmic selection."
        )

        eda_sections = st.radio(
            "Select EDA Mining Topic:",
            [
                "1. Data Quality & Target Distribution",
                "2. Feature Distributions & Tails",
                "3. Churn Drivers & Bivariate Analysis",
                "4. Customer Segments & Geography",
                "5. Correlation & Class Overlap"
            ],
            horizontal=True
        )

        if eda_sections.startswith("1."):
            st.subheader("1. Data Quality, Missingness & Target Distribution")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                img_path = EDA_DIR / "01_missing_values.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 1: Missing Data Asymmetry (CRM Demographics vs Switch Usage)", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Usage/billing fields from switches are 100% complete, while demographic data (income, dwelling size, car counts) is 25–70% missing. "
                    "This informed the decision to base core models on dense behavioral logs rather than sparse demographics."
                )
            with col_e2:
                img_path = EDA_DIR / "02_churn_distribution.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 2: Target Variable Balance (49.6% Churn vs 50.4% Retained)", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** The near-even 50/50 balance establishes a 50.4% naïve majority baseline ('always predict Stay'). "
                    "Because classes are balanced, synthetic oversampling (SMOTE) was rejected."
                )

        elif eda_sections.startswith("2."):
            st.subheader("2. Continuous Feature Distributions & Quality Auditing")
            img_path = EDA_DIR / "03_feature_distributions.png"
            if img_path.exists():
                st.image(str(img_path), caption="Figure 3: Histograms of Key Continuous Usage & Billing Metrics", use_container_width=True)
            st.info(
                "**Mining Takeaway:** Features like usage change (`change_mou`) and revenue change (`change_rev`) exhibit extreme leptokurtic tails (−1000% to +1500%). "
                "This necessitated train-only Winsorization (clamping at 1st–99th percentiles) to prevent outliers from distorting splits."
            )

            img_path_sanity = EDA_DIR / "09_client_vs_record_sanity.png"
            if img_path_sanity.exists():
                st.image(str(img_path_sanity), caption="Figure 4: Lifetime vs Recent Rolling Usage Consistency Check", use_container_width=True)
            st.info(
                "**Mining Takeaway:** Bivariate scatter plots between customer lifetime metrics and recent 3-month records verify join integrity and positive correlation."
            )

        elif eda_sections.startswith("3."):
            st.subheader("3. Churn Driver Bivariate Analysis")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                img_path = EDA_DIR / "04_features_by_churn.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 5: Feature Distributions Grouped by Churn Status", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Churners have significantly older equipment (+101 days older on average) and negative usage momentum (−15.8% vs −1.2% for stayers)."
                )
            with col_b2:
                img_path = EDA_DIR / "04b_features_by_churn_dots.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 6: Strip Distributions Across Usage & Handset Tenure", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Bivariate dots confirm that handset aging beyond 400 days sharply elevates defection risk regardless of base tariff plan."
                )

        elif eda_sections.startswith("4."):
            st.subheader("4. Categorical Segments & Regional Variance")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                img_path = EDA_DIR / "05_categorical_churn_rates.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 7: Churn Rates Across Account Flags & Hardware Segments", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Accounts subject to spending limits (`asl_flag = Y`) have higher churn (54.2% vs 48.9%). "
                    "Refurbished handset recipients also experience elevated defection (53.8% vs 49.1%)."
                )
            with col_c2:
                img_path = EDA_DIR / "06_geographic_churn_rates.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 8: Geographic Defection Rates by Regional Operating Territory", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Churn varies by territory (46.2% Mid-Atlantic vs 53.1% South Florida and Mountain regions), reflecting regional competitive promotional intensity."
                )

        elif eda_sections.startswith("5."):
            st.subheader("5. Correlation Heatmap & Joint Class Overlap")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                img_path = EDA_DIR / "07_correlation_heatmap.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 9: Pearson Correlation Matrix & Multicollinearity Clusters", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Massive redundancy exists across usage windows (lifetime MOU, 6-mo MOU, 3-mo MOU share $r > 0.90$). "
                    "This validated dropping cloned windows to prevent multicollinearity in linear models."
                )
            with col_m2:
                img_path = EDA_DIR / "08_pairplot.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Figure 10: Pairplot Highlighting Heavy Multi-Dimensional Class Overlap", use_container_width=True)
                st.info(
                    "**Mining Takeaway:** Scatter plots reveal that churners and stayers heavily overlap across all 2D projections. "
                    "There is no simple linear boundary, proving why tree-based models and gradient boosting are required."
                )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4: MODEL ARCHITECTURES & PIPELINES
    # ═════════════════════════════════════════════════════════════════════════
    with tab_arch:
        st.markdown("### 4-Stage Progressive Model Architecture")
        st.write(
            "Every model upgrade in this project was justified through controlled holdout benchmarking on the exact same 20,000 test accounts."
        )

        arch_tab1, arch_tab2, arch_tab3, arch_tab4, arch_bench = st.tabs([
            "Stage 1: Logistic Regression",
            "Stage 2: Random Forest",
            "Stage 3: Tuned XGBoost",
            "Stage 4: Soft-Vote Ensemble (Champion)",
            "Master Comparison Benchmark"
        ])

        with arch_tab1:
            st.markdown("#### Stage 1 Architecture: Transparent Linear Baseline")
            img_s1 = DIAG_DIR / "stage_1.png"
            if img_s1.exists():
                st.image(str(img_s1), caption="Stage 1 Pipeline: Scaled Lean Matrix + L2 Logistic Regression", use_container_width=True)
            st.write(
                "**Pipeline Specs:** 25 Lean features, median imputation, `StandardScaler`, One-Hot Encoding, L2 regularization (`max_iter=1000`). "
                "Achieved 58.02% accuracy and 0.6090 ROC-AUC. Proved that linear decision boundaries cannot resolve class overlap."
            )

        with arch_tab2:
            st.markdown("#### Stage 2 Architecture: Non-Linear Tree Bagging")
            img_s2 = DIAG_DIR / "stage_2.png"
            if img_s2.exists():
                st.image(str(img_s2), caption="Stage 2 Pipeline: Same Lean Features + Random Forest (200 Trees)", use_container_width=True)
            st.write(
                "**Pipeline Specs:** Exact same 25 lean features, 200 trees, max depth 12, min leaf 20. "
                "Accuracy jumped to 61.33% (+3.3 pp) and ROC-AUC reached 0.6658 (+0.057). Proved that non-linear interaction rules are mandatory."
            )

        with arch_tab3:
            st.markdown("#### Stage 3 Architecture: Sequential Gradient Boosting")
            img_s3 = DIAG_DIR / "stage_3.png"
            if img_s3.exists():
                st.image(str(img_s3), caption="Stage 3 Pipeline: 38 Rich Features + Tuned XGBoost Booster", use_container_width=True)
            st.write(
                "**Pipeline Specs:** 38 Rich features with derived interaction terms (`eqpdays_x_change_mou`, `overage_ratio`, `mou_per_month`). "
                "Tuned via 20-draw randomized CV (`max_depth=7`, `learning_rate=0.03`, `n_estimators=600`, `subsample=0.8`). Achieved 63.27% accuracy and 0.6889 ROC-AUC."
            )

        with arch_tab4:
            st.markdown("#### Stage 4 Architecture: Dual-Architecture Soft-Vote Ensemble (Champion)")
            img_s4 = DIAG_DIR / "stage_4.png"
            if img_s4.exists():
                st.image(str(img_s4), caption="Stage 4 Pipeline: Level-wise XGBoost (50%) + Leaf-wise LightGBM (50%)", use_container_width=True)
            st.write(
                "**Pipeline Specs:** Blends level-wise XGBoost with leaf-wise LightGBM (`num_leaves=63`, `n_estimators=800`, `reg_alpha=0.1`). "
                "Soft-vote averaging decorrelates individual tree errors on difficult border cases, setting project records across every metric: "
                "**63.49% accuracy, 0.6899 ROC-AUC, and 1.59× top-10% lift**."
            )

        with arch_bench:
            st.markdown("#### Master Benchmark Comparison Across All 4 Stages")
            st.markdown(
                """
                | Metric | Stage 1 (LogReg) | Stage 2 (Random Forest) | Stage 3 (Tuned XGBoost) | Stage 4 (Soft-Vote Ensemble) | Net Gain (1 → 4) |
                |---|:---:|:---:|:---:|:---:|:---:|
                | **Feature Tier** | Lean (25) | Lean (25) | Rich (38) | **Rich (38)** | +13 features |
                | **Holdout Accuracy** | 0.5802 | 0.6133 | 0.6327 | **0.6349** | **+5.47 pp** |
                | **ROC-AUC Score** | 0.6090 | 0.6658 | 0.6889 | **0.6899** | **+0.0809** |
                | **Churner Recall** | 0.5786 | 0.6645 | 0.6392 | **0.6422** | **+0.0636** |
                | **F1-Score** | 0.5774 | 0.6301 | 0.6330 | **0.6355** | **+0.0581** |
                | **Top-10% Decile Lift** | 1.27× | 1.47× | 1.58× | **1.59×** | **+0.32×** |
                """
            )
            st.info(
                "**Scientific Takeaway:** The accuracy ceiling near 63.5% is driven by human behavioral factors external to CRM logs. "
                "The system's true commercial value is its **1.59× Top-10% Lift**, allowing marketing retention teams to capture 78.8% actual churners in their top target band."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5: DATA DICTIONARY
    # ═════════════════════════════════════════════════════════════════════════
    with tab_dict:
        st.markdown("### Feature Dictionary & Modeling Definitions")
        st.write("Reference guide explaining all 38+ attributes used across feature engineering and prediction.")

        dict_cat = st.selectbox(
            "Select Feature Group:",
            ["Key Risk Levers & Derived Features", "Usage & Minutes of Use (MOU)", "Revenue, Tariff & Overage", "Customer Care & Service Friction", "Handset & Demographics"]
        )

        if dict_cat == "Key Risk Levers & Derived Features":
            st.dataframe(pd.DataFrame([
                ("eqpdays", "Continuous", "Number of days customer has possessed current phone (#1 churn driver)."),
                ("overage_ratio", "Derived Ratio", "ovrrev_Mean / (|totmrc_Mean| + eps). Quantifies bill shock relative to plan price."),
                ("mou_decline_flag", "Derived Binary", "1 if change_mou < 0, else 0. Flags directional disengagement momentum."),
                ("eqpdays_x_change_mou", "Derived Interaction", "Product of equipment age and change in minutes."),
                ("drop_rate", "Derived Ratio", "drop_vce_Mean / (plcd_vce_Mean + eps). Voice call failure rate."),
                ("care_per_mou", "Derived Ratio", "custcare_Mean / (mou_Mean + eps). Support inquiry intensity per minute."),
            ], columns=["Feature", "Type", "Operational Definition"]), hide_index=True, use_container_width=True)

        elif dict_cat == "Usage & Minutes of Use (MOU)":
            st.dataframe(pd.DataFrame([
                ("mou_Mean", "Continuous", "Mean monthly voice minutes of use."),
                ("change_mou", "Continuous", "Percentage change in minutes vs preceding 3-month baseline."),
                ("avg3mou", "Continuous", "Average monthly minutes over previous 3 months."),
                ("avg6mou", "Continuous", "Average monthly minutes over previous 6 months."),
                ("complete_Mean", "Continuous", "Mean completed calls placed."),
                ("attempt_Mean", "Continuous", "Mean attempted calls placed."),
                ("roam_Mean", "Continuous", "Mean roaming call count."),
            ], columns=["Feature", "Type", "Operational Definition"]), hide_index=True, use_container_width=True)

        elif dict_cat == "Revenue, Tariff & Overage":
            st.dataframe(pd.DataFrame([
                ("rev_Mean", "Continuous", "Mean monthly total billed revenue."),
                ("totmrc_Mean", "Continuous", "Mean monthly base recurring subscription plan charge."),
                ("ovrrev_Mean", "Continuous", "Mean revenue incurred from overage penalties."),
                ("ovrmou_Mean", "Continuous", "Mean minutes used exceeding plan allowance."),
                ("change_rev", "Continuous", "Percentage change in billed revenue vs 3-month baseline."),
                ("avgrev", "Continuous", "Lifetime average monthly revenue."),
            ], columns=["Feature", "Type", "Operational Definition"]), hide_index=True, use_container_width=True)

        elif dict_cat == "Customer Care & Service Friction":
            st.dataframe(pd.DataFrame([
                ("drop_vce_Mean", "Continuous", "Mean dropped voice calls due to network failure."),
                ("custcare_Mean", "Continuous", "Mean calls placed to customer care support center."),
            ], columns=["Feature", "Type", "Operational Definition"]), hide_index=True, use_container_width=True)

        elif dict_cat == "Handset & Demographics":
            st.dataframe(pd.DataFrame([
                ("hnd_price", "Continuous", "Current handset purchase price."),
                ("months", "Continuous", "Total customer tenure in months."),
                ("asl_flag", "Categorical (Y/N)", "Account spending limit credit restriction flag."),
                ("creditcd", "Categorical (Y/N)", "Credit card on file with carrier."),
                ("refurb_new", "Categorical (R/N)", "Handset was issued as refurbished (R) or brand new (N)."),
                ("area", "Categorical (19 areas)", "Regional operating geographic territory."),
                ("prizm_social_one", "Categorical (Letters)", "PRIZM household lifestyle segmentation group."),
            ], columns=["Feature", "Type", "Operational Definition"]), hide_index=True, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        "Telecom Churn Intelligence Platform | UCSY Data Mining Project | "
        "Production Ensembles: `models/stage4_ensemble.joblib` & `models/final_model.joblib`"
    )


if __name__ == "__main__":
    main()
