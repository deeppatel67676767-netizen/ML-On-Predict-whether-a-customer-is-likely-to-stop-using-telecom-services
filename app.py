"""
Customer Churn Prediction — Streamlit Application
"""

import os
import sys
import warnings
import io

warnings.filterwarnings("ignore")

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.data_preprocessing import load_dataset, clean_dataset, prepare_data
from src.train_models import train_all_models
from src.evaluate_models import evaluate_all_models, select_best_model, get_confusion_matrix
from src.predict import predict_churn

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1100px; }
    h1 { color: #1a1a2e; font-size: 1.8rem; }
    h2 { color: #16213e; font-size: 1.3rem; border-bottom: 2px solid #2563eb; padding-bottom: 4px; }
    h3 { color: #1e3a5f; font-size: 1.1rem; }
    .metric-box { background: #f0f4ff; border-left: 4px solid #2563eb;
                  padding: 12px 16px; border-radius: 6px; margin: 4px 0; }
    .result-churn { background: #fff0f0; border-left: 4px solid #dc2626;
                    padding: 14px; border-radius: 6px; }
    .result-nochurn { background: #f0fff4; border-left: 4px solid #16a34a;
                      padding: 14px; border-radius: 6px; }
    .stButton>button { background-color: #2563eb; color: white; border: none;
                       border-radius: 6px; padding: 8px 20px; font-size: 0.95rem; }
    .stButton>button:hover { background-color: #1d4ed8; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(source="default", uploaded_bytes=None):
    try:
        if source == "upload" and uploaded_bytes is not None:
            df_raw = pd.read_csv(io.BytesIO(uploaded_bytes))
            if "Churn" not in df_raw.columns:
                return None, "Uploaded file does not contain a 'Churn' column."
            df = clean_dataset(df_raw)
            return df, None
        else:
            if not os.path.exists(DEFAULT_DATA_PATH):
                return None, (
                    f"Default dataset not found at `{DEFAULT_DATA_PATH}`.\n\n"
                    "Download the IBM Telco Customer Churn dataset from "
                    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
                    "and place `Telco-Customer-Churn.csv` inside the `data/` folder."
                )
            df = load_dataset(DEFAULT_DATA_PATH)
            return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


@st.cache_resource(show_spinner=False)
def run_training_pipeline(data_key):
    """Train all models. Cached by data_key so it only reruns when data changes."""
    df, err = load_data(data_key[0], data_key[1])
    if df is None:
        return None

    X_train, X_test, y_train, y_test, preprocessor, feature_cols = prepare_data(df)
    trained_models = train_all_models(X_train, y_train)
    metrics_df = evaluate_all_models(trained_models, X_test, y_test)
    best_name, best_model = select_best_model(metrics_df, trained_models)
    cm = get_confusion_matrix(best_model, X_test, y_test)

    # Save best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": best_model, "preprocessor": preprocessor,
                 "feature_cols": feature_cols, "best_name": best_name}, MODEL_PATH)

    # Save metrics CSV
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    metrics_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "trained_models": trained_models,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "confusion_matrix": cm
    }


# ── Helper: small matplotlib figure ──────────────────────────────────────────
def small_fig(w=5, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ── App header ────────────────────────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.markdown("*Machine Learning Based Customer Retention Analysis*")
st.markdown("---")

# ── Dataset source selector ───────────────────────────────────────────────────
with st.expander("📂 Dataset Source", expanded=False):
    upload_file = st.file_uploader("Upload your own dataset (CSV)", type=["csv"])

if upload_file:
    data_key = ("upload", upload_file.read())
    upload_file.seek(0)  # reset after reading
    df, data_error = load_data("upload", data_key[1])
else:
    data_key = ("default", None)
    df, data_error = load_data("default", None)

# ── Navigation ────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Dashboard", "🤖 Model Comparison", "🔮 Predict Churn", "📋 Dataset Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Dashboard")

    if data_error:
        st.error(data_error)
    elif df is None:
        st.warning("No dataset loaded.")
    else:
        total = len(df)
        churned = int(df["Churn"].sum())
        retained = total - churned
        churn_rate = round(churned / total * 100, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("Churned", f"{churned:,}")
        c3.metric("Retained", f"{retained:,}")
        c4.metric("Churn Rate", f"{churn_rate}%")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        # Chart 1 — Churn distribution
        with col1:
            st.subheader("Churn Distribution")
            fig, ax = small_fig(3.5, 3)
            labels = ["Retained", "Churned"]
            sizes = [retained, churned]
            colors = ["#2563eb", "#ef4444"]
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 9})
            ax.set_title("Churn vs Retained", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Chart 2 — Churn by Contract
        with col2:
            st.subheader("By Contract Type")
            if "Contract" in df.columns:
                raw_df, _ = load_data(data_key[0], data_key[1])
                if raw_df is not None and "Contract" in raw_df.columns:
                    # Use the original loaded (clean) df which has Churn as 0/1
                    ct = df.groupby("Contract")["Churn"].mean() * 100
                    fig, ax = small_fig(3.5, 3)
                    ct.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="white")
                    ax.set_title("Churn Rate by Contract", fontsize=10)
                    ax.set_ylabel("Churn Rate (%)", fontsize=8)
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=20, labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Contract column not available.")

        # Chart 3 — Churn by Internet Service
        with col3:
            st.subheader("By Internet Service")
            if "InternetService" in df.columns:
                is_rate = df.groupby("InternetService")["Churn"].mean() * 100
                fig, ax = small_fig(3.5, 3)
                is_rate.plot(kind="bar", ax=ax, color="#7c3aed", edgecolor="white")
                ax.set_title("Churn Rate by Internet Service", fontsize=10)
                ax.set_ylabel("Churn Rate (%)", fontsize=8)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=15, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("InternetService column not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Model Comparison")

    if data_error:
        st.error(data_error)
    elif df is None:
        st.warning("No dataset loaded.")
    else:
        with st.spinner("Training models — this may take a moment..."):
            results = run_training_pipeline(data_key)

        if results is None:
            st.error("Model training failed. Check dataset format.")
        else:
            metrics_df = results["metrics_df"]
            best_name = results["best_name"]
            cm = results["confusion_matrix"]

            st.success(f"Best Model: **{best_name}** (selected by F1 Score)")
            st.markdown("---")

            # Metrics table
            st.subheader("Performance Metrics")
            display_df = metrics_df.copy()
            display_df["ROC-AUC"] = display_df["ROC-AUC"].fillna("N/A")
            display_df = display_df.set_index("Model")
            st.dataframe(display_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1 Score": "{:.4f}"
            }), use_container_width=True)

            st.markdown("---")
            col1, col2 = st.columns(2)

            # Confusion matrix
            with col1:
                st.subheader(f"Confusion Matrix — {best_name}")
                fig, ax = small_fig(4, 3.2)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Retained", "Churned"],
                            yticklabels=["Retained", "Churned"],
                            annot_kws={"size": 11})
                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("Actual", fontsize=9)
                ax.set_title("Confusion Matrix", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # F1 comparison bar chart
            with col2:
                st.subheader("F1 Score Comparison")
                fig, ax = small_fig(4, 3.2)
                colors = ["#2563eb" if m == best_name else "#93c5fd"
                          for m in metrics_df["Model"]]
                ax.barh(metrics_df["Model"], metrics_df["F1 Score"],
                        color=colors, edgecolor="white")
                ax.set_xlabel("F1 Score", fontsize=9)
                ax.set_title("F1 Score by Model", fontsize=10)
                ax.tick_params(labelsize=8)
                ax.set_xlim(0, 1)
                for i, v in enumerate(metrics_df["F1 Score"]):
                    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Predict Customer Churn
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Predict Customer Churn")

    if data_error:
        st.error(data_error)
    elif df is None:
        st.warning("No dataset loaded.")
    else:
        with st.spinner("Loading model..."):
            results = run_training_pipeline(data_key)

        if results is None:
            st.error("Could not load model.")
        else:
            preprocessor = results["preprocessor"]
            feature_cols = results["feature_cols"]
            best_model = results["best_model"]
            best_name = results["best_name"]

            st.caption(f"Using model: **{best_name}**")
            st.markdown("Enter customer details below and click **Predict Customer Churn**.")

            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Personal Info**")
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
                    partner = st.selectbox("Partner", ["Yes", "No"])
                    dependents = st.selectbox("Dependents", ["No", "Yes"])
                    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)

                with col2:
                    st.markdown("**Services**")
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

                with col3:
                    st.markdown("**Billing & Contract**")
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                    payment_method = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ])
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0,
                                                       max_value=200.0, value=65.0, step=0.5)
                    total_charges = st.number_input("Total Charges ($)", min_value=0.0,
                                                     max_value=10000.0,
                                                     value=float(tenure * monthly_charges),
                                                     step=1.0)

                submitted = st.form_submit_button("🔮 Predict Customer Churn", use_container_width=True)

            if submitted:
                input_dict = {
                    "gender": gender,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges
                }

                try:
                    result = predict_churn(best_model, preprocessor, feature_cols, input_dict)
                    st.markdown("---")
                    st.subheader("Prediction Result")

                    is_churn = result["prediction"] == "Churn"
                    box_class = "result-churn" if is_churn else "result-nochurn"
                    icon = "🔴" if is_churn else "🟢"

                    st.markdown(f"""
                    <div class="{box_class}">
                        <h3>{icon} {result['prediction']}</h3>
                        <p><strong>Risk Category:</strong> {result['risk_category']}</p>
                        <p><strong>Churn Probability:</strong> {result['churn_probability']}%</p>
                        <p><strong>Retention Probability:</strong> {result['retain_probability']}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown(f"**Insight:** {result['explanation']}")

                    # Probability bar
                    st.markdown("**Probability Breakdown**")
                    prob_col1, prob_col2 = st.columns(2)
                    prob_col1.metric("Churn Probability", f"{result['churn_probability']}%")
                    prob_col2.metric("Retention Probability", f"{result['retain_probability']}%")
                    st.progress(int(result["churn_probability"]))

                    st.caption(
                        "⚠️ This is a machine learning prediction based on patterns in historical data. "
                        "It is not a guarantee of actual customer behavior."
                    )

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dataset Information
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Dataset Information")

    if data_error:
        st.error(data_error)
    elif df is None:
        st.warning("No dataset loaded.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.markdown("---")
        st.subheader("Target Variable Distribution")
        churn_counts = df["Churn"].value_counts()
        label_map = {1: "Churned (1)", 0: "Retained (0)"}
        churn_counts.index = churn_counts.index.map(label_map)
        st.dataframe(churn_counts.reset_index().rename(
            columns={"index": "Class", "Churn": "Count"}))

        st.subheader("Column Names")
        cols_df = pd.DataFrame({
            "Column": df.columns.tolist(),
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing": [df[c].isnull().sum() for c in df.columns]
        })
        st.dataframe(cols_df, use_container_width=True)

        st.subheader("Dataset Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Basic Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
