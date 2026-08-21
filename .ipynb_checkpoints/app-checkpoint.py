"""
Customer Churn Prediction — Streamlit App
Loads the trained sklearn Pipeline (StandardScaler + OneHotEncoder + XGBoost, tuned via
RandomizedSearchCV) and predicts churn probability for a single customer entered via the UI.

Matches customerchurn.ipynb exactly:
  - engineered features: tenure_group, charge_per_month (no interaction flag)
  - gender is kept as a model feature
  - TotalCharges is collected but dropped before prediction (only used to derive charge_per_month)

Run with:  streamlit run app.py
Requires:  models/churn_pipeline.pkl to exist (produced by the notebook's export cell)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    """Cached so the pipeline is loaded from disk only once per session, not on every rerun."""
    return joblib.load("models/churn_pipeline.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "models/churn_pipeline.pkl not found. Run customerchurn.ipynb through the export "
        "cell first, then place the saved `models/` folder next to this app."
    )
    st.stop()

st.title("📉 Customer Churn Predictor")
st.caption("XGBoost (RandomizedSearchCV-tuned) — trained on IBM Telco Customer Churn data")

st.markdown("Enter a customer's details below to estimate their churn risk.")

with st.form("customer_form"):
    st.subheader("Account details")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=0.5)
    with col2:
        total_charges = st.number_input(
            "Total charges to date ($)", min_value=0.0, value=840.0, step=1.0,
            help="Used only to compute charge-per-month — not sent to the model directly."
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    st.subheader("Demographics")
    col3, col4, col5 = st.columns(3)
    with col3:
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
    with col4:
        partner = st.selectbox("Has partner", ["No", "Yes"])
    with col5:
        dependents = st.selectbox("Has dependents", ["No", "Yes"])

    st.subheader("Services")
    col7, col8 = st.columns(2)
    with col7:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    with col8:
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    st.subheader("Billing")
    col9, col10 = st.columns(2)
    with col9:
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    with col10:
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

if submitted:
    # ── Engineer the same 2 features created in the notebook — same bins, same formula ──
    tenure_group = pd.cut(
        [tenure],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        include_lowest=True,   # matches the notebook: tenure=0 falls into the first bucket
    )[0]

    charge_per_month = total_charges / (tenure + 1)

    # ── Build the row with exactly the columns the pipeline expects ────────────
    # (TotalCharges itself is NOT included — it was dropped after engineering charge_per_month)
    row = pd.DataFrame([{
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
        "tenure_group": tenure_group,
        "charge_per_month": charge_per_month,
    }])

    prob = model.predict_proba(row)[0, 1]
    pred = model.predict(row)[0]

    if prob < 0.4:
        risk_label, risk_color = "Low Risk", "green"
    elif prob < 0.7:
        risk_label, risk_color = "Medium Risk", "orange"
    else:
        risk_label, risk_color = "High Risk", "red"

    st.divider()
    st.subheader("Prediction")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Churn probability", f"{prob:.1%}")
    with c2:
        st.markdown(f"### :{risk_color}[{risk_label}]")

    st.progress(min(prob, 1.0))

    if pred == 1:
        st.warning(
            "This customer is predicted to **churn**. Consider proactive retention outreach — "
            "SHAP analysis on this model shows contract type, tenure, and internet/security "
            "add-ons carry the most weight in these predictions."
        )
    else:
        st.success("This customer is predicted to **stay**.")

    with st.expander("What drives this model's predictions (overall, not per-customer)"):
        st.markdown(
            "- **Contract type** — month-to-month customers churn far more than annual/biennial\n"
            "- **Tenure** — risk is highest in the first 12 months\n"
            "- **Internet service & security add-ons** — Fiber optic without OnlineSecurity/TechSupport "
            "churns more\n"
            "- **Monthly charges / charge-per-month** — higher recurring cost correlates with churn\n\n"
            "See the SHAP summary plot in the notebook for the exact ranked feature impacts "
            "behind this deployed model."
        )

st.divider()
st.caption(
    "Educational project using the IBM Telco Customer Churn dataset. "
    "Not a production system — thresholds and features were tuned on a single historical snapshot."
)