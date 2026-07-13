import streamlit as st
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load models and preprocessing objects ---
model = joblib.load('churn_model.pkl')                  # stacked ensemble — used for the actual prediction
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
best_threshold = joblib.load('best_threshold.pkl')      # tuned threshold (~0.304), not hardcoded

shap_model = XGBClassifier()
shap_model.load_model('churn_shap_model.json')           # separate tuned XGBoost — used only for SHAP explanations
explainer = shap.TreeExplainer(shap_model)

st.title("📱 Telecom Customer Churn Predictor")
st.write("Enter customer details to predict churn risk and get retention recommendations.")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict Churn Risk", type="primary"):
    # Build a raw input row matching the original preprocessing
    raw = {
        'gender': 0, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0, 'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure, 'PhoneService': 1, 'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    }
    for col in feature_columns:
        if col not in raw:
            raw[col] = 0

    if internet_service == "Fiber optic":
        raw['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        raw['InternetService_No'] = 1
    if contract == "One year":
        raw['Contract_One year'] = 1
    elif contract == "Two year":
        raw['Contract_Two year'] = 1
    if payment_method == "Electronic check":
        raw['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        raw['PaymentMethod_Mailed check'] = 1
    elif payment_method == "Credit card (automatic)":
        raw['PaymentMethod_Credit card (automatic)'] = 1
    if online_security == "Yes":
        raw['OnlineSecurity_Yes'] = 1
    elif online_security == "No internet service":
        raw['OnlineSecurity_No internet service'] = 1
    if tech_support == "Yes":
        raw['TechSupport_Yes'] = 1
    elif tech_support == "No internet service":
        raw['TechSupport_No internet service'] = 1

    input_df = pd.DataFrame([raw])[feature_columns].astype(float)
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Prediction comes from the stacked ensemble
    prob = model.predict_proba(input_df)[0][1]

    # Explanation comes from the separate tuned XGBoost (native format, SHAP-compatible)
    shap_vals = explainer.shap_values(input_df)[0]

    st.divider()
    colA, colB = st.columns([1, 2])

    with colA:
        st.metric("Churn Probability", f"{prob:.1%}")
        risk = "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > best_threshold else "🟢 Low")
        st.metric("Risk Level", risk)
        st.caption(f"Decision threshold: {best_threshold:.3f}")

    with colB:
        shap_series = pd.Series(shap_vals, index=feature_columns)
        top_factors = shap_series.sort_values(ascending=False).head(5)
        st.write("**Top factors influencing this prediction:**")
        st.bar_chart(top_factors)

    st.info("💡 Recommended Action: " + (
        "Offer contract upgrade incentive and review payment method." if prob > best_threshold
        else "No action needed — customer profile is stable."
    ))