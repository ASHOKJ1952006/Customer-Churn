import streamlit as st
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load models and preprocessing objects (Dataset 1 / Telco pipeline, Optuna-tuned) ---
model = joblib.load('ds1opt_churn_model.pkl')
scaler = joblib.load('ds1opt_scaler.pkl')
feature_columns = joblib.load('ds1opt_feature_columns.pkl')
best_threshold = joblib.load('ds1opt_best_threshold.pkl')

if hasattr(model, 'estimators_'):
    for est in model.estimators_:
        if isinstance(est, XGBClassifier):
            est.set_params(device='cpu')

shap_model = XGBClassifier()
shap_model.load_model('ds1opt_shap_model.json')
shap_model.set_params(device='cpu')
explainer = shap.TreeExplainer(shap_model)

st.title("📱 Telecom Customer Churn Predictor")

ACTION_RULES = {
    'Contract': 'Offer discount to upgrade to a longer-term contract for better lock-in.',
    'InternetService_Fiber optic': 'Investigate fiber service quality or offer a fiber plan discount.',
    'PaymentMethod_Electronic check': 'Encourage switch to auto-debit/UPI with a small incentive.',
    'PaperlessBilling': 'Offer a loyalty reward tied to billing preference.',
    'MonthlyCharges': 'Offer a bundled discount to reduce effective monthly cost.',
    'tenure': 'Assign to early-tenure onboarding/retention outreach program.',
    'OnlineSecurity': 'Bundle free online security add-on for 3 months.',
    'TechSupport': 'Proactively offer free tech support session.',
}

def top_factors_to_action(top_factors):
    for feature in top_factors:
        for key, action in ACTION_RULES.items():
            if key in feature:
                return action
    return 'Review customer profile for a tailored retention offer.'


def preprocess_raw_dataframe(raw_df):
    """Takes a raw dataframe in the ORIGINAL Telco dataset schema (same columns as
    WA_Fn-UseC_-Telco-Customer-Churn.csv, minus Churn) and returns the encoded,
    scaled feature matrix ready for the model, plus a customer identifier series."""
    df = raw_df.copy()

    customer_ids = df['customerID'].astype(str) if 'customerID' in df.columns else pd.Series(
        [f"row_{i}" for i in range(len(df))]
    )
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == object:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    present_multi = [c for c in multi_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=present_multi, drop_first=True)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_columns].astype(float)
    df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    return df_encoded, customer_ids


tab1, tab2 = st.tabs(["👤 Single Customer", "📋 Batch Upload"])

# =========================================================
# TAB 1: Single customer form (existing functionality)
# =========================================================
with tab1:
    st.write("Enter customer details to predict churn risk and get retention recommendations.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**More Services & Demographics**")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

    if st.button("Predict Churn Risk", type="primary"):
        raw = {
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
        }
        for col in feature_columns:
            if col not in raw:
                raw[col] = 0

        if multiple_lines == "Yes":
            raw['MultipleLines_Yes'] = 1
        elif multiple_lines == "No phone service":
            raw['MultipleLines_No phone service'] = 1

        if internet_service == "Fiber optic":
            raw['InternetService_Fiber optic'] = 1
        elif internet_service == "No":
            raw['InternetService_No'] = 1

        if online_security == "Yes":
            raw['OnlineSecurity_Yes'] = 1
        elif online_security == "No internet service":
            raw['OnlineSecurity_No internet service'] = 1

        if online_backup == "Yes":
            raw['OnlineBackup_Yes'] = 1
        elif online_backup == "No internet service":
            raw['OnlineBackup_No internet service'] = 1

        if device_protection == "Yes":
            raw['DeviceProtection_Yes'] = 1
        elif device_protection == "No internet service":
            raw['DeviceProtection_No internet service'] = 1

        if tech_support == "Yes":
            raw['TechSupport_Yes'] = 1
        elif tech_support == "No internet service":
            raw['TechSupport_No internet service'] = 1

        if streaming_tv == "Yes":
            raw['StreamingTV_Yes'] = 1
        elif streaming_tv == "No internet service":
            raw['StreamingTV_No internet service'] = 1

        if streaming_movies == "Yes":
            raw['StreamingMovies_Yes'] = 1
        elif streaming_movies == "No internet service":
            raw['StreamingMovies_No internet service'] = 1

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

        input_df = pd.DataFrame([raw])[feature_columns].astype(float)
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        prob = model.predict_proba(input_df)[0][1]
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

        st.info("💡 Recommended Action: " + top_factors_to_action(top_factors.index.tolist()))

        with st.expander("ℹ️ How this works"):
            st.write(
                "The prediction comes from a stacked ensemble (Random Forest + XGBoost + LightGBM, "
                "hyperparameters tuned via Optuna's TPE search) combined via a Logistic Regression "
                "meta-learner, trained with SMOTE class balancing and threshold-tuned to maximize F1 "
                "on the churn class."
            )

# =========================================================
# TAB 2: Batch CSV upload
# =========================================================
with tab2:
    st.write(
        "Upload a CSV of multiple customers to get churn risk predictions for all of them at once, "
        "ranked by risk, with a downloadable report."
    )

    st.markdown(
        "**Expected columns** (same as the Telco Customer Churn dataset): "
        "`customerID` (optional), `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, "
        "`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, "
        "`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, "
        "`PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`"
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type="csv")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(raw_df)} customers.")

            with st.spinner("Scoring customers..."):
                X_batch, customer_ids = preprocess_raw_dataframe(raw_df)
                probs = model.predict_proba(X_batch)[:, 1]
                shap_vals_batch = explainer.shap_values(X_batch)

                results = []
                for i in range(len(X_batch)):
                    row_shap = pd.Series(shap_vals_batch[i], index=feature_columns)
                    top3 = row_shap.sort_values(ascending=False).head(3).index.tolist()
                    prob = probs[i]
                    risk = "High" if prob > 0.6 else ("Medium" if prob > best_threshold else "Low")
                    results.append({
                        'Customer ID': customer_ids.iloc[i],
                        'Churn Probability': round(float(prob), 4),
                        'Risk Level': risk,
                        'Top Risk Factors': ", ".join(top3),
                        'Recommended Action': top_factors_to_action(top3),
                    })

                results_df = pd.DataFrame(results).sort_values('Churn Probability', ascending=False).reset_index(drop=True)

            st.success(f"Scored {len(results_df)} customers.")

            colX, colY, colZ = st.columns(3)
            colX.metric("High Risk", int((results_df['Risk Level'] == 'High').sum()))
            colY.metric("Medium Risk", int((results_df['Risk Level'] == 'Medium').sum()))
            colZ.metric("Low Risk", int((results_df['Risk Level'] == 'Low').sum()))

            st.dataframe(
                results_df.style.format({'Churn Probability': '{:.1%}'}),
                use_container_width=True,
                height=400
            )

            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Risk Report (CSV)",
                data=csv_bytes,
                file_name="churn_risk_report.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
            st.write("Make sure your CSV has the expected columns listed above.")