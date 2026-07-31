import streamlit as st
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load models and preprocessing objects (Dataset 1 / Telco, feature-engineered, primary) ---
model = joblib.load('ds1fe_churn_model.pkl')
scaler = joblib.load('ds1fe_scaler.pkl')
feature_columns = joblib.load('ds1fe_feature_columns.pkl')
best_threshold = joblib.load('ds1fe_best_threshold.pkl')

if hasattr(model, 'estimators_'):
    for est in model.estimators_:
        if isinstance(est, XGBClassifier):
            est.set_params(device='cpu')

shap_model = XGBClassifier()
shap_model.load_model('ds1fe_shap_model.json')
shap_model.set_params(device='cpu')
explainer = shap.TreeExplainer(shap_model)

st.title("📱 Telecom Customer Churn Predictor")

ACTION_RULES = {
    'Contract': 'Offer discount to upgrade to a longer-term contract for better lock-in.',
    'HighRiskCombo': 'High-risk profile (month-to-month + electronic check) — prioritize proactive outreach.',
    'InternetService_Fiber optic': 'Investigate fiber service quality or offer a fiber plan discount.',
    'PaymentMethod_Electronic check': 'Encourage switch to auto-debit/UPI with a small incentive.',
    'PaperlessBilling': 'Offer a loyalty reward tied to billing preference.',
    'ChargeIncrease': 'Recent bill increase detected — consider a loyalty discount to offset it.',
    'MonthlyCharges': 'Offer a bundled discount to reduce effective monthly cost.',
    'IsNewCustomer': 'New customer — assign to onboarding/early retention outreach program.',
    'tenure': 'Assign to early-tenure onboarding/retention outreach program.',
    'NumServices': 'Low service engagement — offer a bundle to increase platform investment.',
    'OnlineSecurity': 'Bundle free online security add-on for 3 months.',
    'TechSupport': 'Proactively offer free tech support session.',
}

def top_factors_to_action(top_factors):
    for feature in top_factors:
        for key, action in ACTION_RULES.items():
            if key in feature:
                return action
    return 'Review customer profile for a tailored retention offer.'


def compute_tenure_group(tenure):
    """Matches the pd.cut bins/labels used during training:
    bins=[-1,6,12,24,48,72], labels=['0-6mo','7-12mo','13-24mo','25-48mo','49-72mo']"""
    if tenure <= 6:
        return '0-6mo'
    elif tenure <= 12:
        return '7-12mo'
    elif tenure <= 24:
        return '13-24mo'
    elif tenure <= 48:
        return '25-48mo'
    else:
        return '49-72mo'


def add_engineered_features(raw, tenure, monthly_charges, total_charges, contract, payment_method,
                             online_security, online_backup, device_protection,
                             tech_support, streaming_tv, streaming_movies):
    """Computes the 6 engineered features and adds them (as raw dict entries, including
    the correct one-hot TenureGroup dummy) — used by single-customer, batch, and what-if paths."""
    avg_monthly_spend = total_charges / (tenure + 1)
    charge_increase = monthly_charges - avg_monthly_spend
    num_services = sum([
        online_security == "Yes", online_backup == "Yes", device_protection == "Yes",
        tech_support == "Yes", streaming_tv == "Yes", streaming_movies == "Yes"
    ])
    high_risk_combo = 1 if (contract == "Month-to-month" and payment_method == "Electronic check") else 0
    is_new_customer = 1 if tenure <= 3 else 0
    tenure_group = compute_tenure_group(tenure)

    raw['AvgMonthlySpend'] = avg_monthly_spend
    raw['ChargeIncrease'] = charge_increase
    raw['NumServices'] = num_services
    raw['HighRiskCombo'] = high_risk_combo
    raw['IsNewCustomer'] = is_new_customer

    # TenureGroup one-hot — '0-6mo' was the dropped baseline category during training
    tg_col = f'TenureGroup_{tenure_group}'
    if tg_col in feature_columns:
        raw[tg_col] = 1

    return raw


# =========================================================
# Shared: form rendering + encoding
# =========================================================

def render_customer_inputs(key_prefix):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{key_prefix}_tenure")
        monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0, key=f"{key_prefix}_mc")
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0, key=f"{key_prefix}_tc")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key=f"{key_prefix}_contract")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key=f"{key_prefix}_pm")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key=f"{key_prefix}_pb")

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key=f"{key_prefix}_phone")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key=f"{key_prefix}_ml")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key=f"{key_prefix}_isp")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key=f"{key_prefix}_os")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key=f"{key_prefix}_ob")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key=f"{key_prefix}_dp")

    with col3:
        st.markdown("**More Services & Demographics**")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key=f"{key_prefix}_ts")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key=f"{key_prefix}_stv")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key=f"{key_prefix}_smov")
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"{key_prefix}_gender")
        partner = st.selectbox("Has Partner", ["Yes", "No"], key=f"{key_prefix}_partner")
        dependents = st.selectbox("Has Dependents", ["Yes", "No"], key=f"{key_prefix}_dep")
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key=f"{key_prefix}_senior")

    return {
        'tenure': tenure, 'monthly_charges': monthly_charges, 'total_charges': total_charges,
        'contract': contract, 'payment_method': payment_method, 'paperless_billing': paperless_billing,
        'phone_service': phone_service, 'multiple_lines': multiple_lines, 'internet_service': internet_service,
        'online_security': online_security, 'online_backup': online_backup, 'device_protection': device_protection,
        'tech_support': tech_support, 'streaming_tv': streaming_tv, 'streaming_movies': streaming_movies,
        'gender': gender, 'partner': partner, 'dependents': dependents, 'senior_citizen': senior_citizen,
    }


def build_feature_vector(inputs):
    raw = {
        'gender': 1 if inputs['gender'] == "Male" else 0,
        'SeniorCitizen': 1 if inputs['senior_citizen'] == "Yes" else 0,
        'Partner': 1 if inputs['partner'] == "Yes" else 0,
        'Dependents': 1 if inputs['dependents'] == "Yes" else 0,
        'tenure': inputs['tenure'],
        'PhoneService': 1 if inputs['phone_service'] == "Yes" else 0,
        'PaperlessBilling': 1 if inputs['paperless_billing'] == "Yes" else 0,
        'MonthlyCharges': inputs['monthly_charges'],
        'TotalCharges': inputs['total_charges'],
    }
    for col in feature_columns:
        if col not in raw:
            raw[col] = 0

    if inputs['multiple_lines'] == "Yes":
        raw['MultipleLines_Yes'] = 1
    elif inputs['multiple_lines'] == "No phone service":
        raw['MultipleLines_No phone service'] = 1

    if inputs['internet_service'] == "Fiber optic":
        raw['InternetService_Fiber optic'] = 1
    elif inputs['internet_service'] == "No":
        raw['InternetService_No'] = 1

    if inputs['online_security'] == "Yes":
        raw['OnlineSecurity_Yes'] = 1
    elif inputs['online_security'] == "No internet service":
        raw['OnlineSecurity_No internet service'] = 1

    if inputs['online_backup'] == "Yes":
        raw['OnlineBackup_Yes'] = 1
    elif inputs['online_backup'] == "No internet service":
        raw['OnlineBackup_No internet service'] = 1

    if inputs['device_protection'] == "Yes":
        raw['DeviceProtection_Yes'] = 1
    elif inputs['device_protection'] == "No internet service":
        raw['DeviceProtection_No internet service'] = 1

    if inputs['tech_support'] == "Yes":
        raw['TechSupport_Yes'] = 1
    elif inputs['tech_support'] == "No internet service":
        raw['TechSupport_No internet service'] = 1

    if inputs['streaming_tv'] == "Yes":
        raw['StreamingTV_Yes'] = 1
    elif inputs['streaming_tv'] == "No internet service":
        raw['StreamingTV_No internet service'] = 1

    if inputs['streaming_movies'] == "Yes":
        raw['StreamingMovies_Yes'] = 1
    elif inputs['streaming_movies'] == "No internet service":
        raw['StreamingMovies_No internet service'] = 1

    if inputs['contract'] == "One year":
        raw['Contract_One year'] = 1
    elif inputs['contract'] == "Two year":
        raw['Contract_Two year'] = 1

    if inputs['payment_method'] == "Electronic check":
        raw['PaymentMethod_Electronic check'] = 1
    elif inputs['payment_method'] == "Mailed check":
        raw['PaymentMethod_Mailed check'] = 1
    elif inputs['payment_method'] == "Credit card (automatic)":
        raw['PaymentMethod_Credit card (automatic)'] = 1

    # --- Engineered features (must match training exactly) ---
    raw = add_engineered_features(
        raw, inputs['tenure'], inputs['monthly_charges'], inputs['total_charges'],
        inputs['contract'], inputs['payment_method'],
        inputs['online_security'], inputs['online_backup'], inputs['device_protection'],
        inputs['tech_support'], inputs['streaming_tv'], inputs['streaming_movies']
    )

    input_df = pd.DataFrame([raw])[feature_columns].astype(float)
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])
    return input_df


def predict_probability(inputs):
    input_df = build_feature_vector(inputs)
    return model.predict_proba(input_df)[0][1]


def preprocess_raw_dataframe(raw_df):
    """Batch version — same engineered features computed row-wise."""
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

    # Engineered features, computed row-wise
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
    df['TenureGroup'] = df['tenure'].apply(compute_tenure_group)
    df['HighRiskCombo'] = ((df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].dtype == object:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod', 'TenureGroup']
    present_multi = [c for c in multi_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=present_multi, drop_first=True)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns].astype(float)
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']
    df_encoded[scale_cols] = scaler.transform(df_encoded[scale_cols])
    return df_encoded, customer_ids


def risk_label(prob):
    return "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > best_threshold else "🟢 Low")


tab1, tab2, tab3 = st.tabs(["👤 Single Customer", "📋 Batch Upload", "🔄 What-If Simulator"])

# =========================================================
# TAB 1: Single customer form
# =========================================================
with tab1:
    st.write("Enter customer details to predict churn risk and get retention recommendations.")
    inputs = render_customer_inputs(key_prefix="t1")

    if st.button("Predict Churn Risk", type="primary"):
        prob = predict_probability(inputs)
        input_df = build_feature_vector(inputs)
        shap_vals = explainer.shap_values(input_df)[0]

        st.divider()
        colA, colB = st.columns([1, 2])

        with colA:
            st.metric("Churn Probability", f"{prob:.1%}")
            st.metric("Risk Level", risk_label(prob))
            st.caption(f"Decision threshold: {best_threshold:.3f}")

        with colB:
            shap_series = pd.Series(shap_vals, index=feature_columns)
            top_factors = shap_series.sort_values(ascending=False).head(5)
            st.write("**Top factors influencing this prediction:**")
            st.bar_chart(top_factors)

        st.info("💡 Recommended Action: " + top_factors_to_action(top_factors.index.tolist()))

# =========================================================
# TAB 2: Batch CSV upload
# =========================================================
with tab2:
    st.write(
        "Upload a CSV of multiple customers to get churn risk predictions for all of them at once, "
        "ranked by risk, with a downloadable report."
    )
    st.markdown(
        "**Expected columns**: `customerID` (optional), `gender`, `SeniorCitizen`, `Partner`, "
        "`Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, "
        "`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, "
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
                use_container_width=True, height=400
            )

            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Risk Report (CSV)", data=csv_bytes,
                file_name="churn_risk_report.csv", mime="text/csv", type="primary"
            )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
            st.write("Make sure your CSV has the expected columns listed above.")

# =========================================================
# TAB 3: What-If Simulator
# =========================================================
with tab3:
    st.write(
        "Set up a customer profile, then explore how changing ONE factor would affect their "
        "churn risk — holding everything else constant."
    )

    inputs = render_customer_inputs(key_prefix="t3")

    baseline_prob = predict_probability(inputs)
    st.divider()
    st.metric("Current Churn Probability", f"{baseline_prob:.1%}", help=f"Risk level: {risk_label(baseline_prob)}")

    st.markdown("### 🔍 Explore a factor")
    factor = st.selectbox(
        "Which factor do you want to simulate changes for?",
        ["Contract Type", "Payment Method", "Internet Service", "Online Security",
         "Tech Support", "Tenure", "Monthly Charges"],
        key="whatif_factor"
    )

    if st.button("Run Simulation", type="primary"):
        scenarios = []

        if factor == "Contract Type":
            for val in ["Month-to-month", "One year", "Two year"]:
                modified = dict(inputs); modified['contract'] = val
                scenarios.append((val, predict_probability(modified)))
        elif factor == "Payment Method":
            for val in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]:
                modified = dict(inputs); modified['payment_method'] = val
                scenarios.append((val, predict_probability(modified)))
        elif factor == "Internet Service":
            for val in ["DSL", "Fiber optic", "No"]:
                modified = dict(inputs); modified['internet_service'] = val
                scenarios.append((val, predict_probability(modified)))
        elif factor == "Online Security":
            for val in ["Yes", "No"]:
                modified = dict(inputs); modified['online_security'] = val
                scenarios.append((val, predict_probability(modified)))
        elif factor == "Tech Support":
            for val in ["Yes", "No"]:
                modified = dict(inputs); modified['tech_support'] = val
                scenarios.append((val, predict_probability(modified)))
        elif factor == "Tenure":
            for val in [0, 6, 12, 24, 36, 48, 60, 72]:
                modified = dict(inputs); modified['tenure'] = val
                scenarios.append((f"{val} mo", predict_probability(modified)))
        elif factor == "Monthly Charges":
            for val in [20, 40, 60, 80, 100, 120]:
                modified = dict(inputs); modified['monthly_charges'] = float(val)
                scenarios.append((f"${val}", predict_probability(modified)))

        scenario_df = pd.DataFrame(scenarios, columns=[factor, 'Churn Probability']).set_index(factor)

        st.write(f"**Churn probability if {factor.lower()} changes (everything else held constant):**")
        st.bar_chart(scenario_df)

        best_scenario = scenario_df['Churn Probability'].idxmin()
        best_prob = scenario_df['Churn Probability'].min()
        reduction = baseline_prob - best_prob

        if reduction > 0.01:
            st.success(
                f"💡 Best option: **{best_scenario}** would lower churn probability to "
                f"**{best_prob:.1%}** — a reduction of **{reduction:.1%}** from the current {baseline_prob:.1%}."
            )
        else:
            st.info(f"Changing {factor.lower()} alone doesn't meaningfully change this customer's risk — other factors dominate.")