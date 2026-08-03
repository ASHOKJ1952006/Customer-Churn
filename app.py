import streamlit as st
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier
from database import ChurnDatabase
from shap_explainer import SHAPToTextExplainer
import uuid

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Initialize database
db = ChurnDatabase()

# Initialize SHAP text explainer
shap_text_explainer = SHAPToTextExplainer()

# --- Load models and preprocessing objects (Dataset 1 / Telco, Calibrated - best recall: 0.810) ---
model = joblib.load('models/ds1calcv_churn_model.pkl')
scaler = joblib.load('models/ds1calcv_scaler.pkl')
feature_columns = joblib.load('models/ds1calcv_feature_columns.pkl')
best_threshold = joblib.load('models/ds1calcv_best_threshold.pkl')

# Force GPU-trained models to CPU for inference stability
def force_cpu_on_xgb(model):
    """Recursively force XGBClassifier to CPU, handling nested structures."""
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            force_cpu_on_xgb(est)
    if hasattr(model, 'named_steps'):
        for step_name, step in model.named_steps.items():
            force_cpu_on_xgb(step)
    if hasattr(model, 'calibrated_classifiers_'):
        for cal in model.calibrated_classifiers_:
            force_cpu_on_xgb(cal.estimator)
    if isinstance(model, XGBClassifier):
        model.set_params(device='cpu')

force_cpu_on_xgb(model)

shap_model = XGBClassifier()
shap_model.load_model('models/ds1fe_shap_model.json')
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
    """Not used for Optuna model (no feature engineering)"""
    return None


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
    
    # Add engineered features
    raw['AvgMonthlySpend'] = inputs['total_charges'] / max(inputs['tenure'], 1)
    raw['ChargeIncrease'] = 1 if inputs['monthly_charges'] > (inputs['total_charges'] / max(inputs['tenure'], 1)) else 0
    raw['NumServices'] = sum([
        1 if inputs['phone_service'] == "Yes" else 0,
        1 if inputs['online_security'] == "Yes" else 0,
        1 if inputs['online_backup'] == "Yes" else 0,
        1 if inputs['device_protection'] == "Yes" else 0,
        1 if inputs['tech_support'] == "Yes" else 0,
        1 if inputs['streaming_tv'] == "Yes" else 0,
        1 if inputs['streaming_movies'] == "Yes" else 0,
    ])
    raw['IsNewCustomer'] = 1 if inputs['tenure'] < 6 else 0
    raw['HighRiskCombo'] = 1 if (inputs['contract'] == "Month-to-month" and inputs['payment_method'] == "Electronic check") else 0
    
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

    input_df = pd.DataFrame([raw])[feature_columns].astype(float)
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])
    return input_df


def predict_probability(inputs):
    input_df = build_feature_vector(inputs)
    return model.predict_proba(input_df)[0][1]


def preprocess_raw_dataframe(raw_df):
    """Batch version with feature engineering for calibrated model."""
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
    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].dtype == object:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    present_multi = [c for c in multi_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=present_multi, drop_first=True)

    # Add engineered features
    df_encoded['AvgMonthlySpend'] = df_encoded['TotalCharges'] / df_encoded['tenure'].replace(0, 1)
    df_encoded['ChargeIncrease'] = (df_encoded['MonthlyCharges'] > df_encoded['AvgMonthlySpend']).astype(int)
    
    # NumServices: count of service subscriptions
    service_cols = ['PhoneService', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 
                    'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
    df_encoded['NumServices'] = 0
    for col in service_cols:
        if col in df_encoded.columns:
            df_encoded['NumServices'] += df_encoded[col]
    
    df_encoded['IsNewCustomer'] = (df_encoded['tenure'] < 6).astype(int)
    
    # HighRiskCombo: month-to-month + electronic check
    has_month_to_month = df_encoded['Contract_One year'] == 0
    has_electronic_check = df_encoded['PaymentMethod_Electronic check'] == 1 if 'PaymentMethod_Electronic check' in df_encoded.columns else False
    df_encoded['HighRiskCombo'] = (has_month_to_month & has_electronic_check).astype(int)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns].astype(float)
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']
    df_encoded[scale_cols] = scaler.transform(df_encoded[scale_cols])
    return df_encoded, customer_ids


def risk_label(prob):
    return "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > best_threshold else "🟢 Low")


tab1, tab2, tab3, tab4 = st.tabs(["👤 Single Customer", "📋 Batch Upload", "🔄 What-If Simulator", "📊 Dashboard"])

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
            
            # Natural language explanation
            st.divider()
            st.write("**📝 Plain English Explanation:**")
            input_df = build_feature_vector(inputs)
            feature_values = input_df.iloc[0].values
            nl_explanation = shap_text_explainer.generate_explanation(
                shap_vals, feature_columns, feature_values, top_n=3
            )
            st.write(nl_explanation)

        action = top_factors_to_action(top_factors.index.tolist())
        st.info("💡 Recommended Action: " + action)

        # Save to database
        customer_id = f"manual_{uuid.uuid4().hex[:8]}"
        db.add_single_prediction(
            customer_id=customer_id,
            churn_probability=prob,
            risk_level=risk_label(prob),
            top_factors=top_factors.index.tolist(),
            recommended_action=action,
            input_features=inputs
        )
        st.success("✅ Prediction saved to database")

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

            # Save to database
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"
            db_predictions = []
            for _, row in results_df.iterrows():
                db_predictions.append({
                    'customer_id': row['Customer ID'],
                    'churn_probability': row['Churn Probability'],
                    'risk_level': row['Risk Level'],
                    'top_factors': row['Top Risk Factors'].split(', '),
                    'recommended_action': row['Recommended Action']
                })
            db.add_batch_predictions(batch_id, db_predictions)
            st.success(f"✅ Batch saved to database (ID: {batch_id})")
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

# =========================================================
# TAB 4: Dashboard
# =========================================================
with tab4:
    st.write("📊 Prediction History & Risk Queue Dashboard")
    
    # Dashboard stats
    stats = db.get_dashboard_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Single Predictions", stats['total_single_predictions'])
    col2.metric("Total Batch Predictions", stats['total_batch_predictions'])
    col3.metric("Pending High-Risk", stats['pending_risks'])
    col4.metric("Recent High-Risk (7d)", stats['recent_high_risk_7days'])
    
    st.divider()
    
    # Risk Queue Section
    st.subheader("🔴 High-Risk Queue (Pending Follow-up)")
    
    risk_queue = db.get_risk_queue(status='pending', limit=50)
    
    if len(risk_queue) > 0:
        st.dataframe(
            risk_queue[['customer_id', 'churn_probability', 'top_factors', 'recommended_action', 'timestamp']],
            use_container_width=True
        )
        
        # Risk queue actions
        st.subheader("Risk Queue Actions")
        colA, colB = st.columns(2)
        
        with colA:
            customer_to_update = st.selectbox(
                "Select customer to update status",
                options=risk_queue['customer_id'].tolist()
            )
        
        with colB:
            new_status = st.selectbox(
                "New status",
                options=['contacted', 'resolved']
            )
        
        follow_up_notes = st.text_area("Follow-up notes (optional)")
        
        if st.button("Update Status", type="primary"):
            db.update_risk_queue_status(
                customer_id=customer_to_update,
                status=new_status,
                follow_up_notes=follow_up_notes if follow_up_notes else None
            )
            st.success(f"✅ Updated {customer_to_update} to {new_status}")
            st.rerun()
    else:
        st.info("No pending high-risk customers in queue.")
    
    st.divider()
    
    # Recent Predictions
    st.subheader("📋 Recent Single Predictions")
    
    recent_single = db.get_single_predictions(limit=20)
    if len(recent_single) > 0:
        st.dataframe(
            recent_single[['customer_id', 'churn_probability', 'risk_level', 'top_factors', 'timestamp']],
            use_container_width=True
        )
    else:
        st.info("No single predictions yet.")
    
    st.divider()
    
    # Batch History
    st.subheader("📦 Batch Upload History")
    
    batch_metadata = db.get_batch_metadata(limit=20)
    if len(batch_metadata) > 0:
        st.dataframe(
            batch_metadata,
            use_container_width=True
        )
        
        # View specific batch
        if len(batch_metadata) > 0:
            selected_batch = st.selectbox(
                "View details for batch:",
                options=batch_metadata['batch_id'].tolist()
            )
            
            batch_details = db.get_batch_predictions(batch_id=selected_batch)
            st.dataframe(
                batch_details,
                use_container_width=True
            )
    else:
        st.info("No batch uploads yet.")
    
    st.divider()
    
    # Data management
    st.subheader("🗄️ Data Management")
    
    if st.button("Clear data older than 30 days"):
        db.clear_old_data(days=30)
        st.success("✅ Old data cleared")
        st.rerun()