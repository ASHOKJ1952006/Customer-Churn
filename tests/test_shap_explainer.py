"""
test_shap_explainer.py — Test the natural language SHAP explanation system
"""

import numpy as np
import pandas as pd
from shap_explainer import SHAPToTextExplainer

# Initialize explainer
explainer = SHAPToTextExplainer()

# Test with sample SHAP values
print("=== Testing SHAP to Natural Language Explainer ===\n")

# Sample 1: High churn risk customer
shap_values_1 = np.array([0.5, -0.2, 0.3, 0.1, -0.05, 0.4, -0.1, 0.2])
feature_names_1 = ['Contract_Month-to-month', 'tenure', 'MonthlyCharges', 
                   'PaymentMethod_Electronic check', 'OnlineSecurity_Yes',
                   'InternetService_Fiber optic', 'Partner', 'TechSupport_Yes']
feature_values_1 = [1, 5, 85.5, 1, 0, 1, 0, 0]

print("Test 1: High churn risk customer")
print("SHAP values:", shap_values_1)
print("Feature names:", feature_names_1)
print("\nNatural Language Explanation:")
explanation_1 = explainer.generate_explanation(
    shap_values_1, feature_names_1, feature_values_1, top_n=3
)
print(explanation_1)
print("\n" + "="*80 + "\n")

# Sample 2: Low churn risk customer
shap_values_2 = np.array([-0.4, 0.1, -0.3, -0.2, 0.05, -0.1, 0.3, -0.2])
feature_names_2 = ['Contract_Two year', 'tenure', 'MonthlyCharges',
                   'PaymentMethod_Bank transfer (automatic)', 'OnlineSecurity_Yes',
                   'InternetService_DSL', 'Partner', 'TechSupport_Yes']
feature_values_2 = [1, 48, 65.2, 1, 1, 0, 1, 1]

print("Test 2: Low churn risk customer")
print("SHAP values:", shap_values_2)
print("Feature names:", feature_names_2)
print("\nNatural Language Explanation:")
explanation_2 = explainer.generate_explanation(
    shap_values_2, feature_names_2, feature_values_2, top_n=3
)
print(explanation_2)
print("\n" + "="*80 + "\n")

# Test summary explanation
print("Test 3: Summary explanation")
summary_1 = explainer.generate_summary_explanation(
    shap_values_1, feature_names_1, churn_probability=0.75
)
print("High risk summary:", summary_1)

summary_2 = explainer.generate_summary_explanation(
    shap_values_2, feature_names_2, churn_probability=0.15
)
print("Low risk summary:", summary_2)

print("\n=== All tests completed ===")
