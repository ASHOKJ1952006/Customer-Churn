"""
shap_explainer.py — Convert SHAP values to natural-language explanations
Transforms SHAP feature importance into plain English for business stakeholders.
"""

import pandas as pd
import numpy as np

class SHAPToTextExplainer:
    """Convert SHAP values to natural language explanations."""
    
    def __init__(self):
        # Feature name mappings to human-readable descriptions
        self.feature_descriptions = {
            # Demographics
            'gender': 'Customer gender',
            'SeniorCitizen': 'Senior citizen status',
            'Partner': 'Having a partner',
            'Dependents': 'Having dependents',
            
            # Account details
            'tenure': 'How long they have been a customer',
            'PhoneService': 'Having phone service',
            'PaperlessBilling': 'Using paperless billing',
            'MonthlyCharges': 'Monthly bill amount',
            'TotalCharges': 'Total amount spent',
            
            # Contract & payment
            'Contract_Month-to-month': 'Month-to-month contract',
            'Contract_One year': 'One-year contract',
            'Contract_Two year': 'Two-year contract',
            'PaymentMethod_Electronic check': 'Paying by electronic check',
            'PaymentMethod_Mailed check': 'Paying by mailed check',
            'PaymentMethod_Credit card (automatic)': 'Automatic credit card payment',
            'PaymentMethod_Bank transfer (automatic)': 'Automatic bank transfer',
            
            # Services
            'MultipleLines_Yes': 'Having multiple phone lines',
            'MultipleLines_No phone service': 'No phone service',
            'InternetService_Fiber optic': 'Fiber optic internet',
            'InternetService_DSL': 'DSL internet',
            'InternetService_No': 'No internet service',
            'OnlineSecurity_Yes': 'Having online security',
            'OnlineSecurity_No internet service': 'No internet service',
            'OnlineBackup_Yes': 'Having online backup',
            'OnlineBackup_No internet service': 'No internet service',
            'DeviceProtection_Yes': 'Having device protection',
            'DeviceProtection_No internet service': 'No internet service',
            'TechSupport_Yes': 'Having tech support',
            'TechSupport_No internet service': 'No internet service',
            'StreamingTV_Yes': 'Having streaming TV',
            'StreamingTV_No internet service': 'No internet service',
            'StreamingMovies_Yes': 'Having streaming movies',
            'StreamingMovies_No internet service': 'No internet service',
            
            # Engineered features
            'AvgMonthlySpend': 'Average monthly spending',
            'ChargeIncrease': 'Recent bill increase',
            'NumServices': 'Number of services subscribed',
            'IsNewCustomer': 'Being a new customer',
            'HighRiskCombo': 'High-risk profile (month-to-month + electronic check)',
        }
        
        # Value interpretation templates
        self.positive_templates = [
            "{feature} increases churn risk",
            "Having {feature} makes them more likely to leave",
            "{feature} is a strong churn indicator",
            "Their {feature} contributes to higher churn probability",
        ]
        
        self.negative_templates = [
            "{feature} reduces churn risk",
            "Having {feature} makes them more likely to stay",
            "{feature} helps retain the customer",
            "Their {feature} contributes to lower churn probability",
        ]
        
        # Numerical feature interpretation
        self.numerical_ranges = {
            'tenure': {
                'low': (0, 6, 'new customer'),
                'medium': (6, 24, 'short-term customer'),
                'high': (24, 48, 'mid-term customer'),
                'very_high': (48, 100, 'long-term customer')
            },
            'MonthlyCharges': {
                'low': (0, 40, 'low monthly bill'),
                'medium': (40, 80, 'moderate monthly bill'),
                'high': (80, 120, 'high monthly bill'),
                'very_high': (120, 200, 'very high monthly bill')
            },
            'TotalCharges': {
                'low': (0, 500, 'low total spend'),
                'medium': (500, 2000, 'moderate total spend'),
                'high': (2000, 5000, 'high total spend'),
                'very_high': (5000, 100000, 'very high total spend')
            },
            'AvgMonthlySpend': {
                'low': (0, 40, 'low average monthly spend'),
                'medium': (40, 80, 'moderate average monthly spend'),
                'high': (80, 120, 'high average monthly spend'),
                'very_high': (120, 200, 'very high average monthly spend')
            },
            'NumServices': {
                'low': (0, 2, 'few services'),
                'medium': (2, 4, 'moderate number of services'),
                'high': (4, 6, 'many services'),
                'very_high': (6, 10, 'very many services')
            }
        }
    
    def get_feature_description(self, feature_name):
        """Get human-readable description for a feature."""
        return self.feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
    
    def interpret_numerical_value(self, feature_name, value):
        """Interpret a numerical feature value in context."""
        if feature_name not in self.numerical_ranges:
            return f"{value:.2f}"
        
        ranges = self.numerical_ranges[feature_name]
        for range_name, (low, high, description) in ranges.items():
            if low <= value < high:
                return description
        
        return f"{value:.2f}"
    
    def format_shap_value(self, shap_value):
        """Format SHAP value for readability."""
        abs_value = abs(shap_value)
        if abs_value < 0.1:
            return "slightly"
        elif abs_value < 0.3:
            return "moderately"
        elif abs_value < 0.5:
            return "significantly"
        else:
            return "very strongly"
    
    def generate_single_explanation(self, feature_name, shap_value, feature_value=None):
        """Generate a natural language explanation for a single feature."""
        description = self.get_feature_description(feature_name)
        intensity = self.format_shap_value(shap_value)
        
        if shap_value > 0:
            # Positive SHAP = increases churn risk
            if feature_name in self.numerical_ranges and feature_value is not None:
                value_desc = self.interpret_numerical_value(feature_name, feature_value)
                return f"Their {value_desc} ({description}) {intensity} increases churn risk."
            else:
                return f"{description.capitalize()} {intensity} increases churn risk."
        else:
            # Negative SHAP = decreases churn risk
            if feature_name in self.numerical_ranges and feature_value is not None:
                value_desc = self.interpret_numerical_value(feature_name, feature_value)
                return f"Their {value_desc} ({description}) {intensity} reduces churn risk."
            else:
                return f"{description.capitalize()} {intensity} reduces churn risk."
    
    def generate_explanation(self, shap_values, feature_names, feature_values=None, top_n=5):
        """
        Generate a comprehensive natural language explanation from SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            feature_values: Optional feature values for context
            top_n: Number of top features to include
            
        Returns:
            Natural language explanation string
        """
        # Create DataFrame for easier manipulation
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values
        })
        
        # Add feature values if provided
        if feature_values is not None:
            shap_df['feature_value'] = feature_values
        
        # Sort by absolute SHAP value
        shap_df['abs_shap'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        # Get top features
        top_features = shap_df.head(top_n)
        
        # Generate explanation
        explanations = []
        
        # Separate positive and negative factors
        positive_factors = top_features[top_features['shap_value'] > 0]
        negative_factors = top_features[top_features['shap_value'] < 0]
        
        # Build narrative
        if len(positive_factors) > 0:
            if len(positive_factors) == 1:
                explanations.append("The main factor increasing churn risk is:")
            else:
                explanations.append("The main factors increasing churn risk are:")
            
            for _, row in positive_factors.iterrows():
                feature_val = row['feature_value'] if 'feature_value' in row else None
                explanation = self.generate_single_explanation(
                    row['feature'], row['shap_value'], feature_val
                )
                explanations.append(f"• {explanation}")
        
        if len(negative_factors) > 0:
            if len(negative_factors) == 1:
                explanations.append("\nThe main factor reducing churn risk is:")
            else:
                explanations.append("\nThe main factors reducing churn risk are:")
            
            for _, row in negative_factors.iterrows():
                feature_val = row['feature_value'] if 'feature_value' in row else None
                explanation = self.generate_single_explanation(
                    row['feature'], row['shap_value'], feature_val
                )
                explanations.append(f"• {explanation}")
        
        # Add summary
        if len(positive_factors) > 0 and len(negative_factors) > 0:
            explanations.append("\nOverall, the customer's profile shows competing factors that influence their churn probability.")
        elif len(positive_factors) > 0:
            explanations.append("\nOverall, the customer's profile indicates elevated churn risk.")
        else:
            explanations.append("\nOverall, the customer's profile indicates lower churn risk.")
        
        return "\n".join(explanations)
    
    def generate_summary_explanation(self, shap_values, feature_names, churn_probability):
        """Generate a concise summary explanation."""
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values
        })
        shap_df['abs_shap'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        top_feature = shap_df.iloc[0]
        description = self.get_feature_description(top_feature['feature'])
        intensity = self.format_shap_value(top_feature['shap_value'])
        
        if churn_probability > 0.6:
            risk_level = "high"
        elif churn_probability > 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        if top_feature['shap_value'] > 0:
            return f"This customer has {risk_level} churn risk ({churn_probability:.1%}). Their {description} {intensity} increases their likelihood to leave."
        else:
            return f"This customer has {risk_level} churn risk ({churn_probability:.1%}). Their {description} {intensity} helps retain them."
