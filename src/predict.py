"""
Prediction utilities for single-customer churn prediction.
"""

import numpy as np
from src.data_preprocessing import preprocess_single_input


def predict_churn(model, preprocessor, feature_cols: list, input_dict: dict) -> dict:
    """
    Predict churn for a single customer.
    Returns a dict with prediction, probabilities, risk category, and explanation.
    """
    X = preprocess_single_input(input_dict, preprocessor, feature_cols)

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        churn_prob = float(proba[1])
        retain_prob = float(proba[0])
    else:
        # Decision function fallback
        churn_prob = 1.0 if prediction == 1 else 0.0
        retain_prob = 1.0 - churn_prob

    # Risk category
    if churn_prob < 0.30:
        risk = "Low Risk"
    elif churn_prob < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # Simple explanation
    explanation = _build_explanation(input_dict, churn_prob)

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(churn_prob * 100, 1),
        "retain_probability": round(retain_prob * 100, 1),
        "risk_category": risk,
        "explanation": explanation
    }


def _build_explanation(input_dict: dict, churn_prob: float) -> str:
    """Generate a short human-readable explanation."""
    factors = []

    contract = input_dict.get("Contract", "")
    if contract == "Month-to-month":
        factors.append("month-to-month contract (higher churn risk)")

    tenure = input_dict.get("tenure", None)
    if tenure is not None and float(tenure) < 12:
        factors.append("short tenure (less than 1 year)")

    monthly = input_dict.get("MonthlyCharges", None)
    if monthly is not None and float(monthly) > 70:
        factors.append("high monthly charges")

    internet = input_dict.get("InternetService", "")
    if internet == "Fiber optic":
        factors.append("fiber optic internet service")

    security = input_dict.get("OnlineSecurity", "")
    if security == "No":
        factors.append("no online security add-on")

    if factors:
        return f"Key factors influencing this prediction: {', '.join(factors)}."
    elif churn_prob < 0.30:
        return "This customer shows strong retention indicators such as a long tenure or a long-term contract."
    else:
        return "The model detected a combination of features associated with customer churn."
