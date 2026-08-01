"""
risk_model.py
--------------
Trains an XGBoost binary classifier to predict patient health risk from
structured clinical features, and uses SHAP to explain individual predictions.

This is the "1. Risk Model" + "2. Explainability" stages of the pipeline.
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLUMNS = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "glucose", "cholesterol", "smoker",
    "exercise_hours_per_week", "family_history",
]
TARGET_COLUMN = "high_risk"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_patients.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "risk_model.pkl")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shap_explainer.pkl")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # With a small demo dataset we skip a held-out test split when there are
    # too few samples per class; in production, always use train/test/CV.
    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, preds))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, probs))
    except ValueError:
        print("ROC-AUC: undefined (only one class present in test split)")

    return model


def build_explainer(model, df: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_patient(model, explainer, df: pd.DataFrame, patient_id: str) -> dict:
    """Return the top contributing risk factors for a single patient."""
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise ValueError(f"Patient {patient_id} not found")

    X_row = row[FEATURE_COLUMNS]
    shap_values = explainer.shap_values(X_row)

    contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    risk_score = float(model.predict_proba(X_row)[0, 1])

    return {
        "patient_id": patient_id,
        "risk_score": round(risk_score, 3),
        "top_factors": [
            {"feature": f, "impact": round(float(v), 3)} for f, v in contributions[:5]
        ],
    }


def save_artifacts(model, explainer):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved explainer -> {EXPLAINER_PATH}")


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    explainer = build_explainer(model, df)
    save_artifacts(model, explainer)

    # Demo: explain a couple of patients
    for pid in ["P0001", "P0003", "P0011"]:
        result = explain_patient(model, explainer, df, pid)
        print(f"\nPatient {pid} — risk score: {result['risk_score']}")
        for factor in result["top_factors"]:
            direction = "increases" if factor["impact"] > 0 else "decreases"
            print(f"  {factor['feature']:>26s}: {direction} risk (impact={factor['impact']})")
