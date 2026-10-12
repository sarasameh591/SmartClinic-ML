
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Basic input validation ranges for this application.
# These are validation limits, not medical diagnostic criteria.
LIMITS = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 500),
    "BloodPressure": (1, 300),
    "SkinThickness": (0, 150),
    "Insulin": (0, 3000),
    "BMI": (1, 100),
    "DiabetesPedigreeFunction": (0, 10),
    "Age": (1, 120),
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    # Only load a model file that you trust.
    return joblib.load(MODEL_PATH)


def predict_diabetes(model, features: dict) -> dict:
    missing = [
        name for name in FEATURES
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {', '.join(missing)}"
        )

    values = {}

    for name in FEATURES:
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be a valid number."
            )

        low, high = LIMITS[name]

        if not low <= value <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}."
            )

        values[name] = value

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_df)[0])
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Illustrative categories only; not medically validated.
    if probability < 0.30:
        risk_band = "Low"
    elif probability < 0.60:
        risk_band = "Medium"
    else:
        risk_band = "High"

    return {
        "prediction": prediction,
        "prediction_label": (
            "Diabetes predicted"
            if prediction == 1
            else "No diabetes predicted"
        ),
        "probability": round(probability, 4),
        "risk_band": risk_band,
        "disclaimer": (
            "Educational model output only. "
            "This result is not a medical diagnosis."
        ),
    }