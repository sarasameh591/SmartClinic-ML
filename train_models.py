
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Input features and target
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
TARGET = "Outcome"

# Read dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place diabetes.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

# Validate columns
required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

df = df[required_columns].copy()

# Convert all columns to numeric
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows without a valid target
df = df.dropna(subset=[TARGET])

# The target must contain only 0 and 1
if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1.")

# Replace impossible zero measurements with missing values.
# Zero is valid for Pregnancies and is NOT replaced there.
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for column in zero_as_missing:
    df.loc[df[column] == 0, column] = np.nan

X = df[FEATURES]
y = df[TARGET].astype(int)

# Split data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Random Forest pipeline
rf_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    ),
])

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

models = {
    "RandomForest": rf_pipeline,
    "LogisticRegression": lr_pipeline,
}

results = {}
best_name = None
best_score = -1
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }

    results[name] = metrics

    print(f"\n{name} metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    # Select model based on F1-score
    if metrics["f1_score"] > best_score:
        best_score = metrics["f1_score"]
        best_name = name
        best_model = model

# Save selected model
model_path = MODEL_DIR / "diabetes_model.joblib"
joblib.dump(best_model, model_path)

# Save evaluation metrics
metrics_path = MODEL_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "selected_model": best_name,
            "selection_metric": "f1_score",
            "test_metrics": results,
            "features": FEATURES,
            "target": TARGET,
        },
        file,
        indent=4,
    )

print("\nTraining completed.")
print(f"Selected model: {best_name}")
print(f"Saved model to: {model_path}")
print(f"Saved metrics to: {metrics_path}")