"""Training script for mental health diagnosis outcome prediction.

This script loads data, preprocesses features, performs feature engineering,
trains a classification model with hyperparameter tuning, evaluates the model,
and saves the best-performing model to disk.
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/raw/mental_health_diagnosis_treatment_.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Clean and preprocess the dataset.

    Parameters
    ----------
    df: pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    Tuple containing features, target and preprocessing transformer.
    """
    df = df.copy()

    # Parse dates
    df["Treatment Start Date"] = pd.to_datetime(df["Treatment Start Date"], errors="coerce")
    df["treatment_start_month"] = df["Treatment Start Date"].dt.month

    # Feature engineering
    df["adherence_rate"] = df["Adherence to Treatment (%)"] / 100.0
    df["symptom_mood_interaction"] = (
        df["Symptom Severity (1-10)"] * df["Mood Score (1-10)"]
    )
    df["stress_sleep_ratio"] = df["Stress Level (1-10)"] / (df["Sleep Quality (1-10)"] + 1)

    # Drop unused columns
    df = df.drop(columns=["Patient ID", "Treatment Start Date"])

    # Separate target
    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X, y, preprocessor


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Split dataset into train and test sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_model(X_train, y_train, preprocessor: ColumnTransformer) -> RandomizedSearchCV:
    """Train model using RandomizedSearchCV for hyperparameter tuning."""
    clf = RandomForestClassifier(random_state=42)

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
    )

    search.fit(X_train, y_train)
    return search


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """Evaluate the model and return classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        metrics["roc_auc"] = roc_auc_score(
            y_test, y_proba, multi_class="ovo", average="weighted"
        )

    return metrics


def save_model(model: Pipeline, path: str = MODEL_PATH) -> None:
    """Persist the model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    """Run the full training pipeline."""
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    search = train_model(X_train, y_train, preprocessor)
    metrics = evaluate_model(search, X_test, y_test)
    save_model(search.best_estimator_)

    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"Best model saved to {MODEL_PATH}")
    print("Best parameters:", search.best_params_)


if __name__ == "__main__":
    main()
