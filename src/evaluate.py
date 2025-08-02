"""
Model evaluation script for mental health diagnosis outcome prediction.

This script loads the trained model and evaluates its performance on test data,
providing detailed metrics and analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from train import load_data, preprocess_data, split_data

# Paths
MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/raw/mental_health_diagnosis_treatment_.csv"


def load_trained_model(model_path: str = MODEL_PATH):
    """Load the trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance and return detailed metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # ROC AUC for multiclass
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovo', average='weighted')
        except ValueError:
            metrics['roc_auc'] = 0.0

    return metrics, y_pred, y_proba


def print_detailed_evaluation(metrics, y_test, y_pred):
    """Print detailed evaluation results."""
    print("="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Class distribution
    print(f"\nTest Set Class Distribution:")
    class_counts = pd.Series(y_test).value_counts().sort_index()
    for class_name, count in class_counts.items():
        percentage = (count / len(y_test)) * 100
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")


def main():
    """Main evaluation function."""
    try:
        print("Loading and preprocessing data...")
        df = load_data(DATA_PATH)
        X, y, preprocessor = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        print("Loading trained model...")
        model = load_trained_model()

        print("Evaluating model performance...")
        metrics, y_pred, y_proba = evaluate_model_performance(model, X_test, y_test)

        print_detailed_evaluation(metrics, y_test, y_pred)

        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()