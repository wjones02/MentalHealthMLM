"""
Prediction script for mental health diagnosis outcome prediction.

This script loads the trained model and makes predictions on new patient data.
"""

import os
import joblib
import pandas as pd
from datetime import datetime

# Paths
MODEL_PATH = "models/best_model.pkl"


def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    return model


def create_sample_patient():
    """Create a sample patient for demonstration."""
    sample_patient = {
        'Patient ID': 999,
        'Age': 35,
        'Gender': 'Female',
        'Diagnosis': 'Major Depressive Disorder',
        'Symptom Severity (1-10)': 7,
        'Mood Score (1-10)': 4,
        'Sleep Quality (1-10)': 5,
        'Physical Activity (hrs/week)': 3,
        'Medication': 'SSRIs',
        'Therapy Type': 'Cognitive Behavioral Therapy',
        'Treatment Start Date': '2024-01-15',
        'Treatment Duration (weeks)': 12,
        'Stress Level (1-10)': 8,
        'Outcome': 'Unknown',  # Placeholder - this is what we want to predict
        'Treatment Progress (1-10)': 6,
        'AI-Detected Emotional State': 'Anxious',
        'Adherence to Treatment (%)': 75
    }
    return pd.DataFrame([sample_patient])


def preprocess_for_prediction(df):
    """Preprocess data for prediction (simplified version)."""
    from train import preprocess_data

    # Apply the same preprocessing as training
    X, _, preprocessor = preprocess_data(df)
    return X


def make_prediction(patient_data):
    """Make prediction for a patient."""
    model = load_model()

    # Preprocess the data
    X = preprocess_for_prediction(patient_data)

    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Get class names
    classes = model.classes_

    return prediction, probabilities, classes


def main():
    """Main prediction function."""
    print("="*60)
    print("MENTAL HEALTH OUTCOME PREDICTION")
    print("="*60)

    try:
        # Create sample patient
        print("\nCreating sample patient data...")
        patient_df = create_sample_patient()

        print("Sample Patient:")
        for col, val in patient_df.iloc[0].items():
            print(f"  {col}: {val}")

        # Make prediction
        print("\nMaking prediction...")
        prediction, probabilities, classes = make_prediction(patient_df)

        print(f"\nPredicted Outcome: {prediction}")
        print("\nPrediction Probabilities:")
        for class_name, prob in zip(classes, probabilities):
            print(f"  {class_name}: {prob:.3f} ({prob*100:.1f}%)")

        print("\n" + "="*60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()