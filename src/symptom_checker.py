"""
Interactive Mental Health Symptom Checker

This tool allows users to input their symptoms and provides:
1. Calculated mood and stress scores based on symptoms
2. Potential mental health diagnosis suggestions
3. Treatment outcome predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib
import os

# Symptom to score mapping
SYMPTOM_MAPPINGS = {
    # Mood-related symptoms (affect mood score)
    'feeling_sad': {'mood': -2, 'stress': 1},
    'feeling_hopeless': {'mood': -3, 'stress': 2},
    'loss_of_interest': {'mood': -2, 'stress': 1},
    'feeling_empty': {'mood': -2, 'stress': 1},
    'crying_spells': {'mood': -2, 'stress': 2},
    'feeling_worthless': {'mood': -3, 'stress': 2},
    'guilt': {'mood': -2, 'stress': 2},
    'irritability': {'mood': -1, 'stress': 3},
    'mood_swings': {'mood': -2, 'stress': 2},
    'feeling_euphoric': {'mood': 2, 'stress': 1},
    'feeling_energetic': {'mood': 1, 'stress': -1},
    
    # Anxiety/stress symptoms (affect stress score more)
    'excessive_worry': {'mood': -1, 'stress': 3},
    'panic_attacks': {'mood': -2, 'stress': 4},
    'fear_of_crowds': {'mood': -1, 'stress': 3},
    'restlessness': {'mood': -1, 'stress': 3},
    'feeling_tense': {'mood': -1, 'stress': 3},
    'racing_thoughts': {'mood': -1, 'stress': 3},
    'difficulty_concentrating': {'mood': -1, 'stress': 2},
    'fear_of_losing_control': {'mood': -2, 'stress': 4},
    'physical_symptoms_anxiety': {'mood': -1, 'stress': 3},
    
    # Sleep and physical symptoms
    'insomnia': {'mood': -1, 'stress': 2},
    'sleeping_too_much': {'mood': -2, 'stress': 1},
    'fatigue': {'mood': -2, 'stress': 1},
    'appetite_changes': {'mood': -1, 'stress': 1},
    'weight_changes': {'mood': -1, 'stress': 1},
    'headaches': {'mood': -1, 'stress': 2},
    'muscle_tension': {'mood': -1, 'stress': 2},
    
    # Behavioral symptoms
    'social_withdrawal': {'mood': -2, 'stress': 1},
    'avoiding_activities': {'mood': -2, 'stress': 2},
    'substance_use': {'mood': -2, 'stress': 2},
    'self_harm_thoughts': {'mood': -4, 'stress': 3},
    'risky_behavior': {'mood': -1, 'stress': 2},
}

# Diagnosis patterns based on symptom combinations
DIAGNOSIS_PATTERNS = {
    'Major Depressive Disorder': {
        'required_symptoms': ['feeling_sad', 'loss_of_interest'],
        'supporting_symptoms': ['feeling_hopeless', 'fatigue', 'sleeping_too_much', 'feeling_worthless', 'guilt'],
        'mood_range': (1, 4),
        'stress_range': (5, 8)
    },
    'Generalized Anxiety': {
        'required_symptoms': ['excessive_worry'],
        'supporting_symptoms': ['restlessness', 'feeling_tense', 'difficulty_concentrating', 'irritability', 'muscle_tension'],
        'mood_range': (3, 6),
        'stress_range': (7, 10)
    },
    'Panic Disorder': {
        'required_symptoms': ['panic_attacks'],
        'supporting_symptoms': ['fear_of_losing_control', 'physical_symptoms_anxiety', 'avoiding_activities'],
        'mood_range': (2, 5),
        'stress_range': (8, 10)
    },
    'Bipolar Disorder': {
        'required_symptoms': ['mood_swings'],
        'supporting_symptoms': ['feeling_euphoric', 'feeling_energetic', 'risky_behavior', 'feeling_sad'],
        'mood_range': (2, 8),  # Can be high or low
        'stress_range': (4, 9)
    }
}


def display_symptom_options():
    """Display available symptoms for user selection."""
    print("\n" + "="*60)
    print("MENTAL HEALTH SYMPTOM CHECKER")
    print("="*60)
    print("\nPlease select the symptoms you've been experiencing:")
    print("(Enter the numbers separated by commas, e.g., 1,3,5)")
    print()
    
    symptoms = [
        "Feeling sad or down most of the time",
        "Feeling hopeless about the future", 
        "Loss of interest in activities you used to enjoy",
        "Feeling empty inside",
        "Frequent crying spells",
        "Feeling worthless or guilty",
        "Excessive guilt about past events",
        "Irritability or anger",
        "Extreme mood swings",
        "Feeling unusually euphoric or 'high'",
        "Feeling unusually energetic",
        "Excessive worry about many things",
        "Panic attacks (sudden intense fear)",
        "Fear of crowds or social situations",
        "Feeling restless or on edge",
        "Feeling tense or unable to relax",
        "Racing thoughts",
        "Difficulty concentrating",
        "Fear of losing control",
        "Physical symptoms of anxiety (sweating, heart racing)",
        "Trouble falling or staying asleep",
        "Sleeping much more than usual",
        "Constant fatigue or low energy",
        "Significant changes in appetite",
        "Unexplained weight gain or loss",
        "Frequent headaches",
        "Muscle tension or aches",
        "Withdrawing from friends and family",
        "Avoiding activities or places",
        "Using alcohol or drugs to cope",
        "Thoughts of self-harm",
        "Engaging in risky or reckless behavior"
    ]
    
    symptom_keys = list(SYMPTOM_MAPPINGS.keys())
    
    for i, (key, description) in enumerate(zip(symptom_keys, symptoms), 1):
        print(f"{i:2d}. {description}")
    
    return symptom_keys


def get_user_symptoms() -> List[str]:
    """Get symptom selection from user."""
    symptom_keys = display_symptom_options()
    
    while True:
        try:
            user_input = input(f"\nEnter symptom numbers (1-{len(symptom_keys)}): ").strip()
            if not user_input:
                print("Please enter at least one symptom number.")
                continue
                
            selected_numbers = [int(x.strip()) for x in user_input.split(',')]
            
            # Validate numbers
            if all(1 <= num <= len(symptom_keys) for num in selected_numbers):
                selected_symptoms = [symptom_keys[num-1] for num in selected_numbers]
                return selected_symptoms
            else:
                print(f"Please enter numbers between 1 and {len(symptom_keys)}")
                
        except ValueError:
            print("Please enter valid numbers separated by commas.")


def calculate_scores(symptoms: List[str]) -> Tuple[int, int, int]:
    """Calculate mood score, stress level, and symptom severity based on symptoms."""
    base_mood = 7  # Start with neutral mood
    base_stress = 3  # Start with low stress
    
    mood_adjustment = 0
    stress_adjustment = 0
    
    for symptom in symptoms:
        if symptom in SYMPTOM_MAPPINGS:
            mood_adjustment += SYMPTOM_MAPPINGS[symptom]['mood']
            stress_adjustment += SYMPTOM_MAPPINGS[symptom]['stress']
    
    # Calculate final scores (1-10 scale)
    mood_score = max(1, min(10, base_mood + mood_adjustment))
    stress_level = max(1, min(10, base_stress + stress_adjustment))
    symptom_severity = max(1, min(10, len(symptoms) + stress_adjustment // 2))
    
    return mood_score, stress_level, symptom_severity


def suggest_diagnosis(symptoms: List[str], mood_score: int, stress_level: int) -> List[Tuple[str, float]]:
    """Suggest potential diagnoses based on symptoms and scores."""
    suggestions = []
    
    for diagnosis, pattern in DIAGNOSIS_PATTERNS.items():
        score = 0
        max_score = 0
        
        # Check required symptoms
        required_present = sum(1 for req in pattern['required_symptoms'] if req in symptoms)
        required_total = len(pattern['required_symptoms'])
        max_score += required_total * 3  # Required symptoms worth 3 points each
        score += required_present * 3
        
        # Check supporting symptoms
        supporting_present = sum(1 for sup in pattern['supporting_symptoms'] if sup in symptoms)
        supporting_total = len(pattern['supporting_symptoms'])
        max_score += supporting_total  # Supporting symptoms worth 1 point each
        score += supporting_present
        
        # Check mood and stress ranges
        mood_min, mood_max = pattern['mood_range']
        stress_min, stress_max = pattern['stress_range']
        
        max_score += 2  # Range checks worth 2 points total
        if mood_min <= mood_score <= mood_max:
            score += 1
        if stress_min <= stress_level <= stress_max:
            score += 1
        
        # Calculate confidence percentage
        if max_score > 0:
            confidence = (score / max_score) * 100
            if confidence > 20:  # Only suggest if confidence > 20%
                suggestions.append((diagnosis, confidence))
    
    # Sort by confidence
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions


def create_patient_profile(symptoms: List[str], mood_score: int, stress_level: int, 
                         symptom_severity: int, age: int, gender: str) -> pd.DataFrame:
    """Create a patient profile for prediction."""
    # Use the most likely diagnosis for prediction
    suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)
    primary_diagnosis = suggestions[0][0] if suggestions else "Generalized Anxiety"
    
    patient_data = {
        'Patient ID': 999,
        'Age': age,
        'Gender': gender,
        'Diagnosis': primary_diagnosis,
        'Symptom Severity (1-10)': symptom_severity,
        'Mood Score (1-10)': mood_score,
        'Sleep Quality (1-10)': max(1, 8 - stress_level // 2),  # Estimate based on stress
        'Physical Activity (hrs/week)': max(1, 7 - len(symptoms) // 3),  # Estimate
        'Medication': 'Not Specified',
        'Therapy Type': 'Not Specified',
        'Treatment Start Date': '2024-01-15',
        'Treatment Duration (weeks)': 12,
        'Stress Level (1-10)': stress_level,
        'Outcome': 'Unknown',
        'Treatment Progress (1-10)': 5,  # Neutral starting point
        'AI-Detected Emotional State': 'Anxious' if stress_level > 6 else 'Neutral',
        'Adherence to Treatment (%)': 75  # Assume moderate adherence
    }
    
    return pd.DataFrame([patient_data])


def predict_treatment_outcome(patient_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Predict treatment outcome using the trained model."""
    model_path = "models/best_model.pkl"
    
    if not os.path.exists(model_path):
        return "Model not available", {}
    
    try:
        from train import preprocess_data
        model = joblib.load(model_path)
        
        # Preprocess data
        X, _, _ = preprocess_data(patient_df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        
        prob_dict = {class_name: prob for class_name, prob in zip(classes, probabilities)}
        
        return prediction, prob_dict
        
    except Exception as e:
        return f"Prediction error: {str(e)}", {}


def main():
    """Main symptom checker function."""
    print("Welcome to the Mental Health Symptom Checker!")
    print("This tool will help assess your symptoms and provide insights.")
    print("\n⚠️  IMPORTANT: This is for educational purposes only.")
    print("Always consult with a healthcare professional for proper diagnosis and treatment.")
    
    # Get user information
    try:
        age = int(input("\nPlease enter your age: "))
        gender = input("Please enter your gender (Male/Female/Other): ").strip()
    except ValueError:
        print("Invalid input. Using default values.")
        age, gender = 25, "Other"
    
    # Get symptoms
    symptoms = get_user_symptoms()
    
    # Calculate scores
    mood_score, stress_level, symptom_severity = calculate_scores(symptoms)
    
    # Display results
    print("\n" + "="*60)
    print("ASSESSMENT RESULTS")
    print("="*60)
    
    print(f"\nCalculated Scores:")
    print(f"  Mood Score: {mood_score}/10 {'(Low mood)' if mood_score <= 4 else '(Good mood)' if mood_score >= 7 else '(Moderate mood)'}")
    print(f"  Stress Level: {stress_level}/10 {'(High stress)' if stress_level >= 7 else '(Moderate stress)' if stress_level >= 4 else '(Low stress)'}")
    print(f"  Symptom Severity: {symptom_severity}/10")
    
    # Suggest diagnoses
    suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)
    
    print(f"\nPotential Conditions to Discuss with a Healthcare Provider:")
    if suggestions:
        for diagnosis, confidence in suggestions[:3]:  # Show top 3
            print(f"  • {diagnosis}: {confidence:.1f}% match")
    else:
        print("  • No specific patterns identified. Consider general mental health consultation.")
    
    # Predict treatment outcome if model is available
    patient_df = create_patient_profile(symptoms, mood_score, stress_level, symptom_severity, age, gender)
    prediction, probabilities = predict_treatment_outcome(patient_df)
    
    if probabilities:
        print(f"\nTreatment Outcome Prediction:")
        print(f"  Most Likely Outcome: {prediction}")
        print(f"  Outcome Probabilities:")
        for outcome, prob in probabilities.items():
            print(f"    {outcome}: {prob:.1f}%")
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("• Consult with a mental health professional for proper evaluation")
    print("• Consider therapy or counseling")
    print("• Practice stress management techniques")
    print("• Maintain regular sleep and exercise routines")
    print("• Reach out to trusted friends or family for support")
    
    if any(symptom in symptoms for symptom in ['self_harm_thoughts']):
        print("\n🚨 URGENT: If you're having thoughts of self-harm, please:")
        print("   • Contact emergency services (911)")
        print("   • Call National Suicide Prevention Lifeline: 988")
        print("   • Go to your nearest emergency room")


if __name__ == "__main__":
    main()
