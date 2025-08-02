"""
Demo of the Mental Health Symptom Checker

This script demonstrates how the symptom checker works with sample inputs.
"""

import sys
import os
sys.path.append('src')

from symptom_checker import (
    calculate_scores, 
    suggest_diagnosis, 
    create_patient_profile, 
    predict_treatment_outcome,
    SYMPTOM_MAPPINGS
)


def demo_case_1():
    """Demo case: Depression symptoms"""
    print("="*60)
    print("DEMO CASE 1: Depression-like Symptoms")
    print("="*60)
    
    # Simulate user selecting depression symptoms
    symptoms = [
        'feeling_sad',
        'feeling_hopeless', 
        'loss_of_interest',
        'fatigue',
        'sleeping_too_much',
        'feeling_worthless',
        'social_withdrawal'
    ]
    
    print("Selected Symptoms:")
    symptom_descriptions = {
        'feeling_sad': "Feeling sad or down most of the time",
        'feeling_hopeless': "Feeling hopeless about the future",
        'loss_of_interest': "Loss of interest in activities",
        'fatigue': "Constant fatigue or low energy",
        'sleeping_too_much': "Sleeping much more than usual",
        'feeling_worthless': "Feeling worthless or guilty",
        'social_withdrawal': "Withdrawing from friends and family"
    }
    
    for symptom in symptoms:
        print(f"  • {symptom_descriptions[symptom]}")
    
    # Calculate scores
    mood_score, stress_level, symptom_severity = calculate_scores(symptoms)
    
    print(f"\nCalculated Scores:")
    print(f"  Mood Score: {mood_score}/10")
    print(f"  Stress Level: {stress_level}/10") 
    print(f"  Symptom Severity: {symptom_severity}/10")
    
    # Get diagnosis suggestions
    suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)
    
    print(f"\nSuggested Diagnoses:")
    for diagnosis, confidence in suggestions:
        print(f"  • {diagnosis}: {confidence:.1f}% match")
    
    # Create patient profile and predict
    patient_df = create_patient_profile(symptoms, mood_score, stress_level, symptom_severity, 28, "Female")
    prediction, probabilities = predict_treatment_outcome(patient_df)
    
    if probabilities:
        print(f"\nTreatment Outcome Prediction:")
        print(f"  Most Likely: {prediction}")
        for outcome, prob in probabilities.items():
            print(f"    {outcome}: {prob:.1f}%")


def demo_case_2():
    """Demo case: Anxiety symptoms"""
    print("\n" + "="*60)
    print("DEMO CASE 2: Anxiety-like Symptoms")
    print("="*60)
    
    symptoms = [
        'excessive_worry',
        'panic_attacks',
        'restlessness',
        'feeling_tense',
        'racing_thoughts',
        'difficulty_concentrating',
        'physical_symptoms_anxiety',
        'avoiding_activities'
    ]
    
    print("Selected Symptoms:")
    symptom_descriptions = {
        'excessive_worry': "Excessive worry about many things",
        'panic_attacks': "Panic attacks (sudden intense fear)",
        'restlessness': "Feeling restless or on edge",
        'feeling_tense': "Feeling tense or unable to relax",
        'racing_thoughts': "Racing thoughts",
        'difficulty_concentrating': "Difficulty concentrating",
        'physical_symptoms_anxiety': "Physical symptoms of anxiety",
        'avoiding_activities': "Avoiding activities or places"
    }
    
    for symptom in symptoms:
        print(f"  • {symptom_descriptions[symptom]}")
    
    # Calculate scores
    mood_score, stress_level, symptom_severity = calculate_scores(symptoms)
    
    print(f"\nCalculated Scores:")
    print(f"  Mood Score: {mood_score}/10")
    print(f"  Stress Level: {stress_level}/10")
    print(f"  Symptom Severity: {symptom_severity}/10")
    
    # Get diagnosis suggestions
    suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)
    
    print(f"\nSuggested Diagnoses:")
    for diagnosis, confidence in suggestions:
        print(f"  • {diagnosis}: {confidence:.1f}% match")
    
    # Create patient profile and predict
    patient_df = create_patient_profile(symptoms, mood_score, stress_level, symptom_severity, 35, "Male")
    prediction, probabilities = predict_treatment_outcome(patient_df)
    
    if probabilities:
        print(f"\nTreatment Outcome Prediction:")
        print(f"  Most Likely: {prediction}")
        for outcome, prob in probabilities.items():
            print(f"    {outcome}: {prob:.1f}%")


def demo_case_3():
    """Demo case: Mixed symptoms"""
    print("\n" + "="*60)
    print("DEMO CASE 3: Mixed Symptoms")
    print("="*60)
    
    symptoms = [
        'mood_swings',
        'irritability',
        'feeling_euphoric',
        'racing_thoughts',
        'risky_behavior',
        'insomnia',
        'feeling_sad'
    ]
    
    print("Selected Symptoms:")
    symptom_descriptions = {
        'mood_swings': "Extreme mood swings",
        'irritability': "Irritability or anger",
        'feeling_euphoric': "Feeling unusually euphoric or 'high'",
        'racing_thoughts': "Racing thoughts",
        'risky_behavior': "Engaging in risky or reckless behavior",
        'insomnia': "Trouble falling or staying asleep",
        'feeling_sad': "Feeling sad or down most of the time"
    }
    
    for symptom in symptoms:
        print(f"  • {symptom_descriptions[symptom]}")
    
    # Calculate scores
    mood_score, stress_level, symptom_severity = calculate_scores(symptoms)
    
    print(f"\nCalculated Scores:")
    print(f"  Mood Score: {mood_score}/10")
    print(f"  Stress Level: {stress_level}/10")
    print(f"  Symptom Severity: {symptom_severity}/10")
    
    # Get diagnosis suggestions
    suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)
    
    print(f"\nSuggested Diagnoses:")
    for diagnosis, confidence in suggestions:
        print(f"  • {diagnosis}: {confidence:.1f}% match")
    
    # Create patient profile and predict
    patient_df = create_patient_profile(symptoms, mood_score, stress_level, symptom_severity, 42, "Other")
    prediction, probabilities = predict_treatment_outcome(patient_df)
    
    if probabilities:
        print(f"\nTreatment Outcome Prediction:")
        print(f"  Most Likely: {prediction}")
        for outcome, prob in probabilities.items():
            print(f"    {outcome}: {prob:.1f}%")


def show_how_to_use():
    """Show how to use the actual symptom checker"""
    print("\n" + "="*60)
    print("HOW TO USE THE INTERACTIVE SYMPTOM CHECKER")
    print("="*60)
    
    print("\nTo run the interactive version:")
    print("  python src/symptom_checker.py")
    
    print("\nThe interactive version will:")
    print("  1. Ask for your age and gender")
    print("  2. Show you 32 different symptoms to choose from")
    print("  3. Calculate mood and stress scores based on your selections")
    print("  4. Suggest potential diagnoses")
    print("  5. Predict treatment outcomes using the ML model")
    print("  6. Provide recommendations and resources")
    
    print(f"\nAvailable symptoms include:")
    sample_symptoms = [
        "Feeling sad or down most of the time",
        "Excessive worry about many things", 
        "Panic attacks (sudden intense fear)",
        "Loss of interest in activities you used to enjoy",
        "Extreme mood swings",
        "Trouble falling or staying asleep",
        "And 26 more symptoms..."
    ]
    
    for symptom in sample_symptoms:
        print(f"  • {symptom}")


def main():
    """Run all demo cases"""
    print("MENTAL HEALTH SYMPTOM CHECKER - DEMO")
    print("This demo shows how the symptom checker works with different cases")
    print("\n⚠️  IMPORTANT: This is for educational purposes only!")
    
    try:
        demo_case_1()
        demo_case_2() 
        demo_case_3()
        show_how_to_use()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The symptom checker can help users:")
        print("• Input their symptoms interactively")
        print("• Get calculated mood and stress scores")
        print("• Receive potential diagnosis suggestions")
        print("• See treatment outcome predictions")
        print("• Get mental health resources and recommendations")
        
    except Exception as e:
        print(f"Demo error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
