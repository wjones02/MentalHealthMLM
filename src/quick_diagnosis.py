"""
Quick Mental Health Diagnosis Tool - Command Line Version

Usage:
    python src/quick_diagnosis.py --symptoms "sad,worry,fatigue" --age 25 --gender female
    python src/quick_diagnosis.py --symptoms "1,3,5" --age 30 --gender male
    python src/quick_diagnosis.py --help

This tool takes command line arguments and provides quick mental health assessments.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from symptom_checker import (
    calculate_scores,
    suggest_diagnosis,
    create_patient_profile,
    predict_treatment_outcome,
    SYMPTOM_MAPPINGS
)


# Symptom name to key mapping for easier input
SYMPTOM_SHORTCUTS = {
    'sad': 'feeling_sad',
    'hopeless': 'feeling_hopeless',
    'empty': 'feeling_empty',
    'worthless': 'feeling_worthless',
    'guilt': 'guilt',
    'interest': 'loss_of_interest',
    'crying': 'crying_spells',
    'irritable': 'irritability',
    'moods': 'mood_swings',
    'euphoric': 'feeling_euphoric',
    'energetic': 'feeling_energetic',
    
    'worry': 'excessive_worry',
    'panic': 'panic_attacks',
    'crowds': 'fear_of_crowds',
    'restless': 'restlessness',
    'tense': 'feeling_tense',
    'racing': 'racing_thoughts',
    'concentrate': 'difficulty_concentrating',
    'control': 'fear_of_losing_control',
    'anxiety': 'physical_symptoms_anxiety',
    
    'insomnia': 'insomnia',
    'sleep': 'sleeping_too_much',
    'fatigue': 'fatigue',
    'appetite': 'appetite_changes',
    'weight': 'weight_changes',
    'headache': 'headaches',
    'tension': 'muscle_tension',
    
    'withdraw': 'social_withdrawal',
    'avoid': 'avoiding_activities',
    'substance': 'substance_use',
    'harm': 'self_harm_thoughts',
    'risky': 'risky_behavior'
}


def parse_symptoms(symptom_input):
    """Parse symptom input - can be numbers or names."""
    symptoms = []
    symptom_list = list(SYMPTOM_MAPPINGS.keys())
    
    for item in symptom_input.split(','):
        item = item.strip().lower()
        
        # Try as number first
        try:
            num = int(item)
            if 1 <= num <= len(symptom_list):
                symptoms.append(symptom_list[num - 1])
                continue
        except ValueError:
            pass
        
        # Try as shortcut name
        if item in SYMPTOM_SHORTCUTS:
            symptoms.append(SYMPTOM_SHORTCUTS[item])
            continue
        
        # Try as full symptom key
        if item in SYMPTOM_MAPPINGS:
            symptoms.append(item)
            continue
        
        print(f"Warning: Unknown symptom '{item}' - skipping")
    
    return symptoms


def format_output(suggestions):
    """Format the output to show only top diagnosis."""
    if suggestions:
        top_diagnosis = suggestions[0]
        return f"{top_diagnosis[0]}: {top_diagnosis[1]:.1f}%"
    else:
        return "No clear diagnosis pattern found"


def show_symptom_list():
    """Show available symptoms with numbers."""
    print("Available Symptoms (use numbers or shortcut names):")
    print("=" * 50)
    
    symptom_list = list(SYMPTOM_MAPPINGS.keys())
    descriptions = {
        'feeling_sad': "Feeling sad or down",
        'feeling_hopeless': "Feeling hopeless",
        'loss_of_interest': "Loss of interest in activities",
        'feeling_empty': "Feeling empty inside",
        'crying_spells': "Frequent crying",
        'feeling_worthless': "Feeling worthless/guilty",
        'guilt': "Excessive guilt",
        'irritability': "Irritability or anger",
        'mood_swings': "Extreme mood swings",
        'feeling_euphoric': "Feeling unusually high/euphoric",
        'feeling_energetic': "Feeling unusually energetic",
        'excessive_worry': "Excessive worry",
        'panic_attacks': "Panic attacks",
        'fear_of_crowds': "Fear of crowds/social situations",
        'restlessness': "Feeling restless",
        'feeling_tense': "Feeling tense",
        'racing_thoughts': "Racing thoughts",
        'difficulty_concentrating': "Difficulty concentrating",
        'fear_of_losing_control': "Fear of losing control",
        'physical_symptoms_anxiety': "Physical anxiety symptoms",
        'insomnia': "Trouble sleeping",
        'sleeping_too_much': "Sleeping too much",
        'fatigue': "Constant fatigue",
        'appetite_changes': "Appetite changes",
        'weight_changes': "Weight changes",
        'headaches': "Frequent headaches",
        'muscle_tension': "Muscle tension",
        'social_withdrawal': "Social withdrawal",
        'avoiding_activities': "Avoiding activities",
        'substance_use': "Using substances to cope",
        'self_harm_thoughts': "Thoughts of self-harm",
        'risky_behavior': "Risky behavior"
    }
    
    for i, symptom in enumerate(symptom_list, 1):
        desc = descriptions.get(symptom, symptom.replace('_', ' ').title())
        print(f"{i:2d}. {desc}")
    
    print(f"\nShortcut names: {', '.join(sorted(SYMPTOM_SHORTCUTS.keys()))}")


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="Quick Mental Health Diagnosis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/quick_diagnosis.py --symptoms "sad,worry,fatigue" --age 25 --gender female
  python src/quick_diagnosis.py --symptoms "1,3,5,12" --age 30 --gender male
  python src/quick_diagnosis.py --symptoms "panic,restless,tense"
  python src/quick_diagnosis.py --list-symptoms
        """
    )

    parser.add_argument('--symptoms', '-s',
                       help='Comma-separated symptoms (numbers, names, or shortcuts)')
    parser.add_argument('--age', '-a', type=int, default=25,
                       help='Age (default: 25)')
    parser.add_argument('--gender', '-g', default='Other',
                       help='Gender (Male/Female/Other, default: Other)')
    parser.add_argument('--list-symptoms', '-l', action='store_true',
                       help='Show available symptoms and exit')
    
    args = parser.parse_args()
    
    # Show symptom list if requested
    if args.list_symptoms:
        show_symptom_list()
        return
    
    # Validate required arguments
    if not args.symptoms:
        print("Error: --symptoms is required")
        print("Use --list-symptoms to see available options")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Parse symptoms
        symptoms = parse_symptoms(args.symptoms)
        
        if not symptoms:
            print("Error: No valid symptoms found")
            print("Use --list-symptoms to see available options")
            sys.exit(1)
        
        # Calculate scores
        mood_score, stress_level, severity = calculate_scores(symptoms)
        
        # Get diagnosis suggestions
        suggestions = suggest_diagnosis(symptoms, mood_score, stress_level)

        # Format and output results
        output = format_output(suggestions)
        print(output)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
