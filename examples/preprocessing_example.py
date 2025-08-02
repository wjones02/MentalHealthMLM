"""
Example script demonstrating the usage of the data preprocessing module.

This script shows how to use the mental health data preprocessing module
for different use cases and scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    load_raw_dataset,
    preprocess_mental_health_data,
    clean_data,
    scale_numeric_features,
    save_cleaned_data
)
import pandas as pd


def example_1_basic_usage():
    """Example 1: Basic preprocessing with default settings."""
    print("="*60)
    print("EXAMPLE 1: Basic Preprocessing")
    print("="*60)
    
    # Simple one-line preprocessing
    processed_df, scalers = preprocess_mental_health_data()
    
    print(f"\nProcessed dataset shape: {processed_df.shape}")
    print(f"Number of scaled features: {len(scalers)}")
    
    return processed_df, scalers


def example_2_step_by_step():
    """Example 2: Step-by-step preprocessing for more control."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Step-by-Step Preprocessing")
    print("="*60)
    
    # Step 1: Load raw data
    print("\nStep 1: Loading raw dataset...")
    df = load_raw_dataset()
    print(f"Raw dataset shape: {df.shape}")
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data...")
    df_cleaned = clean_data(df)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    # Step 3: Scale features
    print("\nStep 3: Scaling features...")
    df_scaled, scalers = scale_numeric_features(df_cleaned)
    print(f"Scaled dataset shape: {df_scaled.shape}")
    
    # Step 4: Save processed data
    print("\nStep 4: Saving processed data...")
    output_path = "data/processed/example_processed_data.csv"
    save_cleaned_data(df_scaled, output_path)
    
    return df_scaled, scalers


def example_3_without_scaling():
    """Example 3: Preprocessing without feature scaling."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Preprocessing Without Scaling")
    print("="*60)
    
    # Process data without scaling
    processed_df, scalers = preprocess_mental_health_data(
        scale_features=False,
        output_path="data/processed/unscaled_data.csv"
    )
    
    print(f"\nProcessed dataset shape: {processed_df.shape}")
    print(f"Scalers returned: {scalers}")
    
    # Show some statistics of unscaled numeric features
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    print(f"\nNumeric columns statistics:")
    for col in numeric_cols[:3]:  # Show first 3 numeric columns
        print(f"{col}: min={processed_df[col].min():.2f}, max={processed_df[col].max():.2f}")
    
    return processed_df


def example_4_data_exploration():
    """Example 4: Exploring the processed data."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Exploration")
    print("="*60)
    
    # Load processed data
    processed_df, scalers = preprocess_mental_health_data()
    
    print("\nDataset Overview:")
    print(f"Shape: {processed_df.shape}")
    print(f"Memory usage: {processed_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\nData Types:")
    print(processed_df.dtypes.value_counts())
    
    print("\nMissing Values:")
    missing_values = processed_df.isnull().sum()
    print(f"Total missing values: {missing_values.sum()}")
    
    print("\nCategorical Variables:")
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = processed_df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    print("\nNew Date Features:")
    date_features = ['treatment_start_month', 'treatment_start_quarter', 'treatment_start_day_of_week']
    for feature in date_features:
        if feature in processed_df.columns:
            print(f"{feature}: range {processed_df[feature].min()} to {processed_df[feature].max()}")
    
    print("\nScaled Features Range Check:")
    if scalers:
        for col in list(scalers.keys())[:5]:  # Check first 5 scaled features
            min_val = processed_df[col].min()
            max_val = processed_df[col].max()
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")


def example_5_using_scalers():
    """Example 5: Using the returned scalers for new data."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Using Scalers for New Data")
    print("="*60)
    
    # Get processed data and scalers
    processed_df, scalers = preprocess_mental_health_data()
    
    print(f"\nAvailable scalers: {len(scalers)}")
    print("Scaler features:", list(scalers.keys())[:3], "...")
    
    # Example: Create a sample new data point
    new_data = {
        'Age': [45],
        'Symptom Severity (1-10)': [7],
        'Mood Score (1-10)': [5]
    }
    new_df = pd.DataFrame(new_data)
    
    print(f"\nOriginal new data:")
    print(new_df)
    
    # Scale the new data using existing scalers
    print(f"\nScaled new data:")
    for col in new_df.columns:
        if col in scalers:
            scaled_values = scalers[col].transform(new_df[[col]])
            new_df[col] = scaled_values
    
    print(new_df)


def main():
    """Run all examples."""
    print("Mental Health Data Preprocessing Examples")
    print("="*60)
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_step_by_step()
        example_3_without_scaling()
        example_4_data_exploration()
        example_5_using_scalers()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
