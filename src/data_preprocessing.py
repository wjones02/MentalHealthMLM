"""
Data preprocessing module for mental health dataset.

This module provides functionality to load, clean, and preprocess the mental health
diagnosis and treatment dataset. It includes robust data cleaning, missing value
handling, date conversion, and numeric scaling to prepare data for machine learning.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Default paths
DEFAULT_RAW_DATA_PATH = "data/raw/mental_health_diagnosis_treatment_.csv"
DEFAULT_PROCESSED_DATA_PATH = "data/processed/cleaned_mental_health_data.csv"


def load_raw_dataset(file_path: str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw mental health dataset from CSV file.

    Args:
        file_path (str): Path to the raw CSV file

    Returns:
        pd.DataFrame: Raw dataset

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {file_path} is empty")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime format and extract useful features.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with converted dates and extracted features
    """
    df = df.copy()

    # Convert Treatment Start Date to datetime
    if 'Treatment Start Date' in df.columns:
        df['Treatment Start Date'] = pd.to_datetime(df['Treatment Start Date'], errors='coerce')

        # Extract useful date features
        df['treatment_start_month'] = df['Treatment Start Date'].dt.month
        df['treatment_start_quarter'] = df['Treatment Start Date'].dt.quarter
        df['treatment_start_day_of_week'] = df['Treatment Start Date'].dt.dayofweek

        print("Date conversion completed. Extracted month, quarter, and day of week features.")

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using appropriate strategies for different data types.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with missing values filled
    """
    df = df.copy()

    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Found missing values in {missing_counts[missing_counts > 0].shape[0]} columns")

        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled {missing_counts[col]} missing values in '{col}' with median: {median_value}")

        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Treatment Start Date' and df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled {missing_counts[col]} missing values in '{col}' with mode: {mode_value}")

        # Handle datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            if df[col].isnull().sum() > 0:
                # For dates, we might want to use forward fill or a specific date
                df[col].fillna(method='ffill', inplace=True)
                print(f"Forward filled {missing_counts[col]} missing values in '{col}'")
    else:
        print("No missing values found in the dataset.")

    return df


def scale_numeric_features(df: pd.DataFrame, target_column: str = 'Outcome') -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Scale numeric features to [0,1] range using MinMaxScaler.

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column to exclude from scaling

    Returns:
        Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]: Scaled dataframe and dictionary of scalers
    """
    df = df.copy()
    scalers = {}

    # Identify numeric columns (excluding target and ID columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target column and ID columns from scaling
    columns_to_exclude = [target_column, 'Patient ID', 'treatment_start_month',
                         'treatment_start_quarter', 'treatment_start_day_of_week']
    numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude and col in df.columns]

    print(f"Scaling {len(numeric_columns)} numeric features to [0,1] range...")

    for col in numeric_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        print(f"Scaled '{col}': min={df[col].min():.3f}, max={df[col].max():.3f}")

    return df, scalers


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive data cleaning operations.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()

    print("Starting data cleaning process...")

    # 1. Convert dates and extract features
    df = convert_dates(df)

    # 2. Fill missing values
    df = fill_missing_values(df)

    # 3. Clean and standardize categorical values
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.title()

    if 'Diagnosis' in df.columns:
        df['Diagnosis'] = df['Diagnosis'].str.strip()

    if 'Medication' in df.columns:
        df['Medication'] = df['Medication'].str.strip()

    if 'Therapy Type' in df.columns:
        df['Therapy Type'] = df['Therapy Type'].str.strip()

    if 'Outcome' in df.columns:
        df['Outcome'] = df['Outcome'].str.strip()

    if 'AI-Detected Emotional State' in df.columns:
        df['AI-Detected Emotional State'] = df['AI-Detected Emotional State'].str.strip().str.title()

    # 4. Validate numeric ranges
    numeric_ranges = {
        'Age': (0, 120),
        'Symptom Severity (1-10)': (1, 10),
        'Mood Score (1-10)': (1, 10),
        'Sleep Quality (1-10)': (1, 10),
        'Physical Activity (hrs/week)': (0, 168),  # Max hours in a week
        'Treatment Duration (weeks)': (0, 104),    # Max 2 years
        'Stress Level (1-10)': (1, 10),
        'Treatment Progress (1-10)': (1, 10),
        'Adherence to Treatment (%)': (0, 100)
    }

    for col, (min_val, max_val) in numeric_ranges.items():
        if col in df.columns:
            # Clip values to valid range
            original_out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if original_out_of_range > 0:
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                print(f"Clipped {original_out_of_range} out-of-range values in '{col}' to [{min_val}, {max_val}]")

    print("Data cleaning completed successfully.")
    return df


def create_processed_directory(output_path: str) -> None:
    """
    Create the directory for processed data if it doesn't exist.

    Args:
        output_path (str): Path where the processed data will be saved
    """
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def save_cleaned_data(df: pd.DataFrame, output_path: str = DEFAULT_PROCESSED_DATA_PATH) -> None:
    """
    Save the cleaned dataset to disk, ensuring the directory exists.

    Args:
        df (pd.DataFrame): Cleaned dataframe to save
        output_path (str): Path where to save the cleaned data
    """
    # Ensure the output directory exists
    create_processed_directory(output_path)

    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned dataset to: {output_path}")
        print(f"Saved dataset shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error saving cleaned dataset: {str(e)}")


def preprocess_mental_health_data(
    input_path: str = DEFAULT_RAW_DATA_PATH,
    output_path: str = DEFAULT_PROCESSED_DATA_PATH,
    scale_features: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict[str, MinMaxScaler]]]:
    """
    Complete preprocessing pipeline for mental health dataset.

    This function orchestrates the entire preprocessing workflow:
    1. Load raw dataset
    2. Clean and preprocess data
    3. Scale numeric features (optional)
    4. Save processed data

    Args:
        input_path (str): Path to raw dataset
        output_path (str): Path to save processed dataset
        scale_features (bool): Whether to scale numeric features to [0,1] range

    Returns:
        Tuple[pd.DataFrame, Optional[Dict[str, MinMaxScaler]]]:
            Processed dataframe and scalers (if scaling was applied)
    """
    print("="*60)
    print("MENTAL HEALTH DATA PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Load raw dataset
    print("\n1. Loading raw dataset...")
    df = load_raw_dataset(input_path)

    # Step 2: Clean data
    print("\n2. Cleaning data...")
    df_cleaned = clean_data(df)

    # Step 3: Scale features (optional)
    scalers = None
    if scale_features:
        print("\n3. Scaling numeric features...")
        df_cleaned, scalers = scale_numeric_features(df_cleaned)
    else:
        print("\n3. Skipping feature scaling...")

    # Step 4: Save processed data
    print("\n4. Saving processed data...")
    save_cleaned_data(df_cleaned, output_path)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Original dataset shape: {df.shape}")
    print(f"Processed dataset shape: {df_cleaned.shape}")
    print(f"Output saved to: {output_path}")

    return df_cleaned, scalers


if __name__ == "__main__":
    # Example usage
    processed_df, feature_scalers = preprocess_mental_health_data()

    # Display basic statistics
    print("\nDataset Info:")
    print(processed_df.info())

    print("\nFirst 5 rows of processed data:")
    print(processed_df.head())