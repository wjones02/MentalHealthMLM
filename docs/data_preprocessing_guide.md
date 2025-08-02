# Mental Health Data Preprocessing Module

## Overview

The `data_preprocessing.py` module provides a comprehensive data preprocessing pipeline for the mental health diagnosis and treatment dataset. It offers robust data cleaning, missing value handling, date conversion, and numeric scaling functionality.

## Features

### 🔄 **Dataset Ingestion**
- **Function**: `load_raw_dataset()`
- Loads the raw mental health dataset from CSV
- Includes error handling for missing files and empty datasets
- Provides informative logging about dataset dimensions

### 🧹 **Robust Data Cleaning**
- **Function**: `clean_data()`
- Converts dates to datetime format and extracts useful features (month, quarter, day of week)
- Fills missing values using appropriate strategies:
  - Numeric columns: filled with median values
  - Categorical columns: filled with mode values
  - Date columns: forward-filled
- Standardizes categorical values (proper casing, trimming whitespace)
- Validates and clips numeric values to expected ranges

### 📊 **Feature Scaling**
- **Function**: `scale_numeric_features()`
- Scales numeric fields to [0,1] range using MinMaxScaler
- Excludes target variables and ID columns from scaling
- Returns both scaled data and fitted scalers for future use
- Provides detailed logging of scaling operations

### 💾 **Data Persistence**
- **Function**: `save_cleaned_data()`
- Ensures output directories exist before writing
- Saves processed data to CSV format
- Includes error handling and informative logging

## Usage

### Basic Usage

```python
from src.data_preprocessing import preprocess_mental_health_data

# Complete preprocessing pipeline
processed_df, scalers = preprocess_mental_health_data()
```

### Advanced Usage

```python
from src.data_preprocessing import (
    load_raw_dataset,
    clean_data,
    scale_numeric_features,
    save_cleaned_data
)

# Step-by-step processing
df = load_raw_dataset("path/to/raw/data.csv")
df_cleaned = clean_data(df)
df_scaled, scalers = scale_numeric_features(df_cleaned)
save_cleaned_data(df_scaled, "path/to/output.csv")
```

### Custom Paths

```python
# Use custom input and output paths
processed_df, scalers = preprocess_mental_health_data(
    input_path="custom/input/path.csv",
    output_path="custom/output/path.csv",
    scale_features=True
)
```

## Data Transformations

### Date Features
- **Original**: `Treatment Start Date` (YYYY-MM-DD format)
- **Added Features**:
  - `treatment_start_month` (1-12)
  - `treatment_start_quarter` (1-4)
  - `treatment_start_day_of_week` (0-6, Monday=0)

### Numeric Scaling
All numeric features are scaled to [0,1] range:
- Age
- Symptom Severity (1-10)
- Mood Score (1-10)
- Sleep Quality (1-10)
- Physical Activity (hrs/week)
- Treatment Duration (weeks)
- Stress Level (1-10)
- Treatment Progress (1-10)
- Adherence to Treatment (%)

### Data Validation
- **Age**: Clipped to [0, 120] years
- **Severity/Mood/Sleep/Stress/Progress Scores**: Clipped to [1, 10]
- **Physical Activity**: Clipped to [0, 168] hours/week
- **Treatment Duration**: Clipped to [0, 104] weeks (2 years max)
- **Adherence**: Clipped to [0, 100] percent

## Output Structure

The processed dataset includes:
- **Original columns**: All original features preserved
- **New date features**: Month, quarter, day of week
- **Scaled numeric features**: All numeric values in [0,1] range
- **Clean categorical data**: Standardized text formatting

## File Structure

```
data/
├── raw/
│   └── mental_health_diagnosis_treatment_.csv  # Input file
└── processed/
    └── cleaned_mental_health_data.csv          # Output file
```

## Testing

Run the test suite to verify functionality:

```bash
python test_preprocessing.py
```

The test suite validates:
- Dataset loading functionality
- Data cleaning operations
- Feature scaling accuracy
- Full pipeline integration
- Data integrity preservation

## Dependencies

- pandas
- numpy
- scikit-learn (MinMaxScaler)
- typing (for type hints)

## Error Handling

The module includes comprehensive error handling for:
- Missing input files
- Empty datasets
- Invalid data types
- File I/O operations
- Directory creation

## Logging

Detailed logging provides information about:
- Dataset dimensions
- Missing value handling
- Feature scaling operations
- File operations
- Processing progress

## Example Output

```
============================================================
MENTAL HEALTH DATA PREPROCESSING PIPELINE
============================================================

1. Loading raw dataset...
Successfully loaded dataset with 500 rows and 17 columns

2. Cleaning data...
Starting data cleaning process...
Date conversion completed. Extracted month, quarter, and day of week features.
No missing values found in the dataset.
Data cleaning completed successfully.

3. Scaling numeric features...
Scaling 9 numeric features to [0,1] range...
Scaled 'Age': min=0.000, max=1.000
...

4. Saving processed data...
Created directory: data/processed
Successfully saved cleaned dataset to: data/processed/cleaned_mental_health_data.csv

============================================================
PREPROCESSING COMPLETED SUCCESSFULLY!
============================================================
```
