# 🧠 Mental Health AI Diagnostic System

An advanced machine learning system for mental health diagnosis prediction and treatment outcome forecasting using ensemble algorithms and comprehensive symptom analysis.

## 🎯 Features

- **🔍 Interactive Symptom Checker**: 32+ comprehensive mental health symptoms
- **⚡ Quick Command-Line Diagnosis**: Instant results with simple commands
- **🤖 ML-Powered Predictions**: Random Forest classification with hyperparameter optimization
- **📊 Treatment Outcome Forecasting**: Predict Improved/No Change/Deteriorated outcomes
- **🧹 Advanced Data Preprocessing**: Automated cleaning, scaling, and feature engineering
- **📈 Comprehensive Evaluation**: Multiple metrics including ROC-AUC, F1-score, precision/recall

## 🚀 Quick Start

### Installation
```bash
pip install -r reqirement.txt
```

### Quick Diagnosis
```bash
python src/quick_diagnosis.py --symptoms "sad,worry,fatigue" --age 25 --gender female
```

### Interactive Symptom Checker
```bash
python src/symptom_checker.py
```

### Train Model
```bash
python src/train.py
```

## 📊 Supported Diagnoses

- **Major Depressive Disorder**
- **Generalized Anxiety Disorder**
- **Panic Disorder**
- **Bipolar Disorder**

## 🔧 Usage Examples

### Command Line (Quick Results)
```bash
# Basic usage
python src/quick_diagnosis.py --symptoms "panic,restless,tense"
# Output: Panic Disorder: 62.5%

# With demographics
python src/quick_diagnosis.py --symptoms "sad,hopeless,fatigue" --age 30 --gender male
# Output: Major Depressive Disorder: 76.9%

# Using symptom numbers
python src/quick_diagnosis.py --symptoms "1,3,5,12"
# Output: Major Depressive Disorder: 61.5%
```

### Interactive Mode
```bash
python src/symptom_checker.py
```
- Select from 32 different symptoms
- Get mood/stress scores automatically calculated
- Receive detailed diagnosis suggestions with confidence percentages
- View treatment outcome predictions

## 🏗️ Project Structure

```
MentalHealthMLM/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data
├── src/
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── train.py               # Model training with hyperparameter tuning
│   ├── evaluate.py            # Model evaluation & metrics
│   ├── predict.py             # Single prediction interface
│   ├── symptom_checker.py     # Interactive symptom assessment
│   └── quick_diagnosis.py     # Command-line diagnosis tool
├── models/
│   └── best_model.pkl         # Trained Random Forest model
├── notebooks/                 # Jupyter notebooks for analysis
├── docs/                      # Documentation
└── examples/                  # Usage examples
```

## 🧮 Technical Details

### Machine Learning Pipeline
- **Algorithm**: Random Forest Classifier with hyperparameter optimization
- **Features**: 20+ engineered features including symptom scores, demographics, treatment data
- **Preprocessing**: MinMax scaling, missing value imputation, categorical encoding
- **Validation**: Stratified k-fold cross-validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Data Processing
- **Input**: 500+ patient records with 17 original features
- **Output**: Processed dataset with 20 features (3 new date-based features)
- **Scaling**: All numeric features normalized to [0,1] range
- **Validation**: Comprehensive data integrity checks

### Performance
- **Accuracy**: ~36% (3-class classification)
- **Processing Speed**: Sub-second inference
- **Scalability**: Handles batch processing and real-time predictions

## 📈 Model Performance

```
Classification Report:
                 precision    recall  f1-score   support
Deteriorated       0.38      0.53      0.44        34
Improved           0.29      0.29      0.29        34
No Change          0.44      0.25      0.32        32

accuracy                           0.36       100
macro avg          0.37      0.36      0.35       100
weighted avg       0.37      0.36      0.35       100
```

## 🔬 Available Symptoms

The system recognizes 32 different mental health symptoms:
- Mood-related: sadness, hopelessness, mood swings, euphoria
- Anxiety-related: excessive worry, panic attacks, restlessness
- Physical: fatigue, insomnia, appetite changes, headaches
- Behavioral: social withdrawal, substance use, risky behavior

## ⚠️ Important Disclaimer

**This system is for educational and research purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper mental health evaluation and care.

## 🛠️ Development

### Running Tests
```bash
python test_symptom_checker.py
```

### Demo Examples
```bash
python demo_symptom_checker.py
```

### Model Retraining
```bash
python src/train.py
python src/evaluate.py
```

## 📚 Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- joblib >= 1.1.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mental health research community
- Open source machine learning libraries
- Healthcare professionals providing domain expertise

---

**Built with ❤️ for mental health awareness and AI-assisted healthcare**