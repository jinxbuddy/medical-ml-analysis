# Medical Machine Learning Analysis

This repository contains some of the tasks carried out during the course of AI2 for medical data analysis, including classification and regression tasks on healthcare datasets. These are some of my very first projects, therefore the coding style is rather crude.

## Projects

### 1. Patient Outcome Classification (AI2_project_take_1-2classes.ipynb)
- **Dataset**: [Support2](https://archive.ics.uci.edu/dataset/880/support2) medical patient data
- **Task**: Binary classification of patient outcomes
- **Methods**: 
  - MLP, SVC, Ensemble Method
  - Feature selection and preprocessing
  - SMOTE for handling class imbalance
  - Cross-validation with StratifiedKFold
- **Features**: Patient demographics, vital signs, lab values, comorbidities
- **Target**: Patient hospital outcomes

### 2. Medical Charges Regression (AI_project_take1_regression_charges.ipynb)
- **Dataset**: Support2 medical patient data
- **Task**: Predict medical charges and total costs
- **Methods**:
  - Deep Neural Networks for regression
  - Batch normalization and dropout for regularization
  - Feature selection and dimensionality reduction
- **Evaluation**: MSE, MAE, R² metrics

### 3. Melanoma Cancer Classification (AI_2_Problem_2_take_1-Copy2.ipynb)
- **Dataset**: Melanoma cancer image dataset
- **Task**: Binary classification (benign vs malignant)
- **Methods**:
  - Custom CNN architectures
  - Transfer learning (ResNet50, VGG16, MobileNet)
  - Ensemble learning with meta-classifier
  - Data augmentation techniques
- **Image Size**: 150x150 RGB images
- **Classes**: Benign and Malignant melanoma

## Data

The `data/support2.csv` file contains medical patient information with features including:
- Patient demographics (age, sex, race, education, income)
- Clinical measurements (vital signs, lab values)
- Disease information (diagnosis groups, comorbidities)
- Outcomes (mortality, hospital stay length, costs)

## Dependencies

- Python 3.10+
- TensorFlow/Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- imbalanced-learn (SMOTE)
- XGBoost

## Usage

Each notebook is self-contained and can be run independently. The notebooks include:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- Performance visualization

## Results

Models achieve competitive performance on their respective tasks:
- Patient classification: ~66% accuracy with neural networks
- Medical charges regression: Evaluated with MSE and MAE metrics
- Melanoma classification: Multiple CNN architectures with ensemble approach

## Author

AI 2 Course Project - University of Pisa
