# Medical Machine Learning Analysis

This repository contains some of the tasks carried out during the course of AI2 for medical data analysis, including classification and regression tasks on healthcare datasets. These are some of my very first projects, therefore **the coding style is rather crude.** You can read the full report for both of these projects [here](https://drive.google.com/file/d/1wqc_lJrObpMXZTVRfpE5AYMV6lvuRNrk/view?usp=sharing).
**Later sometime I will add better comments and compertmentalize the codes writing proper functions.**

## Projects

### 1. Patient Outcome Classification (AI2_project_take_1-2classes.ipynb)
- **Dataset**: [Support2](https://archive.ics.uci.edu/dataset/880/support2) medical patient data
- **Task**:  Classification of patient outcomes
- **Methods**: 
  - Exploratory analysis (univariate, bivariate, plotting to understand data distribution, correlation and other routine analyses)
  - Feature selection and preprocessing (identifying ordinal, numerical and categorial features then embedding accordingly, imputing, scaling)
  - SMOTE for handling class imbalance
  - Cross-validation with StratifiedKFold
  - MLP, SVC, Ensemble Method

- **Features**: Patient demographics, vital signs, lab values, comorbidities
- **Target**: Patient hospital outcomes
- **Results**: MLP brought out the best performance for both surv2m and surv6m in terms of accuracy. SVC, Linear SVC and other models (such as XGBoost), give the best predictions for class 3 but in other classes, such as 0, 1 and 2 the prediction varies a lot. Therefore, for all the models, patient survival with high survival category would be predicted better than all the other risk classes for both surv2m and surv6m

### 2. Medical Charges Regression (AI_project_take1_regression_charges.ipynb)
- **Dataset**: Support2 medical patient data
- **Task**: Predict medical charges and total costs
- **Methods**:
  - Deep Neural Networks, Linear Regression, Random Forrest Regressor, Gradient Boosting Regressor, Lasso, Ridge
  - Batch normalization and dropout for regularization
  - Feature selection and dimensionality reduction
- **Evaluation**: MSE, MAE, R² metrics
- **Results**: Random Forest Regressor for both the regression tasks showed superior performance than all the other models. 


### 3. Melanoma Cancer Classification (AI_2_Problem_2_take_1-Copy2.ipynb)
- **Dataset**: Melanoma cancer image dataset
- **Task**: Binary classification (benign vs malignant)
- **Methods**:
  - Custom CNN architectures, Network in Network, Autoencoders, Sparse Autoencoders
  - Transfer learning (ResNet50, VGG16, MobileNet)
  - Ensemble learning with meta-classifier
  - Data augmentation techniques
- **Image Size**: 150x150 RGB images
- **Classes**: Benign and Malignant melanoma
- **Results**: Rather than all the other pretrained models, after 50 epochs, my custom CNN model shows the highest accuracy for both training (0.9153) and validation (0.9219) with 2nd lowest test loss (0.2075) and validation loss (0.1931). But only a f1-score of 0.49 on the test dataset indicates overfitting of the model, even though it was not the case for the validation dataset. The lowest loss was observed for autoencoders model for both test (0.00041) and validation (0.000408) data sets. Further improvements can be done following the ensemble model and probably running for more epochs. Due to the machine’s constraint to run models for more epochs are quite limited, further improvement steps were not undertaken.


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

Each notebook is self-contained and can be run independently. 
- Performance visualization

AI 2 Course Project - University of Pisa
