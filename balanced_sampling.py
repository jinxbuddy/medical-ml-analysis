import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from tabpfn import TabPFNClassifier

class BalancedSampler(BaseEstimator, TransformerMixin):
    def __init__(self, target_samples=2500, random_state=42):
        self.target_samples = target_samples
        self.random_state = random_state
        self.under_sampler = RandomUnderSampler(sampling_strategy={3: target_samples}, random_state=random_state)
        self.over_sampler = SMOTE(random_state=random_state)
        
    def fit(self, X, y):
        return self
        
    def transform(self, X, y):
        # First undersample the majority class
        X_under, y_under = self.under_sampler.fit_resample(X, y)
        
        # Then oversample the minority classes
        X_resampled, y_resampled = self.over_sampler.fit_resample(X_under, y_under)
        
        return X_resampled, y_resampled

def create_pipeline(numerical_cols, ordinal_cols, onehot_cols):
    # Define the preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['under $11k', '$11-$25k', '$25-$50k', '>$50k'], 
                                            ['Coma or Intub', 'SIP>=30', 'adl>=4 (>=5 if sur)', 
                                             'no(M2 and SIP pres)', '<2 mo. follow-up']])),
        ('scaler', RobustScaler())
    ])

    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('ordinal_cat', ordinal_transformer, ordinal_cols),
            ("onehot_cat", onehot_transformer, onehot_cols)
        ])

    # Create the full pipeline with balanced sampling
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ("select-k-best", SelectKBest(k=10)),
        ("balanced_sampler", BalancedSampler(target_samples=2500, random_state=42)),
        ('TabPFN_classifier', TabPFNClassifier(device="cpu", ignore_pretraining_limits=True))
    ])

    return pipeline

def main():
    # Load your data
    data = pd.read_csv("./support2csv/support2.csv", delimiter=",")
    
    # Your data preprocessing steps here...
    # (Copy the relevant preprocessing steps from your notebook)
    
    # Create and fit the pipeline
    pipeline = create_pipeline(numerical_cols, ordinal_cols, onehot_cols)
    pipeline.fit(X_train, y_train)
    
    # Perform cross-validation
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=strat_kfold, n_jobs=1)
    
    return y_pred

if __name__ == "__main__":
    main() 