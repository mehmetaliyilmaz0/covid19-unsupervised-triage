import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import config

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline for the Mexican Health Ministry COVID-19 dataset.
    
    Handles specific data quality issues defined in the dataset dictionary:
    1. Mapping '97' (Not Applicable) in 'icu'/'intubated' to '2' (No/Negative).
    2. Filtering rows with '97', '98', '99' (Missing/Ignored) in key clinical columns.
    3. Standardization of binary flags: Maps {1: 'Yes', 2: 'No'} to {1: 1, 0: 0} for machine learning compatibility.
    """
    def __init__(self):
        self.binary_columns = config.BINARY_COLUMNS
        self.special_cols = config.SPECIAL_MAPPING_COLS
        self.cols_to_ignore = config.COLS_TO_IGNORE_FOR_MISSING_VALS

    def fit(self, X, y=None):
        """Fit is a no-op for this transformer."""
        return self

    def transform(self, X):
        X = X.copy()
        
        # Handle 97 in icu and intubated (Map to 2 -> No)
        for col in self.special_cols:
            if col in X.columns:
                X[col] = X[col].replace(97, 2)
        
        # Drop another_case if present
        if 'another_case' in X.columns:
            X = X.drop(columns=['another_case'])
            
        # Vectorized Filter: Drop rows where ANY checked column has 97, 98, or 99
        cols_to_check = [c for c in X.columns if c not in self.cols_to_ignore]
        mask = X[cols_to_check].isin([97, 98, 99]).any(axis=1)
        X = X[~mask]
            
        # Binary mapping 1->0, 2->1
        for col in self.binary_columns:
            if col in X.columns:
                # Map 1 (Yes) -> 1, 2 (No) -> 0, 97/98/99 are already filtered
                X[col] = X[col].replace({1: 1, 2: 0})
                
        return X
