import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config

class RiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineering transformer for COVID-19 Risk Stratification.
    
    Generates derived clinical features:
    1. 'total_risk_factors': Aggregate sum of all comorbidities (proxy for frailty).
    2. 'age_bin': Discretizes age into epidemiologically relevant groups (Young, Adult, Senior, Elderly) 
       to capture non-linear risk escalation.
    """
    def __init__(self):
        self.comorbidities = config.COMORBIDITIES

    def fit(self, X, y=None):
        """Fit is a no-op for this transformer."""
        return self

    def transform(self, X):
        X = X.copy()
        
        # Check if comorbidity columns exist
        available_comorbs = [c for c in self.comorbidities if c in X.columns]
        
        if available_comorbs:
            # DataCleaner mapped 1->1 (Yes) and 2->0 (No)
            # So we sum the 1s directly
            X['total_risk_factors'] = X[available_comorbs].sum(axis=1)
            
        # Age Binning
        if 'age' in X.columns:
            X['age_bin'] = pd.cut(
                X['age'], 
                bins=config.AGE_BINS, 
                labels=config.AGE_LABELS, 
                include_lowest=True
            ).astype(int)
            
        return X
