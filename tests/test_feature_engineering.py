import pytest
import pandas as pd
import sys
import os

# Ensure the library is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covid_risk_lib import RiskFeatureEngineer

def test_risk_feature_engineer():
    engineer = RiskFeatureEngineer()
    # Mock data after DataCleaner (1=Yes, 0=No)
    df = pd.DataFrame({
        'pneumonia': [1, 0, 1], # Yes, No, Yes
        'diabetes': [1, 1, 0],  # Yes, Yes, No
        'copd': [0, 0, 0],
        'asthma': [0, 0, 0],
        'immunosuppression': [0, 0, 0],
        'hypertension': [0, 0, 0],
        'other_diseases': [0, 0, 0],
        'cardiovascular': [0, 0, 0],
        'obesity': [0, 0, 0],
        'chronic_kidney_failure': [0, 0, 0],
        'smoker': [0, 0, 0]
    })
    
    df_transformed = engineer.transform(df)
    
    # Row 0: pneumonia=Yes(1), diabetes=Yes(1) -> 2
    # Row 1: pneumonia=No(0), diabetes=Yes(1) -> 1
    # Row 2: pneumonia=Yes(1), diabetes=No(0) -> 1
    assert df_transformed['total_risk_factors'].tolist() == [2, 1, 1]
