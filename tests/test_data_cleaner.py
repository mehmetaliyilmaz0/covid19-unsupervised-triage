import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the library is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covid_risk_lib import DataCleaner

def test_data_cleaner_binary_mapping():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        'sex': [1, 2, 1],
        'patient_type': [2, 1, 2],
        'age': [25, 45, 65],
        'id': ['a', 'b', 'c']
    })
    
    df_transformed = cleaner.transform(df)
    
    # 1 -> 1 (Yes), 2 -> 0 (No)
    assert df_transformed['sex'].tolist() == [1, 0, 1]
    assert df_transformed['patient_type'].tolist() == [0, 1, 0]

def test_data_cleaner_missing_value_filtering():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        'pneumonia': [1, 98, 2], # 98 should cause row drop
        'diabetes': [2, 1, 99],  # 99 should cause row drop
        'age': [25, 30, 35],
        'sex': [1, 2, 1]
    })
    
    df_transformed = cleaner.transform(df)
    
    # Only the first row should remain because of 98 and 99 in others
    assert len(df_transformed) == 1
    assert df_transformed.iloc[0]['age'] == 25

def test_data_cleaner_special_mapping():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        'icu': [97, 1, 2], # 97 mapped to 2, then binary 2 -> 1
        'sex': [1, 1, 1],
        'age': [20, 21, 22]
    })
    
    df_transformed = cleaner.transform(df)
    
    # icu: 97 -> 2 -> 0 (No)
    # icu: 1 -> 1 (Yes)
    # icu: 2 -> 0 (No)
    assert df_transformed['icu'].tolist() == [0, 1, 0]
