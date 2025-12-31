import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the library is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covid_risk_lib import (
    DataCleaner, 
    RiskFeatureEngineer, 
    perform_lasso_selection, 
    tune_dbscan,
    build_clustering_pipeline
)
import config

def test_full_pipeline_integration():
    """
    Integration Test: Runs the entire pipeline flow on synthetic data.
    Goal: Ensure no runtime errors and correct data passing between modules.
    """
    # 1. Generate Synthetic Data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(n_samples),
        'sex': np.random.choice([1, 2], n_samples),
        'patient_type': np.random.choice([1, 2], n_samples),
        'age': np.random.randint(20, 90, n_samples),
        'date_died': np.random.choice(['9999-99-99', '2020-05-01'], n_samples, p=[0.9, 0.1]),
        'intubated': np.random.choice([1, 2, 97], n_samples),
        'pneumonia': np.random.choice([1, 2], n_samples), # 1=Yes, 2=No
        'pregnant': np.random.choice([1, 2, 97], n_samples),
        'diabetes': np.random.choice([1, 2], n_samples),
        'copd': np.random.choice([1, 2], n_samples),
        'asthma': np.random.choice([1, 2], n_samples),
        'immunosuppression': np.random.choice([1, 2], n_samples),
        'hypertension': np.random.choice([1, 2], n_samples),
        'other_diseases': np.random.choice([1, 2], n_samples),
        'cardiovascular': np.random.choice([1, 2], n_samples),
        'obesity': np.random.choice([1, 2], n_samples),
        'chronic_kidney_failure': np.random.choice([1, 2], n_samples),
        'smoker': np.random.choice([1, 2], n_samples),
        'icu': np.random.choice([1, 2, 97], n_samples)
    }
    
    df_raw = pd.DataFrame(data)
    
    # 2. Data Cleaning
    cleaner = DataCleaner()
    df_clean = cleaner.transform(df_raw)
    
    # Mirroring main.py logic: Create target column MANUALLY
    # DataCleaner cleans values but doesn't create 'is_dead'
    if 'date_died' in df_clean.columns:
        df_clean['is_dead'] = df_clean['date_died'].apply(lambda x: 0 if x == '9999-99-99' else 1)
    
    assert len(df_clean) > 0
    assert 'is_dead' in df_clean.columns
    # Check if encoding is 1/0
    assert df_clean['diabetes'].isin([0, 1]).all()
    
    # 3. Feature Engineering
    engineer = RiskFeatureEngineer()
    df_engineered = engineer.transform(df_clean)
    
    assert 'age_bin' in df_engineered.columns
    assert 'total_risk_factors' in df_engineered.columns
    
    # 4. Lasso Selection
    # Ensure we have the target
    X = df_engineered[config.CLUSTERING_FEATURES]
    y = df_engineered['is_dead']
    
    selected_feats, coefs, optimal_C = perform_lasso_selection(X, y)
    
    # Should return a list (empty or not)
    assert isinstance(selected_feats, list)
    
    # If Lasso selects nothing (possible on random data), we mock a fallback
    if len(selected_feats) == 0:
        selected_feats = config.CLUSTERING_FEATURES
        
    # 5. DBSCAN Tuning (Integration Check)
    # We use a small subset to be fast
    X_subset = df_engineered[selected_feats]
    best_params, best_score = tune_dbscan(X_subset, df_engineered, 'is_dead', 
                                          eps_range=[0.5], min_samples_range=[5])
    
    assert 'eps' in best_params
    assert 'min_samples' in best_params
    
    # 6. K-Means Clustering
    pipeline = build_clustering_pipeline(n_clusters=3, random_state=42)
    pipeline.fit(X_subset)
    
    labels = pipeline.named_steps['kmeans'].labels_
    assert len(labels) == len(df_engineered)
    assert len(set(labels)) <= 3 # Might find fewer than 3 if data is sparse
    
    print("\nIntegration Test Passed! Pipeline flows successfully.")
