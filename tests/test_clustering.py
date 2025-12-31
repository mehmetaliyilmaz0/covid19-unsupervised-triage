import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Ensure the library is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covid_risk_lib import build_clustering_pipeline, perform_lasso_selection

def test_build_clustering_pipeline_structure():
    """Test if the pipeline is built with correct steps"""
    pipeline = build_clustering_pipeline(n_clusters=3, random_state=42)
    
    assert isinstance(pipeline, Pipeline)
    assert 'scaler' in pipeline.named_steps
    assert 'kmeans' in pipeline.named_steps
    assert isinstance(pipeline.named_steps['scaler'], StandardScaler)
    assert isinstance(pipeline.named_steps['kmeans'], KMeans)
    assert pipeline.named_steps['kmeans'].n_clusters == 3

def test_lasso_selection_logic():
    """
    Test if Lasso correctly identifies a highly correlated feature 
    and ignores a random noise feature.
    """
    # Create synthetic data
    # 'predictive': perfectly matches target
    # 'noise': random noise
    np.random.seed(42)
    n_samples = 50
    
    # Target: 50% ones, 50% zeros
    y = np.array([0] * 25 + [1] * 25)
    
    # Feature 1: Predictive (90% match with y)
    # We flip a few bits so it's not perfect correlation (LassoCV needs some variance)
    feat_predictive = y.copy()
    feat_predictive[0] = 1 - feat_predictive[0] 
    feat_predictive[-1] = 1 - feat_predictive[-1]
    
    # Feature 2: Noise (Random)
    feat_noise = np.random.randint(0, 2, n_samples)
    
    X = pd.DataFrame({
        'predictive_feat': feat_predictive,
        'noise_feat': feat_noise
    })
    y_series = pd.Series(y)
    
    # Run Lasso
    # With 50 samples, CV should work
    selected_feats, coefs, optimal_C = perform_lasso_selection(X, y_series)
    
    # Assertions
    print(f"\nSelected: {selected_feats}")
    print(f"Coefs:\n{coefs}")
    
    # Predictive feature should definitely be selected
    assert 'predictive_feat' in selected_feats
    
    # Predictive feature coefficient should be non-zero and positive
    assert coefs['predictive_feat'] > 0
    
    # Noise feature should ideally be ignored, or have much lower coefficient
    # (Note: In very small N, noise might rarely get picked, but with 50 samples it usually drops)
    if 'noise_feat' in selected_feats:
        assert abs(coefs['predictive_feat']) > abs(coefs['noise_feat'])
