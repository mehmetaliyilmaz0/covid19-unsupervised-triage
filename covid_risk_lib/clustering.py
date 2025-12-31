import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shap
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

def perform_lasso_selection(X, y):
    """
    Performs L1-regularized Logistic Regression (Lasso) with 5-Fold Cross-Validation.

    Scientific Rationale:
    In epidemiological datasets, high-dimensional comorbidity data is often sparse and noisy.
    L1 regularization induces sparsity, effectively "zeroing out" features that do not
    contribute significantly to the outcome (Mortality), acting as a strict feature selector.

    This method identifies the "Minimal Sufficient Set" of predictors.

    Returns:
        selected_features (list): List of features with non-zero coefficients.
        coefs (pd.Series): Coefficients of all features.
        optimal_C (float): The optimal inverse regularization strength (C = 1/alpha).
    """
    # L1 penalty requires 'liblinear' or 'saga' solver
    # Scaling is CRITICAL for Lasso (L1) regularization to treat all features fairly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        cv=5,
        penalty='l1',
        solver='liblinear',
        Cs=10,
        random_state=42,
        class_weight='balanced', # Good for imbalanced mortality
        max_iter=5000
    )
    model.fit(X_scaled, y)

    coefs = pd.Series(model.coef_[0], index=X.columns)
    selected_features = coefs[coefs != 0].index.tolist()

    return selected_features, coefs, model.C_[0]

def calculate_vif(X):
    """
    Calculates Variance Inflation Factor (VIF) to detect multicollinearity.

    Methodology:
    Multicollinearity inflates the variance of coefficient estimates, making models unstable across samples.
    - VIF = 1 / (1 - R^2_i)
    - Thresholds: VIF > 5 indicates moderate correlation; VIF > 10 indicates high redundancy.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.sort_values(by="VIF", ascending=False)

def build_clustering_pipeline(n_clusters=5, random_state=42):
    """
    Builds a K-Means clustering pipeline.

    Algorithm Choice:
    K-Means is chosen for its efficiency O(n) and strong performance on globular clusters.
    While it assumes spherical variance, it serves as our baseline for "General Population" stratification.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    return pipeline

def build_gmm_pipeline(n_clusters=5, random_state=42):
    """
    Builds a Gaussian Mixture Model (GMM) pipeline.

    Rationale:
    Unlike K-Means, GMM allows for "Soft Clustering" (probabilistic assignment).
    This is critical for medical data where patients may exhibit borderline phenotypes
    that don't fit squarely into a single category.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gmm', GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type='diag', n_init=5))
    ])
    return pipeline

def build_dbscan_pipeline(eps=0.5, min_samples=10):
    """
    Builds a DBSCAN pipeline.

    Rationale:
    DBSCAN is density-based and does not require a pre-specified number of clusters (K).
    It is uniquely suited for **Anomaly Detection** in this study, identifying "Noise" points (-1)
    which correspond to patients with rare, high-risk, or non-conforming comorbidity profiles.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('dbscan', DBSCAN(eps=eps, min_samples=min_samples))
    ])
    return pipeline

def build_agglomerative_pipeline(n_clusters=5):
    """
    Builds a clustering pipeline with scaling and Agglomerative Clustering.
    Note: Agglomerative is memory-intensive for large datasets.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('agg', AgglomerativeClustering(n_clusters=n_clusters))
    ])
    return pipeline

def build_birch_pipeline(n_clusters=5):
    """
    Builds a clustering pipeline with scaling and BIRCH.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('birch', Birch(n_clusters=n_clusters))
    ])
    return pipeline

def build_pca_dbscan_pipeline(eps=0.5, min_samples=5, n_components=0.95):
    """
    Builds a pipeline with StandardScaler, PCA, and DBSCAN.
    n_components: If float between 0-1, represents variance to retain.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('dbscan', DBSCAN(eps=eps, min_samples=min_samples))
    ])
    return pipeline

def train_surrogate_explainer(X, cluster_labels, max_depth=3, random_state=42):
    """
    Trains a global Surrogate Decision Tree.

    Methodology:
    Clustering algorithms are unsupervised and often "black boxes". To interpret the resulting
    groups clinically, we train a shallow Supervised Decision Tree to predict the cluster labels.
    The splits of this tree reveal the logical rules (e.g., "Age > 60 AND Diabetes = True")
    that define each phenotype.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X, cluster_labels)
    return dt

def plot_surrogate_explainer(dt_model, feature_names, class_names=None, figsize=(20, 10), filename='cluster_explanation_tree.png'):
    """
    Visualizes the surrogate decision tree and saves it to a file.

    Args:
        dt_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names (cluster labels)
        figsize: Tuple for figure size
        filename: Path to save the image
    """
    plt.figure(figsize=figsize)
    plot_tree(dt_model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Surrogate Decision Tree Explaining Clusters")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Decision tree visualization saved to {filename}")

def assess_cluster_stability(X, cluster_labels):
    """
    Calculates Silhouette Score and Davies-Bouldin Index.
    Note: Silhouette Score can be slow for large datasets.
    """

    if len(X) > 10000:
        # Use a fixed random state for reproducibility
        indices = np.random.RandomState(42).choice(len(X), 10000, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        labels_sample = cluster_labels[indices]
    else:
        X_sample = X
        labels_sample = cluster_labels

    if len(np.unique(labels_sample)) < 2:
        return {
            'silhouette_score': -1.0,
            'davies_bouldin_score': np.inf
        }

    sil = silhouette_score(X_sample, labels_sample)
    db = davies_bouldin_score(X, cluster_labels) # DB score is fast enough usually

    return {
        'silhouette_score': sil,
        'davies_bouldin_score': db
    }

def save_shap_summary(dt_model, X, feature_names, filename='shap_summary.png', cluster_idx=1):
    """
    Calculates and saves SHAP summary plot for the surrogate model targeting a specific cluster.
    """
    explainer = shap.TreeExplainer(dt_model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    # For multi-class (clusters), shap_values can be a list OR a 3D numpy array
    # Robust check for different SHAP/Sklearn versions
    if isinstance(shap_values, list):
        # Old version: list of (samples, features)
        target_shap = shap_values[cluster_idx]
    elif len(shap_values.shape) == 3:
        # New version: (samples, features, classes)
        target_shap = shap_values[:, :, cluster_idx]
    else:
        target_shap = shap_values

    shap.summary_plot(target_shap, X, feature_names=feature_names, show=False)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {filename}")

def save_experiment_metadata(config_obj, results, filename='experiment_metadata.json'):
    """
    Saves experiment details and results to a JSON file for tracking.
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_clusters': config_obj.N_CLUSTERS,
            'random_state': config_obj.RANDOM_STATE,
            'features': config_obj.CLUSTERING_FEATURES
        },
        'results': results
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # Clean dictionary of Infinity/NaN for strictly valid JSON
    def clean_nan(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nan(i) for i in obj]
        return obj

    metadata = clean_nan(metadata)

    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NpEncoder)
    print(f"Experiment metadata saved to {filename}")

def tune_dbscan(X, df, target_col='is_dead', eps_range=[0.3, 0.5, 0.7, 0.9], min_samples_range=[5, 10, 20]):
    """
    Auto-tunes DBSCAN parameters to maximize (Silhouette + Mortality Spread).
    Uses a sample for speed on large datasets.
    """
    # Sample for performance on 90k+ rows
    sample_size = min(20000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    df_sample = df.loc[X_sample.index]

    best_score = -1
    best_params = {}

    for eps in eps_range:
        for min_samples in min_samples_range:
            p = build_dbscan_pipeline(eps=eps, min_samples=min_samples)
            labels = p.fit_predict(X_sample)

            # 1. Stability (Silhouette)
            metrics = assess_cluster_stability(X_sample, labels)
            sil = metrics['silhouette_score']

            # 2. Clinical Relevance (Mortality Spread)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                spread = 0
            else:
                temp_df = df_sample.copy()
                temp_df['labels'] = labels
                mortality = temp_df.groupby('labels')[target_col].mean()
                spread = mortality.max() - mortality.min()

            # Combined Score (scaled 0-1)
            total_score = (0.5 * (sil + 1) / 2) + (0.5 * spread)

            if total_score > best_score:
                best_score = total_score
                best_params = {'eps': eps, 'min_samples': min_samples}

    return best_params, best_score


