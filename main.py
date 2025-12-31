import pandas as pd
import numpy as np
import os
import logging
import json
import matplotlib.pyplot as plt
import joblib
import config
from covid_risk_lib import (
    DataCleaner,
    RiskFeatureEngineer,
    build_clustering_pipeline,
    build_gmm_pipeline,
    build_dbscan_pipeline,
    build_pca_dbscan_pipeline,
    build_agglomerative_pipeline,
    build_birch_pipeline,
    assess_cluster_stability,
    train_surrogate_explainer,
    plot_surrogate_explainer,
    save_shap_summary,
    save_experiment_metadata,
    calculate_vif,
    tune_dbscan,
    perform_lasso_selection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"{data_path} not found.")

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Original shape: {df.shape}")
    return df

def preprocess_data(df):
    logger.info("Starting data cleaning...")
    cleaner = DataCleaner()
    df_clean = cleaner.transform(df)
    logger.info(f"Shape after cleaning: {df_clean.shape}")

    logger.info("Starting feature engineering...")
    engineer = RiskFeatureEngineer()
    df_engineered = engineer.transform(df_clean)

    if 'death_date' in df_engineered.columns:
        df_engineered['is_dead'] = (df_engineered['death_date'] != '9999-99-99').astype(int)
        logger.info(f"Mortality rate: {df_engineered['is_dead'].mean():.2%}")

    return df_engineered

def find_optimal_k(X):
    logger.info(f"Running Elbow Method for K range: {config.ELBOW_K_RANGE}...")
    inertias = []
    for k in config.ELBOW_K_RANGE:
        pipeline = build_clustering_pipeline(n_clusters=k, random_state=config.RANDOM_STATE)
        pipeline.fit(X)
        inertias.append(pipeline.named_steps['kmeans'].inertia_)

    # Save elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(config.ELBOW_K_RANGE, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'elbow_method.png'))
    plt.close()
    logger.info(f"Elbow plot saved to {os.path.join(config.FIGURES_DIR, 'elbow_method.png')}")

def run_clustering(df, features, model_type='kmeans'):
    X = df[features]
    if model_type == 'kmeans':
        logger.info(f"Running K-Means (K={config.N_CLUSTERS})...")
        pipeline = build_clustering_pipeline(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE)
    elif model_type == 'gmm':
        logger.info(f"Running GMM (K={config.N_CLUSTERS})...")
        pipeline = build_gmm_pipeline(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE)
    elif model_type == 'dbscan':
        logger.info(f"Running DBSCAN (eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES})...")
        pipeline = build_dbscan_pipeline(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES)
    elif model_type == 'pca_dbscan':
        # Using same EPS but with PCA pre-processing
        logger.info(f"Running PCA + DBSCAN (eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES})...")
        pipeline = build_pca_dbscan_pipeline(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES)
    elif model_type == 'agg':
        logger.info(f"Running Agglomerative Clustering (K={config.N_CLUSTERS}) on sample (10k)...")
        # Sample for Agglomerative due to complexity
        X_sample = X.sample(min(10000, len(X)), random_state=config.RANDOM_STATE)
        pipeline = build_agglomerative_pipeline(n_clusters=config.N_CLUSTERS)
        pipeline.fit(X_sample)
        # Note: Agglomerative doesn't have predict easily, we'll return the sample labels
        labels = pipeline.named_steps['agg'].labels_
        # To keep it consistent, we'll only update the sample in the df with these labels
        df.loc[X_sample.index, f'cluster_{model_type}'] = labels
        return df, pipeline, X_sample, labels
    elif model_type == 'birch':
        logger.info(f"Running BIRCH Clustering (K={config.N_CLUSTERS})...")
        pipeline = build_birch_pipeline(n_clusters=config.N_CLUSTERS)

    if model_type != 'agg':
        pipeline.fit(X)

    if model_type == 'kmeans':
        cluster_labels = pipeline.named_steps['kmeans'].labels_
    elif model_type == 'gmm':
        cluster_labels = pipeline.named_steps['gmm'].predict(X)
    elif model_type == 'dbscan':
        cluster_labels = pipeline.named_steps['dbscan'].labels_
    elif model_type == 'pca_dbscan':
        cluster_labels = pipeline.named_steps['dbscan'].labels_
    elif model_type == 'birch':
        cluster_labels = pipeline.named_steps['birch'].predict(X)

    if model_type != 'agg':
        df[f'cluster_{model_type}'] = cluster_labels
    return df, pipeline, X, cluster_labels

def evaluate_and_explain(X, cluster_labels, features, model_name):
    logger.info(f"--- Evaluating {model_name} ---")
    unique_labels = np.unique(cluster_labels)
    n_unique = len(unique_labels)

    metrics = assess_cluster_stability(X, cluster_labels)
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

    if n_unique < 2:
        logger.warning(f"Model {model_name} found only {n_unique} cluster(s). Skipping explanation/visualization.")
        return metrics

    logger.info(f"Training Surrogate Explainer for {model_name}...")
    dt = train_surrogate_explainer(X, cluster_labels, max_depth=config.DT_MAX_DEPTH, random_state=config.RANDOM_STATE)

    feature_importances = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)
    logger.info("Top 5 Features:\n%s", feature_importances.head(5).to_string())

    filename = f"cluster_explanation_{model_name}.png"
    # Map actual labels to readable names for the tree
    # If DBSCAN has -1, it will be mapped correctly.
    mapped_class_names = [f"L{label}" for label in sorted(unique_labels)]

    try:
        plot_surrogate_explainer(
            dt,
            feature_names=features,
            class_names=mapped_class_names,
            filename=os.path.join(config.FIGURES_DIR, filename)
        )
    except Exception as e:
        logger.error(f"Failed to plot tree for {model_name}: {e}")

    return metrics

def main():
    try:
        for directory in [config.OUTPUT_DIR, config.FIGURES_DIR, config.MODELS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        df = load_data(config.DATA_PATH)
        df_processed = preprocess_data(df)
        X_feats = config.CLUSTERING_FEATURES
        
        # 1. Feature Redundancy Audit (VIF)
        # We first check for multicollinearity. High VIF values (>5) suggest that some variables
        # (e.g., 'obesity' and 'diabetes') might be providing redundant information, potentially
        # destabilizing the clustering model.
        logger.info("Performing Feature Redundancy Audit (VIF)...")
        vif_df = calculate_vif(df_processed[X_feats])
        logger.info("\n%s", vif_df.to_string(index=False))

        # 2. FEATURE SELECTION (Lasso)
        # To reduce the "Curse of Dimensionality" and focus on clinically impactful drivers,
        # we employ a supervised Lasso (L1) step against the Mortality target.
        lasso_path = os.path.join(config.OUTPUT_DIR, 'lasso_selected_features.json')

        if os.path.exists(lasso_path):
            logger.info(f"Loading cached Lasso features from {lasso_path}...")
            with open(lasso_path, 'r') as f:
                X_feats = json.load(f)
        elif 'is_dead' in df_processed.columns:
            logger.info("Performing Supervised Feature Selection (Lasso - Alpha=0.05)...")
            selected_feats, coefs, optimal_C = perform_lasso_selection(df_processed[config.CLUSTERING_FEATURES], df_processed['is_dead'])

            logger.info(f"Lasso CV found Optimal C: {optimal_C:.4f} (Equivalent Alpha: {1/optimal_C:.4f})")

            if len(selected_feats) == 0:
                logger.warning("Lasso selected 0 features! Falling back to original feature set.")
                X_feats = config.CLUSTERING_FEATURES
            else:
                logger.info(f"Lasso Selected {len(selected_feats)} features.")
                X_feats = selected_feats
                # Cache the selection
                with open(lasso_path, 'w') as f:
                    json.dump(X_feats, f)
        else:
            logger.warning("Target 'is_dead' not found, skipping Lasso selection.")

        if 'is_dead' in df_processed.columns:
            # DBSCAN Parameter Tuning Caching:
            # We optimize Epsilon and MinPts only if not already cached.
            params_path = os.path.join(config.OUTPUT_DIR, 'best_dbscan_params.json')

            if os.path.exists(params_path):
                logger.info(f"Loading cached DBSCAN parameters from {params_path}...")
                with open(params_path, 'r') as f:
                    cached_params = json.load(f)
                config.DBSCAN_EPS = cached_params['eps']
                config.DBSCAN_MIN_SAMPLES = cached_params['min_samples']
                logger.info(f"Using Cached DBSCAN: eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES}")
            else:
                logger.info("No cached parameters found. Auto-Tuning DBSCAN...")
                best_params, best_score = tune_dbscan(df_processed[X_feats], df_processed, target_col='is_dead')

                config.DBSCAN_EPS = best_params['eps']
                config.DBSCAN_MIN_SAMPLES = best_params['min_samples']

                # Save for next time
                with open(params_path, 'w') as f:
                    json.dump(best_params, f)

                logger.info(f"✅ New Optimal DBSCAN Parameters Found and Cached: eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES} (Score={best_score:.4f})")
        else:
            logger.warning("Target 'is_dead' not found, using default DBSCAN parameters.")

        # Determine Optimal K (once)
        find_optimal_k(df_processed[X_feats])

        results = {}
        for m_type in ['kmeans', 'gmm', 'dbscan', 'pca_dbscan', 'agg', 'birch']:
            df_clustered, pipeline, X, cluster_labels = run_clustering(df_processed, X_feats, model_type=m_type)

            # Count actual clusters (excluding noise -1)
            n_found = len(np.unique(cluster_labels[cluster_labels != -1]))
            logger.info(f"{m_type} found {n_found} clusters.")

            metrics = evaluate_and_explain(X, cluster_labels, X_feats, m_type)

            cluster_mortality = df_clustered.groupby(f'cluster_{m_type}')['is_dead'].mean()
            cluster_counts = df_clustered[f'cluster_{m_type}'].value_counts()

            logger.info(f"\n[{m_type.upper()} CLINICAL PROFILE]")
            header = f"{'Cluster ID':<12} | {'Patients':<10} | {'Mortality %':<12} | {'Risk Level'}"
            logger.info("-" * len(header))
            logger.info(header)
            logger.info("-" * len(header))

            for c_id in cluster_mortality.sort_values(ascending=False).index:
                mort = cluster_mortality[c_id]
                size = cluster_counts[c_id]
                risk_level = "[HIGH]" if mort > 0.10 else ("[MEDIUM]" if mort > 0.03 else "[LOW]")
                logger.info(f"{str(c_id):<12} | {size:<10,} | {mort:<12.2%} | {risk_level}")

            results[m_type] = {
                'metrics': metrics,
                'mortality_spread': cluster_mortality.max() - cluster_mortality.min() if len(cluster_mortality) > 1 else 0.0
            }

            # Model Persistence: Save the pipeline
            model_path = os.path.join(config.MODELS_DIR, f"{m_type}_pipeline.joblib")
            joblib.dump(pipeline, model_path)
            logger.info(f"Model persisted to {model_path}")

        logger.info("\n--- Final Model Comparison ---")
        for m, res in results.items():
            logger.info(f"{m}: Sil={res['metrics']['silhouette_score']:.4f}, DB={res['metrics']['davies_bouldin_score']:.4f}, MortSpread={res['mortality_spread']:.2%}")

        # 3. Generate SHAP for Best Model (K-Means) - Targeting High Risk Cluster
        # We identify the cluster with the highest mortality to explain 'Risk'
        mortality_rates = df_processed.groupby('cluster_kmeans')['is_dead'].mean()
        high_risk_cluster = mortality_rates.idxmax()
        logger.info(f"Targeting High-Risk Cluster ({high_risk_cluster}) for SHAP Analysis (Mortality: {mortality_rates.max():.2%})")

        X_sample_shap = df_processed[X_feats].sample(min(1000, len(df_processed)), random_state=config.RANDOM_STATE)
        best_dt = train_surrogate_explainer(df_processed[X_feats], df_processed['cluster_kmeans'], max_depth=config.DT_MAX_DEPTH)

        logger.info("Generating Focused SHAP Summary Plot...")
        save_shap_summary(
            best_dt,
            X_sample_shap,
            X_feats,
            filename=os.path.join(config.FIGURES_DIR, 'shap_summary_kmeans.png'),
            cluster_idx=high_risk_cluster
        )

        # 4. Save Metadata
        save_experiment_metadata(
            config,
            results,
            filename=os.path.join(config.OUTPUT_DIR, 'experiment_metadata.json')
        )

        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.exception("Pipeline failed with an error:")

if __name__ == "__main__":
    main()

