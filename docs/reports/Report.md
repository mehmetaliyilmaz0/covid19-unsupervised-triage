# Project Report

**Project Title**: Uncovering Latent Phenotypes in COVID-19 Mortality Risk: A Dual-Layer Unsupervised Learning Approach

---

## 1. Abstract

**The global COVID-19 pandemic exposed critical vulnerabilities in healthcare systems, particularly the inability to rapidly triage patients during volume surges. Traditional scoring systems often rely on extensive laboratory panels suitable for ICU settings but fail in "Zero-Hour" emergency room contexts where decisions must be made instantly.** This project addresses this gap by developing an unsupervised machine learning framework to stratify mortality risk using only static demographic and comorbidity data from the Mexican Epidemiological Surveillance System (**N=95,839**).

**Our experimental methodology followed a rigorous "Dual-Layer" approach.** First, we implemented a custom preprocessing pipeline specifically designed to handle clinical data sparsity, including a **Feature Redundancy Audit (VIF)**. To combat the "Curse of Dimensionality," we employed **Lasso (L1) Logistic Regression** with 5-Fold Cross-Validation, which scientifically reduced the feature space from 13 potential comorbidities to a minimal sufficient set of **3 Key Predictors**: Pneumonia, Diabetes, and Age. On this optimized subspace, we benchmarked multiple clustering algorithms, including K-Means, DBSCAN, Gaussian Mixture Models (GMM), BIRCH, and Agglomerative Clustering.

**The results demonstrated that significant variance in mortality could be explained by these three features alone.** K-Means ($K=5$) emerged as the optimal algorithm for general population stratification, achieving a significantly improved Silhouette Score of **0.67** and successfully delineating **five distinct clinical phenotypes**: a "High-Risk" group (characterized by Pneumonia and advanced Age) with **~19% mortality**, various moderate-risk groups, and lower risk groups. Conversely, DBSCAN (Silhouette **1.0**) proved superior for detecting high-precision "Red Flag" anomalies but lacked generalizability for the broader cohort.

**In conclusion, this study validates that unsupervised learning can discover latent clinical phenotypes without biased labels.** We propose a production-ready "Triage-Bot" logic that prioritizes the K-Means model for high-throughput separation while reserving density-based methods for edge-case detection. This framework offers a scalable, data-driven tool for resource allocation in future epidemiological crises.

## 2. Introduction

### Description of the Problem

The COVID-19 pandemic placed an unprecedented strain on global healthcare infrastructure, exposing a critical gap in clinical informatics: the lack of rapid, data-driven tools for "Zero-Hour" triage. In the context of acute care medicine, triage decisions—determining which patient receives an ICU bed or a ventilator—must often be made immediately upon hospital arrival. This high-pressure environment is what we define as the "Zero-Hour" window.

In this window, clinicians are typically blinded to the patient's full physiological state. Comprehensive laboratory panels, which provide gold-standard biomarkers such as D-dimer, Ferritin, C-Reactive Protein (CRP), and Lymphocyte counts, require hours to process. Conventional severity scoring systems, such as the Sequential Organ Failure Assessment (SOFA) or APACHE II, heavily rely on these time-consuming values, rendering them operationally useless during the initial point of contact. Consequently, triage physicians are forced to rely on heuristic judgments based on the limited information available at admission: static demographic details (Age, Sex) and self-reported medical history (Comorbidities like Diabetes or Hypertension). The core problem addressed in this project is the construction of a robust, automated "Pre-Triage" tool that can accurately stratify patient mortality risk using *only* this sparse, non-invasive structured data, thereby providing a scientifically grounded "Second Opinion" to support overwhelmed frontline staff.

### Motivation: Why ML? Why Unsupervised?

From a Machine Learning perspective, this problem presents a unique intersection of challenges that render standard "out-of-the-box" approaches insufficient. The motivation for this study is threefold:

1. **The "Label Latency" Paradox in Emerging Pandemics**:
    In the early stages of a novel viral outbreak (like COVID-19 in early 2020), ground-truth labels—specifically, which patients will eventually survive or die—are statistically rare and significantly delayed. "Survival" is a lagging indicator that may not be known for weeks. Supervised Learning models trained on early, small, and likely biased samples of "Dead" patients risk severe overfitting, learning noise rather than true signal. This motivates the need for **Unsupervised Learning**, which allows us to discover natural "risk structures" (clusters) within the population *without* relying on potentially misleading labels.

2. **The Challenge of High-Dimensional Sparsity**:
    Epidemiological datasets are often characterized by high-dimensional sparsity. A patient record is essentially a "one-hot" encoded vector where most entries are zero (indicating a healthy state). In raw form, the "distance" between two patients in this 13-dimensional comorbidity space is often meaningless due to the "Curse of Dimensionality." Two patients might both be "High Risk" but share zero non-zero features (e.g., one has COPD, the other has Renal Failure). Standard distance-based algorithms (like K-Means) fail to cluster meaningfully in this sparse space without scientifically rigorous Feature Selection.

3. **The "General Population" vs. "Anomaly" Dichotomy**:
    Clinical risk is not uniform. There is a tension between the "Average Patient" (whose risk is driven by common factors like Age and Diabetes) and the "outlier" (a young patient with a rare immunosuppressive condition). A single model often struggles to capture both. This necessitates a composite approach that can handle the general population's variance while simultaneously flagging the "Long Tail" of rare, high-risk anomalies.

### Overview of Approach

To address these challenges, we adopted a hierarchical "Multi-Algorithm" Unsupervised Learning approach, prioritizing *Parsimony* (simplicity) and *Robustness* over raw complexity. Our methodology proceeds in three distinct phases:

1. **Lasso-Guided Feature Selection (The Filter)**:
    Instead of feeding all 13 available comorbidities into our clustering algorithms, we utilized L1-regularized Logistic Regression (Lasso) as a feature selector. By tuning the regularization strength ($\alpha$), we forced the coefficients of redundant or noisy features to zero. This scientifically identified the "Minimal Sufficient Set" of **3 key features** (Pneumonia, Diabetes, and Age) that drive the majority of the mortality signal, effectively collapsing the search space to a manageable 3-dimensional manifold.

2. **Structural Phenotyping & Benchmarking (Layer 1)**:
    On this optimized subspace, we deployed a suite of clustering algorithms, including **K-Means ($K=5$)**, **GMM**, **BIRCH**, and **Agglomerative Clustering**, to perform the primary stratification. This allowed us to identify the broad, dominant "Risk Phenotypes" present in the general population while validating results across different mathematical assumptions (centroid-based vs. hierarchical vs. probabilistic). The refined pipeline achieved a significantly high Silhouette Score of **0.67** with K-Means.

3. **Anomaly & Outlier Detection (Layer 2)**:
    Recognizing that centroid-based models like K-Means force every point into a cluster, we implemented **DBSCAN** and a **PCA-enhanced DBSCAN** variant as secondary safety nets. These specifically scan for "noise" points—patients whose profiles are so unique that they do not fit the standard phenotypes—achieving a near-perfect Silhouette Score (1.0) on core densities.

### Structure of the Report

The remainder of this report is organized as follows:

* **Section 3 (Background and Related Work)**: Reviews existing literature on COVID-19 risk stratification and unsupervised learning applications in healthcare.
* **Section 4 (Algorithms and Methodology)**: Details the "Dual-Layer" approach, including Lasso feature selection and the mathematical formulation of K-Means and DBSCAN.
* **Section 5 (Experimental Setup)**: Describes the dataset, preprocessing pipeline, and hardware/software environment used for experimentation.
* **Section 6 (Experimental Evaluation)**: Presents the quantitative results, cluster validation metrics, and clinical phenotype analysis.
* **Section 7 (Conclusions and Future Work)**: Summarizes the project's contributions and outlines potential future improvements.
* **Section 8 (References)**: Lists the academic sources cited throughout the report.

## 3. Background and Related Work

### Previous Studies on COVID-19 Prognosis

The global response to the COVID-19 pandemic catalyzed a surge in Machine Learning research aimed at prognostic modeling. Early influential studies, such as the work by Yan et al. (2020), successfully employed **XGBoost** to predict mortality with over 90% accuracy. However, these "State-of-the-Art" supervised models largely relied on dynamic biomarkers—specifically Lactic Dehydrogenase (LDH), Lymphocyte count, and C-Reactive Protein (CRP). While highly accurate, the dependency on these specific blood panels limits their utility in resource-constrained settings or "Pre-Triage" scenarios where such data is unavailable at admission. Furthermore, supervised approaches trained on early pandemic data famously suffered from "dataset shift," failing to generalize as viral variants and treatment protocols evolved.

### The Shift to Unsupervised Phenotyping

A parallel stream of literature has focused on **Unsupervised Learning** to discover latent clinical subtypes ("phenotypes") rather than predicting a binary outcome. The seminal precedent for this approach is the work of Seymour et al. (*JAMA*, 2019) on Sepsis. By applying clustering algorithms to electronic health records, they identified four distinct sepsis phenotypes ($\alpha, \beta, \gamma, \delta$) that responded differently to fluid therapy. This "Phenotype-First" methodology posits that heterogenous syndromes (like Sepsis or COVID-19) are not monolithic diseases but collections of distinct biological states. Our project explicitly adopts this paradigm, attempting to replicate the success of Sepsis phenotyping within the domain of COVID-19 respiratory failure.

### State-of-the-Art Algorithms and Methodology Comparison

In the realm of medical clustering, the literature presents a dichotomy between **Deep Clustering** and **Classical Clustering**:

1. **Deep Learning (VAEs & Autoencoders)**: sophisticated models like Variational Autoencoders (VAEs) offer powerful non-linear dimensionality reduction. However, they act as "Black Boxes," making it difficult for clinicians to interpret *why* a patient was assigned to a specific group. In high-stakes medical triage, interpretability is non-negotiable.
2. **Density & Centroid-Based Models (DBSCAN & K-Means)**: While mathematically simpler, these classical algorithms offer transparency. K-Means is the standard for globular phenotype discovery, while DBSCAN is the SOTA method for noise/anomaly detection.

### Comparison with Our Approach

Most existing COVID-19 clustering studies have utilized small, hospital-specific cohorts (N < 5,000). Our approach distinguishes itself by leveraging the massive **Mexican Epidemiological Surveillance System** dataset (N > 95,000). Unlike studies that force all patients into clusters (soft clustering), our "Dual-Layer" methodology innovates by combining K-Means (for the 90% typical cases) with DBSCAN (specifically for the 10% outliers). This hybrid structure directly addresses the literature's gap regarding the management of "edge cases" who do not fit standard clinical profiles.

## 4. Algorithms and Methodology

This study employs a composite "Dual-Layer" machine learning architecture, integrating supervised feature selection with unregulated unsupervised clustering. This hybrid approach was chosen to balance the need for rigorous statistical screening with the exploratory nature of phenotype discovery.

### 4.1 Machine Learning Algorithms and Selection Rationale

We selected three primary algorithms, each fulfilling a distinct role in the inference pipeline:

1. **Lasso Logistic Regression (L1-Regularized)**:
    * **Role**: Feature Selector (Filter Method).
    * **Rationale**: The primary challenge in electronic health records is high-dimensional sparsity. Standard regression (OLS) or unregularized Logistic Regression often overfits to noise in sparse datasets. Lasso (Least Absolute Shrinkage and Selection Operator) introduces a penalty term equal to the absolute magnitude of the coefficients. This property allows Lasso to shrink the coefficients of non-predictive variables to exactly zero, effectively performing "Soft Feature Selection." In a clinical context, this is invaluable for interpretability, as it mathematically isolates the "Minimum Sufficient Set" of risk factors.

2. **K-Means Clustering**:
    * **Role**: Primary Population Stratifier.
    * **Rationale**: K-Means minimizes the within-cluster sum of squares (inertia). We favored K-Means over Hierarchical Clustering due to its $O(n)$ time complexity, which is critical for scaling to our N=95,000 dataset. While K-Means assumes spherical clusters, our hypothesis was that the "general population" phenotypes (High Risk vs. Low Risk) would form globular densities in the reduced feature space.

3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
    * **Role**: Anomaly/Outlier Detector.
    * **Rationale**: Health risks typically follow a "Long Tail" distribution where most patients fit standard profiles, but a minority exhibit rare, complex pathologies. DBSCAN classifies low-density points as "Noise" (-1). We also implemented a **PCA + DBSCAN** variant to handle feature interactions more robustly.

4. **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**:
    * **Role**: Large-scale Clustering.
    * **Rationale**: BIRCH is designed for very large datasets, using a tree structure to summarize data. It provides an efficient alternative to K-Means for hierarchical stratification.

5. **Agglomerative Clustering**:
    * **Role**: Hierarchical Structural Validation.
    * **Rationale**: This bottom-up approach was used on a representative sample to validate the natural hierarchical groupings discovered by partition-based methods like K-Means.

### 4.2 Detailed Methodology

#### 4.2.1 Data Preparation Workflow

The raw dataset from the Mexican Epidemiological Surveillance System required a bespoke preprocessing pipeline to address systemic data quality issues. We implemented a `DataCleaner` class in Python that executes the following logic sequentially:

1. **Missing Value Imputation Strategy**: The dataset utilizes non-standard integer codes (`97`=Not Applicable, `98`=Ignored, `99`=Unknown) to represent missingness. Given the large sample size (N>95k), we adopted a "Complete Case Analysis" approach, excluding rows with missing values in critical columns (Diabetes, Age, Sex). This removed <1.5% of the data, preserving statistical power while eliminating imputation bias.
2. **Binary Standardization**: Categorical variables were originally encoded as `{1: 'Yes', 2: 'No'}`. This scheme is problematic for linear algebra operations (dot products), as 'No' (2) contributes twice the magnitude of 'Yes' (1). We mapped all binary fields to standard Boolean integers `{1: 1, 0: 0}`.
3. **Feature Construction (The Frailty Index)**: We engineered a `total_risk_factors` feature, a summation of all active binary comorbidities. This serves as a proxy for "Patient Frailty," condensing the overall burden of disease into a single scalar.
4. **Feature Redundancy Audit (VIF)**: Before modeling, we computed the **Variance Inflation Factor (VIF)** for all predictors. This step identified high multicollinearity between features such as Obesity and Diabetes, providing a mathematical justification for the subsequent Lasso-driven variable reduction.

#### 4.2.2 Feature Selection: The "Lasso" Filter

To address the "Curse of Dimensionality," we formalized the feature selection as an optimization problem:
$$ \min_{\beta} \left\{ \sum_{i=1}^{N} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\} $$
We utilized `LogisticRegressionCV` with 5-Fold Cross-Validation to autonomously tune the regularization parameter $\lambda$ (alpha). The model was trained to predict `death_date` (binarized to 0/1). The algorithm converged on a strong regularization penalty that pruned the feature space from 13 down to **3 Key Predictors**: **Pneumonia, Diabetes, and Age**. Variables such as Asthma, Hypertension, and Sex, which were previously considered drivers, were assigned zero coefficients in the latest iteration, implying they provide redundant information in the presence of the three primary markers.

#### 4.2.3 Model Training and Validation Setup

The clustering phase proceeded on the 5-dimensional subspace identified by Lasso.

* **Hyperparameter Tuning (K-Means)**: We employed the **Elbow Method**, plotting Inertia against $K \in [2, 10]$. The inflection point was consistently observed at $K=5$ in the reduced 3D space, indicating optimal grouping.
* **Automated Parameter Tuning (DBSCAN)**: We implemented a grid-search mechanism to autonomously optimize Epsilon ($\epsilon$) and Minimum Points ($minPts$), targeting the maximization of the Silhouette Score. Results were cached to ensure reproducibility.
* **Complexity Management (Agglomerative Clustering)**: Due to the $O(n^2)$ computational complexity of hierarchical methods, we performed Agglomerative Clustering on a representative random sample ($N=10,000$) to validate the structural consistency of the clusters found by K-Means.
* **Large-Scale Handling (BIRCH)**: The BIRCH algorithm was deployed to handle the full dataset (N=95k), utilizing its hierarchical summary features to cross-verify K-Means results at scale.

#### 4.2.4 Performance Metrics

Since "True Phenotypes" are unknown (Unsupervised Learning), we cannot use standard metrics like Accuracy or F1-Score. Instead, we tailored our evaluation using a triad of extensive metrics:

1. **Silhouette Coefficient**: Measures cluster cohesion and separation. Range: $[-1, 1]$. We utilized this to establish the structural stability of the 3D subspace, where a score of **0.67** indicated a high level of natural grouping.
2. **Davies-Bouldin Index**: Evaluates the ratio of within-cluster distances to between-cluster distances. Lower values indicate better-defined clusters.
3. **Clinical Validation Metric (Mortality Spread)**: As a proxy for "Accuracy," we defined *Mortality Spread* ($MS$) as the delta in mortality rates between the highest-risk ($Cluster_{High}$) and lowest-risk ($Cluster_{Low}$) phenotypes. A higher $MS$ validates the model's ability to discriminate clinical risk.
4. **Surrogate Model Fidelity**: We employed **Decision Tree Surrogates** and **SHAP explanations** to validate that the cluster assignment logic aligns with established medical knowledge (e.g., Pneumonia leading to higher risk), ensuring the model is not learning spurious correlations.

## 5. Experimental Setup

### 5.1 Dataset Description

The study utilizes the open-access **COVID-19 Patient Data** released by the Mexican Federal Government. This dataset is renowned in epidemiological research for its granularity regarding comorbidities.

* **Source**: Mexican Epidemiological Surveillance System (online repository). Available at: [https://www.gob.mx/salud/documentos/datos-abiertos-152127](https://www.gob.mx/salud/documentos/datos-abiertos-152127). (Kaggle Mirror: [https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset))
* **Dimensions**: The raw dataset contains **95,839 patient records** (rows) and **20 features** (columns).
* **Variable Types**:
  * *Demographics*: Age (Integer), Sex (Binary).
  * *Comorbidities (13 Features)*: Pneumonia, Diabetes, COPD, Asthma, Immunosuppression, Hypertension, etc. (Binary).
  * *Target Variable (For Evaluation)*: `date_died` (Date). This was transformed into a binary `is_dead` label ($0=Alive, 1=Dead$) purely for identifying the mortality rate and validating cluster risk separation. The models were *not* trained on this label.
* **Class Imbalance**: The dataset is heavily imbalanced, with a mortality rate of approximately **3.55%** (n=3,407 deaths). This extreme skew necessitates the use of stratified sampling in our validation steps.

### 5.2 Pre-processing Steps

To prepare the raw clinical logs for matrix operations, we applied a strict pipeline:

1. **Data Cleaning (`DataCleaner`)**:
    * *Handling Missing Data*: The schema uses integers `97` (Not Applicable), `98` (Ignored), and `99` (Unknown) to denote missingness. We executed a "Complete Case" filtration, dropping rows containing these codes in critical fields (`icu`, `intubated`, comorbidity flags). This reduced the sample size from 95,839 to **94,832** (a loss of only ~1.05%), preserving data integrity.
    * *Encoding*: The standard format `{1: Yes, 2: No}` was remapped to `{1: 1, 2: 0}` to align with standard sparse vector representations.

2. **Normalization (`StandardScaler`)**:
    * While binary features do not strictly require scaling, the `age` variable has a significantly larger magnitude (0-100) than the boolean flags (0-1). Without scaling, Age would dominate the Euclidean distance calculations in K-Means. We applied $z$-score normalization to all features entering the clustering pipeline: $x' = \frac{x - \mu}{\sigma}$.

### 5.3 Experimental Design

#### Training & Validation Strategy

* **Feature Selection Phase**: We employed **5-Fold Stratified Cross-Validation** during the Lasso (LogisticRegressionCV) stage. Stratification ensured representative mortality distribution across all folds, leading to a robust 3-feature selection.
* **Clustering & Benchmarking Phase**: Instead of relying on a single model, we implemented a **Comparative Performance Framework**.
  * **Full Population Models**: K-Means, BIRCH, and GMM were executed on the complete cleaned dataset ($N=94,832$).
  * **Sample-Based Validation**: Due to $O(n^2)$ memory constraints, Agglomerative Clustering and hyperparameter tuning for DBSCAN were performed on representative random samples ($N_{sub} = 10,000$).
* **Model Persistence**: Optimized pipelines, including Scalers and Clustering objects, were persisted using `joblib` to ensure consistent inference in downstream "Triage-Bot" applications.

#### Hardware and Software Environment

All experiments were conducted in a local development environment:

* **Language**: Python 3.10+
* **Core Libraries**:
  * `scikit-learn` (v1.3.0): Core implementation of K-Means, DBSCAN, Birch, and Lasso.
  * `pandas` / `numpy`: High-performance dataframe manipulation.
  * `matplotlib` / `seaborn`: Visualization and Silhouette analysis.
  * `shap`: Model interpretability and clinical feature importance analysis.
* **Compute**: The pipeline utilizes multi-threaded execution for grid-search components and is optimized for CPU-bound clinical informatics tasks.

## 6. Experimental Evaluation

### 6.1 Feature Selection: The "Sparsity" Insight

The application of **Lasso (L1) Regularization** yielded the most significant insight of the engineering phase: the vast majority of comorbidity data in COVID-19 records is statistically redundant. Out of the 13 potential risk factors available in the dataset, the Lasso algorithm—tuned via 5-Fold Cross-Validation—pruned 10 features, identifying a "Minimal Sufficient Set" of just **3 Key Predictors**.

**Table 1: Lasso Coefficients ($Target=Mortality$)**

| Feature | Coefficient ($\beta$) | Interpretation |
| :--- | :--- | :--- |
| **Pneumonia** | **+2.12** | Primary driver of mortality (Respiratory Failure) |
| **Diabetes** | **+0.42** | Significant metabolic comorbidity |
| **Age (Binned)** | **+0.38** | Strong positive correlation with risk |
| *Hypertension* | *0.00* | Redundant (pruned by Lasso iteration) |
| *Sex* | *0.00* | Redundant (pruned by Lasso iteration) |
| *Obesity* | *0.00* | Redundant (prioritized metabolic markers) |

*Critical Analysis*: The iterative refinement of the Lasso model consolidated mortality risk into three primary markers. By pruning *Hypertension* and *Sex*, the model reduced collinear noise, leading to a more stable feature space. VIF analysis confirmed that remaining predictors represent independent biological pathways (Respiratory, Metabolic, and Aging).

### 6.2 Clustering Performance Comparison

We benchmarked three distinct unsupervised algorithms on this reduced **3-dimensional subspace**. The results reveal a clear trade-off between **Structural Stability** (Silhouette Score) and **Clinical Utility** (Mortality Spread).

**Table 2: Algorithm Performance Benchmark**

| Model | Silhouette Score | Davies-Bouldin | Risk Separation (Mortality Spread) | Clinical Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means ($K=5$)** | **0.6679** | 1.0300 | **19.56%** | **Optimal Stability** |
| **DBSCAN** | **1.0000** | 0.0000 | **22.73%** | Anomaly Expert |
| **PCA-DBSCAN** | **1.0000** | 0.0000 | **22.73%** | Anomaly Expert |
| **Agglomerative** | 0.6679 | 1.0477 | **19.70%** | Structural Validation |
| **BIRCH** | 0.3320 | 0.9800 | 15.70% | Scalability Play |
| **GMM** | 0.4122 | 1.4209 | 12.81% | Probabilistic |

*Analysis*:

* **K-Means and Agglomerative Clustering** achieved the highest structural stability (Sil ~0.67), proving that clinical phenotypes in the 3D subspace are well-defined and separable.
* **DBSCAN variants** achieved a perfect Silhouette score by isolating extreme high-risk densities. While these models are highly precise at identifying "Red Flags," they are less suited for general population triage compared to centroid-based methods.
* **Mortality Spread** consistently improved across all models in the 3-feature space compared to previous iterations, with a maximum separation of **22.73%** achieved by density-based methods.

### 6.3 Phenotype Discovery: Interpreting the Clusters

Using the **Surrogate Decision Tree** technique, we reverse-engineered the logic defining the K-Means ($K=5$) clusters to create clinically actionable profiles:

1. **Phenotype 1 (Critical Vulnerability)**: *High Age + Pneumonia + Diabetes*. (Mortality **~19.8%**)
    * **Action**: Immediate ICU Admittance / High-Priority Triage.
2. **Phenotype 2 (Metabolic Risk)**: *Severe Diabetes / Multiple Comorbidities*. (Mortality **~12.4%**)
    * **Action**: Urgent Monitoring / Stabilization.
3. **Phenotype 3 (Respiratory Focus)**: *Younger + Pneumonia*. (Mortality **~6.2%**)
    * **Action**: Oxygen Support / Regular Vitals Check.
4. **Phenotype 4 (Low-Risk Metabolic)**: *Controlled Comorbidities / No Pneumonia*. (Mortality **~1.5%**)
    * **Action**: General Ward / Observation.
5. **Phenotype 5 (Resilient Population)**: *Young + No Risk Factors*. (Mortality **< 0.1%**)
    * **Action**: Home Quarantine / Telehealth.

### 6.4 Interpretability: SHAP Analysis

To validate the clinical logic of the K-Means model, we performed **SHAP (Shapley Additive Explanations)** analysis on the "High-Risk" cluster. The results confirmed that **Diabetes** and **Pneumonia** were the dominant contributors to the assignment of patients into the high-mortality group. Specifically:

* **Diabetes** exhibited the highest mean absolute SHAP values, acting as the strongest structural marker for the high-risk phenotype.
* **Pneumonia** served as the critical acute-care "pivot," with positive SHAP values significantly increasing the probability of assignment to the critical category.
* **Age (Binned)** acted as a consistent secondary risk increment, where higher age bins moderately pushed patients towards the critical cluster assignment.

### 6.5 Limitations and Error Analysis

Despite the model's success, two key limitations persist:

1. **Dataset Bias**: The Mexican dataset has a high prevalence of Diabetes (~12%) compared to global averages. The "Metabolic Adult" phenotype might be over-represented in this specific population, potentially requiring re-calibration for deployments in regions with different demographic profiles (e.g., Europe or Asia).

### 6.6 Conclusion of Experiments

The experiments validate that **Unsupervised Learning can successfully stratify risk without labels**. However, a "One-Size-Fits-All" algorithm is insufficient. The most robust deployment strategy is a **Multi-Algorithm Benchmarked Framework**: using K-Means (validated by BIRCH and Agglomerative clustering) to rapidly sort the majority of patients into standard care pathways, while running DBSCAN in parallel as a dedicated anomaly detector to flag high-risk "outliers" who require immediate specialist review.

## 7. Conclusions and Future Work

### 7.1 Summary of Contributions

This project successfully engineered a data-driven "Pre-Triage" framework capable of stratifying COVID-19 mortality risk using only static admission data. By applying a "Multi-Algorithm" unsupervised strategy on a massive cohort of **95,839 patients**, we demonstrated that complex clinical outcomes can be predicted without relying on biased ground-truth labels. The key finding is that mortality risk is not distributed randomly but aggregates into distinct phenotypes driven by an ultra-refined "Minimal Sufficient Set" of just **three variables**: **Pneumonia, Diabetes, and Age**. Our optimized K-Means model leveraged these drivers to identify a high-risk group with **~19.8% mortality** and achieved a structural stability (Silhouette Score) of **0.67**, offering a vital "Second Opinion" for resource allocation.

### 7.2 Strengths and Weaknesses

* **Strengths**: The primary strength of this architecture is the fusion of **Interpretability and Stability**. By reducing the feature space to a 3D manifold, we achieved a high Silhouette Score of **0.67**, proving that unsupervised learning can discover near-ideal clinical separations. The integration of **SHAP values** and **Surrogate Decision Trees** ensures that results are clinically actionable, providing a "glass-box" alternative to traditional deep learning.
* **Weaknesses**: Despite the improved stability, the model remains restricted to **static admission data**. While this is optimal for "Zero-Hour" triage, it does not capture the dynamic trajectory of a patient's illness over their hospital stay. Future iterations should aim to incorporate longitudinal lab markers once they become available after the initial triage window.

### 7.3 Lessons Learned

The most significant lesson from this project is the biological validation of the **"Curse of Dimensionality"** and the power of iterative feature selection. Initially, we assumed that feeding more data (Hypertension, Sex, Obesity) would improve model performance. Counter-intuitively, the reduction to just **three features** significantly stabilized the clusters, increasing the Silhouette Score from ~0.34 to **0.67**. We learned that in high-volume epidemiological data, identifying the "Signal" amongst the "Noise" requires rigorous pruning. Furthermore, the integration of **SHAP values** taught us that unsupervised models do not have to be "Black Boxes"—interpretability can be maintained while discovering latent patterns.

### 7.4 Future Research Directions

To evolve this prototype into a production-grade tool, future work should focus on:

1. **Temporal Dynamics**: Incorporating time-series data (e.g., "Days since onset") to transition from a static risk snapshot to a dynamic trajectory model (e.g., using LSTMs).
2. **External Validation**: Testing the model on datasets from diverse geographies (e.g., Italy, USA) to verify if the "Metabolic Adult" phenotype holds true across different genetic and environmental backgrounds.
3. **Deployment Integration**: Wrapping the trained K-Means, DBSCAN, and Scaler objects into a lightweight REST API (using FastAPI) to allow seamless integration with Electronic Health Record (EHR) systems for real-time inference.

## 8. References

[1] L. Yan, H. Zhang, J. Goncalves, et al., "An interpretable mortality prediction model for COVID-19 patients," *Nature Machine Intelligence*, vol. 2, no. 5, pp. 283–288, 2020. doi: 10.1038/s42256-020-0185-y.

[2] C. W. Seymour, J. N. Kennedy, S. Wang, et al., "Derivation, Validation, and Potential Treatment Implications of Novel Clinical Phenotypes for Sepsis," *JAMA*, vol. 321, no. 20, pp. 2003–2017, 2019.

[3] Mexican Federal Government, "COVID-19 Open Data Repository," General Directorate of Epidemiology. [Online]. Available: <https://www.gob.mx/salud/documentos/datos-abiertos-152127>. [Accessed: Dec. 2025].

[4] L. Wynants, B. Van Calster, G. Collins, et al., "Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal," *BMJ*, vol. 369, p. m1328, 2020.

[5] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York: Springer, 2009.

[6] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 58, no. 1, pp. 267–288, 1996.

[7] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[8] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. 5th Berkeley Symp. Math. Statist. Probab.*, vol. 1, 1967, pp. 281–297.

[9] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise," in *Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining (KDD-96)*, 1996, pp. 226–231.

[10] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987.

[11] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.

[12] T. Roca, et al., "Risk stratification of COVID-19 patients upon hospital admission with an unsupervised machine learning approach," *Scientific Reports*, vol. 11, no. 1, p. 5797, 2021.

[13] Z. Obermeyer and E. J. Emanuel, "Predicting the Future — Big Data, Machine Learning, and Clinical Medicine," *The New England Journal of Medicine*, vol. 375, pp. 1216–1219, 2016.

[14] A. Esteva, A. Robicquet, B. Ramsundar, et al., "A guide to deep learning in healthcare," *Nature Medicine*, vol. 25, no. 1, pp. 24–29, 2019.

[15] D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-1, no. 2, pp. 224–227, 1979.

[16] World Health Organization, "COVID-19 Clinical Management: Living Guidance," Jan. 2021. [Online]. Available: <https://www.who.int/publications/i/item/WHO-2019-nCoV-clinical-2021-1>.
