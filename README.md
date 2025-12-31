# COVID-19 Mortality Risk Stratification: A Dual-Layer Unsupervised Learning Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Academic_Complete-success?style=for-the-badge)

## 📑 Table of Contents

1. [Introduction & Problem Statement](#introduction--problem-statement)
2. [The "Label Latency" Hypothesis](#the-label-latency-hypothesis)
3. [Repository Architecture](#repository-architecture)
4. [Dataset & Preprocessing](#dataset--preprocessing)
5. [Methodology: The "Dual-Layer" Engine](#methodology-the-dual-layer-engine)
    * [Layer 1: The Lasso Filter](#layer-1-the-lasso-filter)
    * [Layer 2: Hybrid Clustering](#layer-2-hybrid-clustering)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Configuration Parameters](#configuration-parameters)
9. [Key Results & Clinical Phenotypes](#key-results--clinical-phenotypes)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [License & Acknowledgements](#license--acknowledgements)

---

## Introduction & Problem Statement

The COVID-19 pandemic exposed a critical gap in global healthcare infrastructure: the inability to rapidly triage patients during "Zero-Hour" (initial hospital admission). Traditional severity scoring systems (like SOFA or APACHE II) rely on comprehensive laboratory panels (D-Dimer, Ferritin, CRP) which take hours to process. In resource-limited settings, clinicians need to make life-or-death decisions *instantly*.

**This project addresses the "Pre-Triage" challenge.** We engineered a machine learning system that stratifies patient mortality risk using **only static, non-invasive data** available at the moment of admission:

* **Demographics**: Age, Sex.
* **Comorbidities**: Diabetes, Hypertension, COPD, etc.

By identifying high-risk "phenotypes" without waiting for lab results, this tool serves as a scientifically grounded "Second Opinion" for frontline workers.

---

## The "Label Latency" Hypothesis

Why use **Unsupervised Learning**? Why not just train a classifier?

1. **Label Latency**: In the early weeks of a novel pandemic, we do not know who will live or die. "Survival" is a lagging indicator. Supervised models require labeled data which simply doesn't exist yet.
2. **Selection Bias**: Early "Dead" samples are often biased towards the most severe cases, failing to generalize to the asymptomatic population.
3. **The Solution**: Unsupervised Learning (Clustering) allows us to find natural structures and risk groups in the data **without** needing ground-truth labels. We let the data speak for itself.

---

## Repository Architecture

The project is structured as a production-ready Python package `covid_risk_lib` with a robust separation of concerns.

```plaintext
ml_project/
│
├── data/
│   └── raw/                    # 💾 RAW DATA (patient.csv)
│
├── docs/                       # 📄 DOCUMENTATION
│   ├── reports/                #   - Generated Analysis Reports
│   └── Academic_Paper.md       
│
├── models/                     # 🧠 SAVED MODELS (*.joblib)
│
├── output/                     # 📂 ARTIFACTS
│   └── figures/                #   - Plots and Visualizations
│
├── covid_risk_lib/             # 📦 CORE LIBRARY
│   ├── __init__.py
│   ├── clustering.py
│   └── ...
│
├── tests/                      # 🧪 TESTING
├── scripts/                    # 🛠️ UTILITIES
├── main.py                     # 🚀 ORCHESTRATOR
├── config.py                   # ⚙️ CONFIGURATION
├── requirements.txt            # 📦 DEPENDENCIES
└── README.md
```

---

## Dataset & Preprocessing

### Data Source

**Mexican Epidemiological Surveillance System** (Open Access).

* **Rows**: ~95,000 Patient Records.
* **Columns**: 20 Features (Demographics + Comorbidities).

### The "Zero-Trust" Cleaning Pipeline (`data_cleaner.py`)

Real-world clinical data is messy. We implemented a rigorous cleaning strategy:

1. **Missing Value Sanitation**: The dataset uses integer codes `97` (Not Applicable), `98` (Ignored), and `99` (Unknown) to denote missingness.
    * *Strategy*: **Complete Case Analysis**. Rows containing these codes in critical fields (Diabetes, Sex, Age) are dropped. This removes <1.5% of data while preserving statistical integrity.
2. **Binary Standardization**:
    * *Raw Format*: `{1: 'Yes', 2: 'No'}`.
    * *Processed*: `{1: True, 0: False}`. This is crucial for sparse vector mathematics (dot products).
3. **Impossibility Checks**: Removing rows where `date_died` < `date_symptoms` (Data entry errors).

---

## Methodology: The "Dual-Layer" Engine

To combat the "Curse of Dimensionality" inherent in sparse medical data, we designed a two-stage pipeline.

### Layer 1: The Lasso Filter

We utilize **L1-Regularized Logistic Regression** (Lasso) not for prediction, but for **Feature Selection**.

* **Concept**: L1 regularization adds a penalty equivalent to the absolute magnitude of coefficients: $\lambda \sum |\beta_j|$.
* **Result**: This forces weak features to exactly zero.
* **Outcome**: The model pruned **10 redundant features** and identified the "Minimal Sufficient Set" of 3 features: **Pneumonia, Diabetes, Age**.

### Layer 2: Hybrid Clustering

We combine two algorithms to handle different types of risk:

1. **K-Means (The Population Stratifier)**:
    * *Target*: The 90% "Normal" distribution.
    * *Logic*: Partitions patients into $K=5$ globular clusters based on distance in the 3D feature space.
    * *Best For*: Identifying broad risk categories (Low, Medium, High).
2. **DBSCAN (The Anomaly Detector)**:
    * *Target*: The 10% "Tail" distribution.
    * *Logic*: Groups points based on density. Points in low-density regions are flagged as "Noise" (-1).
    * *Best For*: Detecting rare, complex outliers that defy standard categorization.

---

## Installation & Setup

### Prerequisites

* OS: Windows 10/11, Linux, or macOS.
* Python: Version 3.8 or newer.

### Step 1: Clone & Environment

```bash
git clone https://github.com/your-repo/ml_project.git
cd ml_project
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Dependencies

Install the scientific computing stack:

```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Running the Analysis

The entire workflow is automated in `main.py`.

```bash
python main.py
```

**Process Flow**:

1. **Ingest**: Loads `patient.csv`.
2. **Clean**: Executes `DataCleaner`.
3. **Select**: Runs LassoCV to find the 5 key features.
4. **Cluster**: Trains K-Means ($K=5$) and DBSCAN.
5. **Evaluate**: Calculates Silhouette Scores and Mortality Spread.
6. **Explain**: Generates SHAP plots to explain cluster assignments.

### Checking Outputs

* Open `pipeline.log` to see the detailed step-by-step mathematical logs.
* Check `output/figures/shap_summary_kmeans.png` to visually see which features (e.g., Pneumonia) drove the clustering decisions.

---

## Configuration Parameters

You can tweak the experiment settings in `config.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `DATA_PATH` | `data/raw/patient.csv` | Path to source data |
| `CLUSTERING_FEATURES` | `['pneumonia', 'diabetes', ...]` | Initial candidate list before Lasso |
| `DBSCAN_EPS` | `0.5` | Epsilon radius for density clustering |
| `DBSCAN_MIN_SAMPLES` | `10` | Min points to form a core cluster |
| `RANDOM_STATE` | `42` | Seed for reproducibility |

---

## Key Results & Clinical Phenotypes

The model, configured with **$K=5$**, successfully discovered five distinct clinical phenotypes with significantly different risk profiles:

| Phenotype | Characteristics | Mortality Rate | Triage |
| :--- | :--- | :--- | :--- |
| **1. The Critical** | *High Age + Pneumonia + Diabetes* | **~19.8%** | 🚨 **Red (ICU)** |
| **2. The Metabolic** | *Severe Diabetes / Multiple Comorbidities* | **~12.4%** | 🟠 **Orange (Urgent)** |
| **3. Respiratory Focus** | *Younger + Pneumonia* | **~6.2%** | 🟡 **Yellow (Monitor)** |
| **4. Low-Risk Metabolic** | *Controlled Comorbidities / No Pneumonia* | **~1.5%** | � **Green (Observation)** |
| **5. The Healthy** | *Young + No Risk Factors* | **< 0.1%** | 🟢 **Green (Home)** |

*Note: The "Healthy" cluster accounts for the largest segment of the population (~60%), heavily skewing the global mortality average downwards. Separation of the top 3 high-risk clusters is the key contribution.*

---

## Troubleshooting & FAQ

**Q: Why did Lasso remove 'Obesity'?**
A: Statistical Redundancy. Obesity is highly correlated with Diabetes and Hypertension. Once the model knows a patient has Diabetes, knowing they are also Obese provides negligible new information for mortality prediction. Lasso mathematically removed this "noise."

**Q: The script crashes with `FileNotFoundError`?**
A: Ensure `patient.csv` is in the `data/raw/` directory. If you are running from an IDE, check your "Working Directory" setting is the project root.

**Q: Why is the Silhouette Score (0.67) significant?**
A: A score of 0.67 is exceptionally high for clinical data, indicating that the 3-feature subspace has successfully isolated distinct, globular risk phenotypes with minimal overlap.

---
