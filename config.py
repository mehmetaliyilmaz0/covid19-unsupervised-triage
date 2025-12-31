import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'patient.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Column Definitions
BINARY_COLUMNS = [
    'sex', 'patient_type', 'intubated', 'pneumonia', 'pregnant',
    'diabetes', 'copd', 'asthma', 'immunosuppression', 'hypertension',
    'other_diseases', 'cardiovascular', 'obesity', 'chronic_kidney_failure',
    'smoker', 'icu'
]

SPECIAL_MAPPING_COLS = ['icu', 'intubated', 'pregnant']

# Columns to skip when filtering 97, 98, 99
COLS_TO_IGNORE_FOR_MISSING_VALS = ['age', 'date_died', 'id', 'death_date']

COMORBIDITIES = [
    'pneumonia', 'diabetes', 'copd', 'asthma', 'immunosuppression',
    'hypertension', 'other_diseases', 'cardiovascular', 'obesity',
    'chronic_kidney_failure', 'smoker'
]

CLUSTERING_FEATURES = [
    'sex', 'diabetes', 'copd', 'asthma', 'immunosuppression', 
    'hypertension', 'other_diseases', 'cardiovascular', 'obesity', 
    'chronic_kidney_failure', 'smoker', 'pneumonia', 'age_bin'
]

# Age Binning
AGE_BINS = [0, 20, 40, 60, 150]
AGE_LABELS = [0, 1, 2, 3] # Young, Adult, Senior, Elderly

# Hyperparameters
N_CLUSTERS = 5
RANDOM_STATE = 42
DT_MAX_DEPTH = 3
ELBOW_K_RANGE = range(2, 11)

# DBSCAN Hyperparameters (Initial Estimates)
# DBSCAN (Optimized in Phase 8)
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 5
