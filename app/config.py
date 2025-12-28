# Configuration for ML-Powered Resume Screening System

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database configuration
DATABASE_PATH = os.path.join(DATA_DIR, 'resume_screener.db')

# Model configuration
MODEL_VERSION = '1.0'
MIN_TRAINING_SAMPLES = 50  # Minimum samples needed before training
RETRAIN_THRESHOLD = 50  # Retrain after this many new feedback samples

# Feature extraction parameters
MAX_FEATURES_TFIDF = 500  # Maximum features for TF-IDF
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
MAX_RESUME_LENGTH = 5000  # Maximum characters to process

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Classification thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to use ML prediction
HIGHLY_RECOMMENDED_THRESHOLD = 0.8
RECOMMENDED_THRESHOLD = 0.65
MAYBE_THRESHOLD = 0.5

# Logging
LOG_LEVEL = 'INFO'
