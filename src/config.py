import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'dropouts_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'root')
}

MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'n_splits': 5
}

FEATURE_IMPORTANCE_THRESHOLD = 0.02
MODEL_PATH = 'models/ensemble_model.joblib'