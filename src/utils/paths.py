from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Data
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Logs and Cache
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = Path("/home/abbatenicolas/data/cache")

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# External Datasets
IMAGERY_ROOT = Path(os.getenv("IMAGERY_ROOT"))
ACS_ROOT_DIR = Path(os.getenv("ACS_ROOT_DIR"))