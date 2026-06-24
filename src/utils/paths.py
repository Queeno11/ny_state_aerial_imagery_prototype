from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
# Secrets (API keys) live in a separate .env.secrets so the sandbox can deny
# reading them without hiding the non-secret path vars above. When that file is
# absent or read-denied (e.g. inside the sandbox), run without secrets rather
# than crashing on import.
try:
    load_dotenv(PROJECT_ROOT / ".env.secrets")
except OSError:
    pass

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