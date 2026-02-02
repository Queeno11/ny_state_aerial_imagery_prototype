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

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# External imagery
IMAGERY_ROOT = Path(os.getenv("IMAGERY_ROOT"))
