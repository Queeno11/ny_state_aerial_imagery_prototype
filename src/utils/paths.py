from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Optional external override (this is the key part)
NY_IMAGERY_ROOT = os.getenv("NY_IMAGERY_ROOT")

if NY_IMAGERY_ROOT:
    NY_IMAGERY_ROOT = Path(NY_IMAGERY_ROOT)
else:
    NY_IMAGERY_ROOT = RAW_DATA_DIR / "ny_state_imagery"
