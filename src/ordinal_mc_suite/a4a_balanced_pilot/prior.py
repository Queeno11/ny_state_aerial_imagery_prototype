"""Locate the original, source-hashed pilot without editing it."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "a4a_change_pilot"
sys.path.insert(0, str(ORIGINAL))
