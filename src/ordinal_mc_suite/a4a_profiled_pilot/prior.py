"""Use the immutable A4(a) data, scorer, head, and evaluation implementations."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "a4a_change_pilot"
BALANCED = HERE.parent / "a4a_balanced_pilot"
sys.path.insert(0, str(ORIGINAL))
