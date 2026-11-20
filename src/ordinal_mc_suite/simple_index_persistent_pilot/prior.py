"""Reuse the immutable simple-index model and optimization implementations."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent / "simple_index_pilot"
sys.path.insert(0, str(SIMPLE))
