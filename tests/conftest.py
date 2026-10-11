import sys
from pathlib import Path

# make the project root importable (main.py, database.py, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
