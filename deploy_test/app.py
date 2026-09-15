"""Compatibility launcher for the shared FastAPI application."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from api.main import main

if __name__ == "__main__":
    main()
