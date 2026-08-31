import sys
import os

# Add project root to sys.path so backend package and predict module are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Import the FastAPI app — Vercel picks this up automatically
from backend.main import app  # noqa: F401
