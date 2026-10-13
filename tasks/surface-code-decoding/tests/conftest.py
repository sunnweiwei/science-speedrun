"""Put the baseline project (qecbench, baselines) on sys.path: environment/baseline/alphaqubit in the
repository, /opt/task in the verifier image."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "environment" / "baseline" / "alphaqubit"
sys.path.insert(0, str(PROJECT if PROJECT.is_dir() else ROOT))
