"""All paths for Money Leak, defined with pathlib. Run modules from the project root,
for example `python -m backend.build_db` and `python -m ml_model.train`.

The five raw CSVs are READ ONLY. They live wherever they live: this module searches
./ then ./datasets/ then ./data/ then ./data/raw (and their parents) and uses the first
folder that holds all five files. Nothing is ever moved or regenerated here.
"""
from pathlib import Path

# money_leak/
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ raw data
RAW_NAMES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "transactions": "transactions.csv",
    "inventory": "inventory.csv",
    "campaigns": "campaigns.csv",
}
EXPECTED_ROWS = {
    "customers": 3000,
    "products": 250,
    "transactions": 18000,
    "inventory": 5000,
    "campaigns": 1200,
}

_CANDIDATE_DIRS = []
for _base in (BASE_DIR, BASE_DIR.parent):
    for _sub in ("", "datasets", "data", "data/raw"):
        _cand = (_base / _sub).resolve() if _sub else _base.resolve()
        if _cand not in _CANDIDATE_DIRS:
            _CANDIDATE_DIRS.append(_cand)

# last resort: any sibling project folder next to money_leak that holds all five
for _sib in sorted(p for p in BASE_DIR.parent.iterdir() if p.is_dir()):
    if _sib not in _CANDIDATE_DIRS:
        _CANDIDATE_DIRS.append(_sib)


def find_data_dir():
    """First candidate folder that contains all five CSVs, in search order."""
    for cand in _CANDIDATE_DIRS:
        if all((cand / name).exists() for name in RAW_NAMES.values()):
            return cand
    return None


DATA_DIR = find_data_dir()
if DATA_DIR is None:
    print("DATA_DIR NOT FOUND. Searched, in order:")
    for cand in _CANDIDATE_DIRS:
        missing = [n for n in RAW_NAMES.values() if not (cand / n).exists()]
        print(f"  {cand}  missing: {', '.join(missing) if missing else '-'}")
    raise SystemExit("STOP: the five raw CSVs are not present in any searched folder.")

print(f"DATA_DIR = {DATA_DIR}")

DB_DIR = BASE_DIR / "data" / "db"          # database only; never raw data
DATA_RAW_DIR = BASE_DIR / "data" / "raw"   # read-only copies of the five CSVs

BACKEND_DIR = BASE_DIR / "backend"
ML_DIR = BASE_DIR / "ml_model"             # training code and the saved bundle
MODELS_DIR = ML_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "evidence"      # metrics.json, charts and training logs
CHARTS_DIR = ARTIFACTS_DIR / "charts"
FRONTEND_DIR = BASE_DIR / "frontend"       # React + TypeScript source
FRONTEND_DIST = FRONTEND_DIR / "dist"      # built output, served by backend/main.py
DIST_DIR = FRONTEND_DIST

DB_PATH = DB_DIR / "money_leak.db"
BUNDLE_PATH = MODELS_DIR / "bundle.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RAW_FILES = {name: DATA_DIR / fname for name, fname in RAW_NAMES.items()}


def check_raw_rows():
    """Read each CSV header+rows and compare counts to EXPECTED_ROWS.

    Returns a list of (name, path, expected, actual, ok) tuples.
    """
    import pandas as pd

    out = []
    for name, path in RAW_FILES.items():
        actual = int(len(pd.read_csv(path)))
        expected = EXPECTED_ROWS[name]
        out.append((name, path, expected, actual, actual == expected))
    return out


def assert_raw_rows():
    """STOP loudly if any raw file is missing or has an unexpected row count."""
    bad = []
    print("raw row counts (expected vs actual):")
    for name, path, expected, actual, ok in check_raw_rows():
        flag = "OK" if ok else "MISMATCH"
        print(f"  {name:<13} {expected:>6} vs {actual:>6}   {flag}   {path.name}")
        if not ok:
            bad.append(f"{name} ({path.name}): expected {expected}, got {actual}")
    missing = check_raw_files()
    if missing or bad:
        raise SystemExit("STOP - raw data problem: " + "; ".join(missing + bad))
    return True

# The 5 regions. Region is an OUTPUT of the model, never an input feature.
REGIONS = ["Central", "East", "North", "South", "West"]

# Feature columns used by the profit model (region is deliberately absent).
NUM_COLS = [
    "quantity",
    "discount_pct",
    "list_price",
    "unit_cost",
    "cost_to_list",
    "supplier_lead_days",
]
CAT_COLS = ["category", "channel", "customer_segment"]

# Month split: train = months 1-9, test = months 10-12.
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TEST_MONTHS = [10, 11, 12]

RANDOM_STATE = 0


def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for d in (DATA_RAW_DIR, DB_DIR, MODELS_DIR, ARTIFACTS_DIR, CHARTS_DIR, FRONTEND_DIR, DIST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_raw_files():
    """Return the list of missing raw CSV names (empty list means all present)."""
    return [name for name, path in RAW_FILES.items() if not path.exists()]


def get_connection():
    """Open a sqlite3 connection to the project database."""
    import sqlite3

    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)
