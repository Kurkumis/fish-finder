"""Central place for project paths, so no other module hard-codes them"""
from pathlib import Path

# This file lives at <root>/src/fishfinder/utils/paths.py,
# so the project root is three levels above the file's folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INAT_RAW_DIR = RAW_DIR / "inaturalist"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"