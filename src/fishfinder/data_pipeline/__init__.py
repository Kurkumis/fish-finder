"""Helpers for reading YAML configuration files."""
from pathlib import Path

import yaml

from fishfinder.utils.paths import CONFIGS_DIR


def load_classes(path: Path | None = None) -> list[dict]:
    """Load class definitions from configs/classes.yaml (or a custom path)."""
    path = path or CONFIGS_DIR / "classes.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["classes"]