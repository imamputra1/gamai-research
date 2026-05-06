# src/utils/config_loader.py
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/pipeline_config.yaml") -> dict[str, Any]:
    """Load and parse YAML configuration.

    Args:
        path: Relative or absolute path to YAML config.

    Returns:
        dict[str, Any]: Nested dictionary with pipeline parameters.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
