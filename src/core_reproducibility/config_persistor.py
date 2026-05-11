# src/core_reproducibility/config_persistor.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigPersistor:
    """Persists configuration dictionaries to JSON files.

    Encapsulates filesystem I/O so callers remain agnostic of the underlying
    storage mechanism.
    """
    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def persist_json(self, data: dict[str, Any], filename: str) -> Path:
        """Serialize *data* to JSON and write it to *filename* under
        :attr:`base_path`.

        Args:
            data: Dictionary to persist.
            filename: Target filename (e.g. ``config_analisis.json``).

        Returns:
            Absolute path to the written file.
        """
        target: Path = self.base_path / filename
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return target
