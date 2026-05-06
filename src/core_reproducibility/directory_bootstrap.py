# src/core_reproducibility/directory_bootstrap.py
from __future__ import annotations

from pathlib import Path
from typing import Any


def bootstrap_block_d_directories(app_config: dict[str, Any]) -> list[Path]:
    block_d_paths: dict[str, str] = app_config["reproducibility"]["block_d"]["paths"]
    created: list[Path] = []

    for key in ("output_tables", "output_figures"):
        dir_path = Path(block_d_paths[key])
        dir_path.mkdir(parents=True, exist_ok=True)
        created.append(dir_path)

    return created
