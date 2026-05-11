# src/core_reproducibility/directory_bootstrap.py
from __future__ import annotations

from pathlib import Path
from typing import List

from src.core_reproducibility.constants import OUTPUT_FIGURES_KEY, OUTPUT_TABLES_KEY
from src.core_reproducibility.schema import AnalisisConfig


def bootstrap_analisis_directories(analisis_config: AnalisisConfig) -> List[Path]:
    """Create the analisis output directories declared in *analisis_config*.

    Args:
        analisis_config: Canonical analisis configuration containing the
            ``paths`` subsection.

    Returns:
        List of :class:`pathlib.Path` objects for every created (or
        pre-existing) directory.
    """
    paths: Dict[str, str] = analisis_config["paths"]
    created: List[Path] = []

    for key in (OUTPUT_TABLES_KEY, OUTPUT_FIGURES_KEY):
        dir_path: Path = Path(paths[key])
        dir_path.mkdir(parents=True, exist_ok=True)
        created.append(dir_path)

    return created
