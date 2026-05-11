# src/core_reproducibility/schema.py
from __future__ import annotations

from typing import Dict, List, TypedDict


class AnalisisPaths(TypedDict):
    output_tables: str
    output_figures: str


class AnalisisConfig(TypedDict):
    signifikansi_alpha: float
    jumlah_bootstrap: int
    kunci_random_seed: int
    arsitektur_model: Dict[str, List[str]]
    paths: AnalisisPaths
