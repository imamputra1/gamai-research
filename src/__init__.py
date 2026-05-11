# src/__init__.py
"""Thesis Synthetic Pipeline."""
from src.core_reproducibility import initialize_workspace
from src.core_regression import run_fase_12_1, run_fase_12_2
from src.core_hypothesis import run_fase_13_1
from src.core_mediation import run_fase_13_2
from src.utils import load_config

__version__ = "0.1.0"
__all__ = [
    "initialize_workspace",
    "run_fase_12_1",
    "run_fase_12_2",
    "run_fase_13_1",
    "run_fase_13_2",
    "load_config",
]
