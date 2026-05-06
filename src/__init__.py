# src/__init__.py
"""Thesis Synthetic Pipeline."""
from src.core_reproducibility import ReproducibilityOrchestrator, run_fase_12_0
from src.core_regression import SubStruktur1Orchestrator, run_fase_12_1
from src.utils import load_config

__version__ = "0.1.0"
__all__ = [
    "ReproducibilityOrchestrator",
    "SubStruktur1Orchestrator",
    "run_fase_12_0",
    "run_fase_12_1",
    "load_config",
]

"""
Utils coroutine:
    "load_config",
    "setup_logger",
    "export_dataframe_to_excel",
    "sanitize_headers",
    "align_column_order",
    "enforce_likert_types",
    "impute_missing_values",
"""

