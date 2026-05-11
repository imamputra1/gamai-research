# src/__init__.py
"""Thesis Synthetic Pipeline."""
from src.core_reproducibility import initialize_workspace
from src.core_regression import estimate_antecedent_effects, estimate_mediator_outcomes
from src.core_hypothesis import run_fase_13_1
from src.core_mediation import run_fase_13_2
from src.utils import load_config

__version__ = "0.1.1"
__all__ = [
    "initialize_workspace",
    "estimate_antecedent_effects",
    "estimate_mediator_outcomes",
    "run_fase_13_1",
    "run_fase_13_2",
    "load_config",
]
