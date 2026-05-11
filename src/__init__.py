# src/__init__.py
"""Thesis Synthetic Pipeline."""
from src.core_reproducibility import initialize_workspace
from src.core_regression import estimate_antecedent_effects, estimate_mediator_outcomes
from src.core_hypothesis import validate_hypotheses
from src.core_mediation import compute_mediation_bootstrap
from src.utils import load_config

__version__ = "0.1.1"
__all__ = [
    "initialize_workspace",
    "estimate_antecedent_effects",
    "estimate_mediator_outcomes",
    "validate_hypotheses",
    "compute_mediation_bootstrap",
    "load_config",
]
