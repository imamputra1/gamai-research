# src/__init__.py
"""Thesis Synthetic Pipeline."""

# Teknik Redundant Alias (import X as X) untuk membungkam Linter/Pylance
from src.core_reproducibility import initialize_workspace as initialize_workspace
from src.core_regression import estimate_antecedent_effects as estimate_antecedent_effects
from src.core_regression import estimate_mediator_outcomes as estimate_mediator_outcomes
from src.core_hypothesis import validate_hypotheses as validate_hypotheses
from src.core_mediation import compute_mediation_bootstrap as compute_mediation_bootstrap
from src.core_mediation import compute_decomposition as compute_decomposition
from src.core_visualization import run_path_diagram as run_path_diagram
from src.core_visualization import run_radar_chart as run_radar_chart
from src.utils import load_config as load_config

__version__ = "0.1.1"

# __all__ wajib berupa list string statis murni
__all__ = [
    "initialize_workspace",
    "estimate_antecedent_effects",
    "estimate_mediator_outcomes",
    "validate_hypotheses",
    "compute_mediation_bootstrap",
    "compute_decomposition",
    "load_config",
    "run_path_diagram",
    "run_radar_chart"
]
