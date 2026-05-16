# src/__init__.py
"""Thesis Synthetic Pipeline."""

from src.core_reproducibility import initialize_workspace as initialize_workspace
from src.core_regression import estimate_antecedent_effects as estimate_antecedent_effects
from src.core_regression import estimate_mediator_outcomes as estimate_mediator_outcomes
from src.core_hypothesis import validate_hypotheses as validate_hypotheses
from src.core_mediation import compute_mediation_bootstrap as compute_mediation_bootstrap
from src.core_mediation import compute_decomposition as compute_decomposition
from src.core_visualization import run_path_diagram as run_path_diagram
from src.core_visualization import run_radar_chart as run_radar_chart
from src.core_nlp import run_nlp_frequential as run_nlp_frequential
from src.core_nlp import run_nlp_preprocessing as run_nlp_preprocessing
from src.core_nlp import run_nlp_sentiment as run_nlp_sentiment
from src.core_nlp import run_nlp_aggregation as run_nlp_aggregation
from src.core_nlp import run_nlp_llm_synthesis as run_nlp_llm_synthesis
from src.utils import load_config as load_config
from src.test_mocks import MockIndoBERTClient

__version__ = "0.1.1"

__all__ = [
    "initialize_workspace",
    "estimate_antecedent_effects",
    "estimate_mediator_outcomes",
    "validate_hypotheses",
    "compute_mediation_bootstrap",
    "compute_decomposition",
    "load_config",
    "run_path_diagram",
    "run_radar_chart",
    "run_nlp_preprocessing",
    "run_nlp_frequential",
    "run_nlp_sentiment",
    "run_nlp_aggregation",
    "run_nlp_llm_synthesis",
    "MockIndoBERTClient",
]
