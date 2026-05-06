# src/core_synthesis/__init__.py
"""Core synthesis: data ingestion, schema alignment, and Monte Carlo generation."""

from src.core_synthesis.monte_carlo import (
    synthesize_likert_monte_carlo,
    compute_correlation_matrix,
    validate_correlation_preservation,
)
from src.core_synthesis.ingestion import run_ingestion
from src.core_synthesis.preprocessing_laten import run_preprocessing_laten
from src.core_synthesis.generator_setup import run_generator_setup
from src.core_synthesis.hierarchical_sim_laten import run_hierarchical_sim_laten
from src.core_synthesis.hierarchical_sim_likert import run_hierarchical_sim_likert
from src.core_synthesis.post_processing_likert import run_post_processing_likert
from src.core_synthesis.synthesis_demo import run_synthesis_demo
from src.core_synthesis.synthesis_text import run_synthesis_text
from src.core_synthesis.assembly_final import run_assembly_final

__all__ = [
    "synthesize_likert_monte_carlo",
    "compute_correlation_matrix",
    "validate_correlation_preservation",
    "run_ingestion",
    "run_preprocessing_laten",
    "run_generator_setup",
    "run_hierarchical_sim_laten",
    "run_hierarchical_sim_likert",
    "run_post_processing_likert",
    "run_synthesis_demo",
    "run_synthesis_text",
    "run_assembly_final",
]

