# src/core_reproducibility/constants.py
from __future__ import annotations

# =============================================================================
# CONFIG SECTION KEYS
# =============================================================================
REPRODUCIBILITY_KEY: str = "reproducibility"
ANALISIS_KEY: str = "analisis"
PATHS_KEY: str = "paths"
ROOT_PATHS_KEY: str = "paths"

# =============================================================================
# NESTED PATH KEYS (Single Source of Truth)
# =============================================================================
OUTPUT_TABLES_KEY: str = "output_tables"
OUTPUT_FIGURES_KEY: str = "output_figures"

# =============================================================================
# ROOT PATH KEYS (transitional fallback mapping)
# =============================================================================
REPORTS_TABLES_KEY: str = "reports_tables"
REPORTS_FIGURES_KEY: str = "reports_figures"

# =============================================================================
# PARAMETER KEYS
# =============================================================================
SIGNIFIKANSI_ALPHA_KEY: str = "signifikansi_alpha"
JUMLAH_BOOTSTRAP_KEY: str = "jumlah_bootstrap"
KUNCI_RANDOM_SEED_KEY: str = "kunci_random_seed"
ARSITEKTUR_MODEL_KEY: str = "arsitektur_model"

# =============================================================================
# DEFAULT FILENAMES & PATHS
# =============================================================================
DEFAULT_CONFIG_PATH: str = "config/pipeline_config.yaml"
DEFAULT_PERSIST_BASE_PATH: str = "config"
DEFAULT_ANALISIS_CONFIG_FILENAME: str = "config_analisis.json"
