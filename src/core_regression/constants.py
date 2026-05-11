# src/core_regression/constants.py
from __future__ import annotations

# =============================================================================
# CONFIG SECTION KEYS
# =============================================================================
REPRODUCIBILITY_KEY: str = "reproducibility"
ANALISIS_KEY: str = "analisis"
ARSITEKTUR_MODEL_KEY: str = "arsitektur_model"

# =============================================================================
# ARCHITECTURE MODEL KEYS
# =============================================================================
INDEPENDEN_KEY: str = "Independen"
MEDIATOR_KEY: str = "Mediator"
DEPENDEN_KEY: str = "Dependen"

# =============================================================================
# PARAMETER KEYS
# =============================================================================
SIGNIFIKANSI_ALPHA_KEY: str = "signifikansi_alpha"
KUNCI_RANDOM_SEED_KEY: str = "kunci_random_seed"

# =============================================================================
# PATH KEYS
# =============================================================================
PATHS_KEY: str = "paths"
OUTPUT_TABLES_KEY: str = "output_tables"
OUTPUT_FIGURES_KEY: str = "output_figures"
ROOT_PATHS_KEY: str = "paths"
REPORT_MASTER_KEY: str = "report_master"

# =============================================================================
# DEFAULT FILENAMES
# =============================================================================
DEFAULT_ANTECEDENT_EFFECTS_FILENAME: str = "estimate_antecedent_effects.xlsx"
DEFAULT_MEDIATOR_OUTCOMES_FILENAME: str = "estimate_mediator_outcomes.xlsx"
DEFAULT_DELTA_R2_FILENAME: str = "delta_R2_mediator.xlsx"

# =============================================================================
# MASTER DATA
# =============================================================================
MASTER_CSV_FILENAME: str = "df_report_master.csv"

# =============================================================================
# DOMAIN KNOWLEDGE: INDICATOR PREFIX MAP
# =============================================================================
DEFAULT_INDICATOR_PREFIX_MAP: dict[str, str] = {
    "People": "X1_",
    "Process": "X2_",
    "Physical_Evidence": "X3_",
    "Experience_Value": "M_",
    "Minat_Kunjungan_Ulang": "Y_",
}
