# src/core_hypothesis/constants.py
from __future__ import annotations

# =============================================================================
# CONFIG SECTION KEYS
# =============================================================================
REPRODUCIBILITY_KEY: str = "reproducibility"
ANALISIS_KEY: str = "analisis"
PATHS_KEY: str = "paths"
OUTPUT_TABLES_KEY: str = "output_tables"

# =============================================================================
# PARAMETER KEYS
# =============================================================================
SIGNIFIKANSI_ALPHA_KEY: str = "signifikansi_alpha"

# =============================================================================
# DEFAULT PATHS
# =============================================================================
DEFAULT_CONFIG_PATH: str = "config/pipeline_config.yaml"

# =============================================================================
# FILENAMES
# =============================================================================
ANTECEDENT_EFFECTS_FILENAME: str = "estimate_antecedent_effects.xlsx"
EVALUASI_HIPOTESIS_FILENAME: str = "evaluasi_hipotesis_langsung.xlsx"

# =============================================================================
# EXCEL SHEET NAMES
# =============================================================================
SHEET_COEFFICIENTS: str = "Coefficients"

# =============================================================================
# COLUMN NAMES (INPUT)
# =============================================================================
COL_VARIABEL: str = "Variabel"
COL_CONST: str = "const"
COL_B: str = "B"
COL_T_STAT: str = "t_stat"
COL_P_VALUE: str = "p_value"

# =============================================================================
# COLUMN NAMES (OUTPUT)
# =============================================================================
COL_KODE: str = "Kode"
COL_HUBUNGAN: str = "Hubungan Variabel"
COL_KOEFISIEN: str = "Koefisien (B)"
COL_T_STATISTIC: str = "t-Statistic"
COL_P_VALUE_OUT: str = "p-Value"
COL_KEPUTUSAN: str = "Keputusan"

# =============================================================================
# DOMAIN KNOWLEDGE: HYPOTHESIS MAPPING
# =============================================================================
DEFAULT_HYPOTHESIS_MAP: dict[str, tuple[str, str]] = {
    "People": ("H1", "People → Experiential Value"),
    "Process": ("H2", "Process → Experiential Value"),
    "Physical_Evidence": ("H3", "Physical Evidence → Experiential Value"),
}

# =============================================================================
# DECISION LABELS
# =============================================================================
LABEL_DITERIMA: str = "DITERIMA"
LABEL_DITOLAK: str = "DITOLAK"
LABEL_NA: str = "N/A"
