# src/core_mediation/constants.py
from __future__ import annotations

# =============================================================================
# CONFIG KEYS
# =============================================================================
REPRODUCIBILITY_KEY: str = "reproducibility"
ANALISIS_KEY: str = "analisis"
ARSITEKTUR_MODEL_KEY: str = "arsitektur_model"
INDEPENDEN_KEY: str = "Independen"
MEDIATOR_KEY: str = "Mediator"
DEPENDEN_KEY: str = "Dependen"
SIGNIFIKANSI_ALPHA_KEY: str = "signifikansi_alpha"
KUNCI_RANDOM_SEED_KEY: str = "kunci_random_seed"
JUMLAH_BOOTSTRAP_KEY: str = "jumlah_bootstrap"
PATHS_KEY: str = "paths"
OUTPUT_TABLES_KEY: str = "output_tables"
ROOT_PATHS_KEY: str = "paths"
REPORT_MASTER_KEY: str = "report_master"

# =============================================================================
# DEFAULT FILENAMES
# =============================================================================
MEDIASI_BOOTSTRAP_FILENAME: str = "mediasi_bootstrap_H4.xlsx"
BOOTSTRAP_ARRAY_FILENAME: str = "bootstrap_array.npy"

# =============================================================================
# OUTPUT COLUMN NAMES
# =============================================================================
COL_PREDIKTOR: str = "Prediktor"
COL_POINT_EST: str = "Point_Estimate"
COL_BOOT_SE: str = "Boot_SE"
COL_LLCI: str = "LLCI_95"
COL_ULCI: str = "ULCI_95"
COL_STATUS: str = "Status"
COL_KEPUTUSAN: str = "Keputusan"

# =============================================================================
# DECISION LABELS
# =============================================================================
LABEL_SIGNIFIKAN: str = "Signifikan"
LABEL_TIDAK_SIGNIFIKAN: str = "Tidak Signifikan"
LABEL_MEDIASI_TERBUKTI: str = "Mediasi Terbukti"
LABEL_MEDIASI_TIDAK_TERBUKTI: str = "Mediasi Tidak Terbukti"
