# src/utils/constants.py
from __future__ import annotations

# =============================================================================
# PATH DEFAULTS
# =============================================================================
DEFAULT_SOURCE_DIRS: tuple[str, ...] = (
    "reports/analisis/tables",
    "reports/tables",
)
DEFAULT_OUTPUT_DIR: str = "reports/csv_export"
DEFAULT_OUTPUT_FILENAME: str = "consolidated_tables.csv"

# =============================================================================
# FILE EXTENSIONS
# =============================================================================
FILE_EXTENSION_XLSX: str = ".xlsx"
FILE_EXTENSION_CSV: str = ".csv"

# =============================================================================
# I/O SETTINGS
# =============================================================================
ENCODING_UTF8_SIG: str = "utf-8-sig"
CSV_INDEX: bool = False

# =============================================================================
# METADATA COLUMNS
# =============================================================================
METADATA_COL_SOURCE: str = "_source_file"
METADATA_COL_SHEET: str = "_sheet_name"
