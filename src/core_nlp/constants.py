# src/core_nlp/constants.py
from __future__ import annotations

from typing import Dict, Set

# =============================================================================
# DEFAULT SLANG / TYPO MAPPING (15.1.2)
# =============================================================================
DEFAULT_SLANG_DICT: Dict[str, str] = {
    "yg": "yang", "bgt": "banget", "krg": "kurang", "jg": "juga",
    "dgn": "dengan", "tdk": "tidak", "dlm": "dalam", "bs": "bisa",
    "utk": "untuk", "dg": "dengan", "dr": "dari", "tp": "tapi",
    "krn": "karena", "sdh": "sudah", "blm": "belum", "bnyk": "banyak",
    "sdkt": "sedikit", "trmsk": "termasuk", "trs": "terus", "bkn": "bukan",
    "jd": "jadi", "sm": "sama", "gak": "tidak", "ga": "tidak",
    "gk": "tidak", "nggak": "tidak", "ok": "baik", "oke": "baik",
    "mantap": "bagus", "top": "bagus",
}

# =============================================================================
# DEFAULT STOPWORDS (15.1.3)
# =============================================================================
DEFAULT_STOPWORDS: Set[str] = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan",
    "adalah", "ini", "itu", "atau", "sebagai", "juga", "sudah",
    "sangat", "akan", "oleh", "karena", "saat", "tentang", "saja",
    "nya", "saya", "anda", "dia", "mereka", "kita", "kami", "kau",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "must",
}

# =============================================================================
# PREPROCESSING COLUMN NAMES (Data Manifestation Contract)
# =============================================================================
COL_TEXT_CLEANED: str = "text_cleaned"
COL_TEXT_NORMALIZED: str = "text_normalized"
COL_TEXT_FILTERED: str = "text_filtered"
COL_TEXT_FINAL: str = "text_final_preprocessed"

# =============================================================================
# FREQUENCY ANALYSIS DEFAULTS (15.2)
# =============================================================================
DEFAULT_TOP_K: int = 20
DEFAULT_NGRAM_MIN_DF: int = 1
DEFAULT_TOKEN_PATTERN: str = r"(?u)\\b\\w\\w+\\b"

# =============================================================================
# VISUALIZATION DEFAULTS (15.2.2, 15.2.3)
# =============================================================================
DEFAULT_WC_BG_COLOR: str = "white"
DEFAULT_WC_COLORMAP: str = "coolwarm"
DEFAULT_WC_WIDTH: int = 1200
DEFAULT_WC_HEIGHT: int = 800
DEFAULT_WC_DPI: int = 300

DEFAULT_CHART_FIGSIZE: tuple[int, int] = (10, 6)
DEFAULT_CHART_DPI: int = 300
DEFAULT_CHART_ORIENTATION: str = "horizontal"

# =============================================================================
# OUTPUT PATH CONTRACTS
# =============================================================================
DIR_REPORTS_TEXT_PREPROCESSING: str = "reports/text_preprocessing"
DIR_REPORTS_FIGURES: str = "reports/figures"
