# src/core_nlp/constants.py
from __future__ import annotations

from typing import Dict, Final, List, Set, Tuple

# ── NLP Pipeline Stages ─────────────────────────────────────────────────────
COL_TEXT_CLEANED: Final[str] = "text_cleaned"
COL_TEXT_NORMALIZED: Final[str] = "text_normalized"
COL_TEXT_FILTERED: Final[str] = "text_filtered"
COL_TEXT_FINAL: Final[str] = "text_final_preprocessed"

# ── Slang & Stopwords ───────────────────────────────────────────────────────
DEFAULT_SLANG_DICT: Final[Dict[str, str]] = {
    "yg": "yang", "bgt": "banget", "krg": "kurang", "jg": "juga",
    "dgn": "dengan", "tdk": "tidak", "dlm": "dalam", "bs": "bisa",
    "utk": "untuk", "dg": "dengan", "dr": "dari", "tp": "tapi",
    "krn": "karena", "sdh": "sudah", "blm": "belum", "bnyk": "banyak",
    "sdkt": "sedikit", "trmsk": "termasuk", "trs": "terus", "bkn": "bukan",
    "jd": "jadi", "sm": "sama", "gak": "tidak", "ga": "tidak",
    "gk": "tidak", "nggak": "tidak", "ok": "baik", "oke": "baik",
    "mantap": "bagus", "top": "bagus",
}

DEFAULT_STOPWORDS: Final[Set[str]] = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan",
    "adalah", "ini", "itu", "atau", "sebagai", "juga", "sudah",
    "sangat", "akan", "oleh", "karena", "saat", "tentang", "saja",
    "nya", "saya", "anda", "dia", "mereka", "kita", "kami", "kau",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "must",
}

# ── Frequential Analysis ───────────────────────────────────────────────────
DEFAULT_TOP_K: Final[int] = 20
DEFAULT_NGRAM_MIN_DF: Final[int] = 1
DEFAULT_TOKEN_PATTERN: Final[str] = r"(?u)\b\w\w+\b"

# ── Visualization ───────────────────────────────────────────────────────────
DEFAULT_WC_BG_COLOR: Final[str] = "white"
DEFAULT_WC_COLORMAP: Final[str] = "coolwarm"
DEFAULT_WC_WIDTH: Final[int] = 1200
DEFAULT_WC_HEIGHT: Final[int] = 800
DEFAULT_WC_DPI: Final[int] = 300
DEFAULT_CHART_FIGSIZE: Final[Tuple[int, int]] = (10, 6)
DEFAULT_CHART_DPI: Final[int] = 300
DEFAULT_CHART_ORIENTATION: Final[str] = "horizontal"

# ── Directory Paths (relative to project root) ──────────────────────────────
DIR_REPORTS_TEXT_PREPROCESSING: Final[str] = "reports/text_preprocessing"
DIR_REPORTS_FIGURES: Final[str] = "reports/figures"

# ── Sentiment Encoding ──────────────────────────────────────────────────────
SENTIMENT_SCORE_MAP: Final[Dict[str, int]] = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1,
}

# ── Column Naming Conventions ───────────────────────────────────────────────
SUFFIX_SENTIMENT: Final[str] = "_sentiment"
SUFFIX_CONFIDENCE: Final[str] = "_confidence"
SUFFIX_FINAL_TEXT: Final[str] = "_final_preprocessed"

# ── Aggregation Output Columns ──────────────────────────────────────────────
COL_OVERALL_SCORE: Final[str] = "Overall_Sentiment_Score"
COL_OVERALL_LABEL: Final[str] = "Overall_Sentiment_Label"

# ── Demographic Anchor Columns ─────────────────────────────────────────────
DEMOGRAPHIC_COLS: Final[List[str]] = [
    "Jenis Kelamin",
    "Usia",
    "Pendidikan terakhir",
    "Pekerjaan",
    "Frekuensi kunjungan ke klinik",
]

# ── Question Key ↔ Dimension Mapping ─────────────────────────────────────────
QUESTION_KEYS: Final[List[str]] = [
    "q1_nakes",
    "q2_proses",
    "q3_fasilitas",
    "q4_keseluruhan",
    "q5_alasan",
]

QUESTION_LABELS: Final[Dict[str, str]] = {
    "q1_nakes": "Tenaga Kesehatan",
    "q2_proses": "Proses Pelayanan",
    "q3_fasilitas": "Fasilitas Fisik",
    "q4_keseluruhan": "Kesan Keseluruhan",
    "q5_alasan": "Alasan & Harapan",
}

# ── Likert Scale Prefixes ──────────────────────────────────────────────────
LIKERT_PREFIXES: Final[List[str]] = ["X1_", "X2_", "X3_", "M_", "Y_"]

# ── Summary Matrix Schema ───────────────────────────────────────────────────
SUMMARY_MATRIX_COLS: Final[List[str]] = [
    "Dimensi",
    "N",
    "Positive_N",
    "Positive_Pct",
    "Neutral_N",
    "Neutral_Pct",
    "Negative_N",
    "Negative_Pct",
    "Top_3_Unigram",
    "Top_3_Bigram",
    "Top_3_Trigram",
]

# ── Default Output Paths ────────────────────────────────────────────────────
DEFAULT_OUTPUT_CSV: Final[str] = "data/02_processed/nlp_master_aggregated.csv"
DEFAULT_OUTPUT_XLSX: Final[str] = "reports/tables/nlp_summary_matrix.xlsx"
DEFAULT_SHEET_NAME: Final[str] = "Ringkasan_NLP"
