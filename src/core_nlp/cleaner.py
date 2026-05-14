# src/core_nlp/cleaner.py
from __future__ import annotations

from typing import Optional

import pandas as pd


def clean_text(
    series: pd.Series,
    output_column: Optional[str] = None,
) -> pd.Series:
    """15.1.1: Pembersihan Teks Dasar (Cleaning & Case Folding).

    Menyeragamkan teks mentah menjadi format standar:
    - Huruf kecil
    - Tanda baca/digit/simbol → spasi (kritis: bukan langsung dihapus)
    - Collapse multiple spaces
    - Output murni [a-z\s]

    Args:
        series: pd.Series containing raw text.
        output_column: Optional name for the returned series.

    Returns:
        pd.Series: Cleaned text series.
    """
    cleaned: pd.Series = series.astype(str).str.lower()

    # Kritis: punctuation, symbols, emojis → space (bukan hapus langsung)
    cleaned = cleaned.str.replace(r"[^a-z0-9\s]", " ", regex=True)

    # Digits → space
    cleaned = cleaned.str.replace(r"[0-9]", " ", regex=True)

    # Collapse multiple spaces
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)

    result: pd.Series = cleaned.str.strip()
    if output_column:
        result.name = output_column
    return result
