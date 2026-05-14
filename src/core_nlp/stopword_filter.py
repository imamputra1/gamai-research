# src/core_nlp/stopword_filter.py
from __future__ import annotations

from typing import List, Optional, Set

import pandas as pd

from src.core_nlp.constants import DEFAULT_STOPWORDS


def remove_stopwords(
    series: pd.Series,
    custom_stopwords: Optional[List[str]] = None,
    output_column: Optional[str] = None,
) -> pd.Series:
    """15.1.3: Stopwords Removal (Penyaringan Kata Hubung).

    Tokenization & set-intersection filtering. Mendukung custom stopwords
    spesifik konteks penelitian.

    Args:
        series: pd.Series containing normalized text.
        custom_stopwords: Additional stopwords specific to research context.
        output_column: Optional name for the returned series.

    Returns:
        pd.Series: Filtered text series.
    """
    stopwords: Set[str] = DEFAULT_STOPWORDS.copy()
    if custom_stopwords:
        stopwords.update(custom_stopwords)

    def _filter(text: str) -> str:
        words: List[str] = text.split()
        filtered: List[str] = [w for w in words if w not in stopwords]
        return " ".join(filtered)

    result: pd.Series = series.apply(_filter)
    if output_column:
        result.name = output_column
    return result
