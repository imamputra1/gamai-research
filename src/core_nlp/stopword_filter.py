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
