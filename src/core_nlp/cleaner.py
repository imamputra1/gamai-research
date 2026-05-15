# src/core_nlp/cleaner.py
from __future__ import annotations

from typing import Optional

import pandas as pd


def clean_text(series: pd.Series, output_column: Optional[str] = None) -> pd.Series:
    cleaned: pd.Series = series.astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    cleaned = cleaned.str.replace(r"[0-9]", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    result: pd.Series = cleaned.str.strip()
    if output_column:
        result.name = output_column
    return result
