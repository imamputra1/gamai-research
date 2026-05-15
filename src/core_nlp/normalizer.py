# src/core_nlp/normalizer.py
from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

from src.core_nlp.constants import DEFAULT_SLANG_DICT


def normalize_slang(
    series: pd.Series,
    slang_dict: Optional[Dict[str, str]] = None,
    output_column: Optional[str] = None,
) -> pd.Series:
    mapping: Dict[str, str] = slang_dict if slang_dict is not None else DEFAULT_SLANG_DICT
    result: pd.Series = series.copy()
    for slang, standard in mapping.items():
        pattern: str = r"\b" + re.escape(slang) + r"\b"
        result = result.str.replace(pattern, standard, regex=True)
    if output_column:
        result.name = output_column
    return result
