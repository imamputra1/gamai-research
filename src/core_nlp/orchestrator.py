# src/core_nlp/orchestrator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.core_nlp.cleaner import clean_text
from src.core_nlp.constants import (
    COL_TEXT_CLEANED,
    COL_TEXT_FILTERED,
    COL_TEXT_FINAL,
    COL_TEXT_NORMALIZED,
    DEFAULT_SLANG_DICT,
    DEFAULT_STOPWORDS,
)
from src.core_nlp.normalizer import normalize_slang
from src.core_nlp.stopword_filter import remove_stopwords
from src.core_nlp.stemmer import stem_text


class NLPPreprocessorOrchestrator:
    """Orchestrator NLP Preprocessing Pipeline.

    Composition-based wrapper yang menyembunyikan kerumitan 4 tahap
    preprocessing (clean → normalize → stopword → stem).

    Attributes:
        slang_dict: Mapping untuk normalisasi slang.
        custom_stopwords: Stopwords tambahan spesifik konteks.
    """

    def __init__(
        self,
        slang_dict: Optional[Dict[str, str]] = None,
        custom_stopwords: Optional[List[str]] = None,
    ) -> None:
        self.slang_dict: Dict[str, str] = slang_dict if slang_dict is not None else DEFAULT_SLANG_DICT.copy()
        self.custom_stopwords: List[str] = custom_stopwords if custom_stopwords is not None else []

    def run(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Eksekusi pipeline 4 tahap pada DataFrame.

        Args:
            df: Source DataFrame.
            text_column: Nama kolom teks mentah.

        Returns:
            pd.DataFrame: DataFrame baru dengan kolom tambahan:
                text_cleaned, text_normalized, text_filtered,
                text_final_preprocessed.
        """
        df_out: pd.DataFrame = df.copy()

        # 15.1.1: Cleaning & Case Folding
        df_out[COL_TEXT_CLEANED] = clean_text(df_out[text_column])

        # 15.1.2: Slang Normalization
        df_out[COL_TEXT_NORMALIZED] = normalize_slang(
            df_out[COL_TEXT_CLEANED], self.slang_dict
        )

        # 15.1.3: Stopwords Removal
        df_out[COL_TEXT_FILTERED] = remove_stopwords(
            df_out[COL_TEXT_NORMALIZED], self.custom_stopwords
        )

        # 15.1.4: Stemming (wajib setelah stopwords removal)
        df_out[COL_TEXT_FINAL] = stem_text(df_out[COL_TEXT_FILTERED])

        return df_out

    def run_series(self, series: pd.Series) -> pd.Series:
        """Eksekusi pipeline pada single pd.Series.

        Args:
            series: Raw text series.

        Returns:
            pd.Series: Final preprocessed text.
        """
        cleaned: pd.Series = clean_text(series)
        normalized: pd.Series = normalize_slang(cleaned, self.slang_dict)
        filtered: pd.Series = remove_stopwords(normalized, self.custom_stopwords)
        final: pd.Series = stem_text(filtered)
        return final
