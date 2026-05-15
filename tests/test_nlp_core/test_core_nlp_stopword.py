# tests/test_core_nlp_stopword.py
from __future__ import annotations

import pandas as pd

from src.core_nlp.stopword_filter import remove_stopwords


class TestStopwordsRemoval:
    def test_default_stopwords_removed(self) -> None:
        series = pd.Series(["pelayanannya yang kurang banget"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "pelayanannya kurang banget"

    def test_custom_stopwords_removed(self) -> None:
        series = pd.Series(["rumah sakit bagus"])
        result = remove_stopwords(series, custom_stopwords=["rumah", "sakit"])
        assert result.iloc[0] == "bagus"

    def test_all_stopwords_returns_empty(self) -> None:
        series = pd.Series(["yang dan di"])
        result = remove_stopwords(series)
        assert result.iloc[0] == ""

    def test_no_stopwords_unchanged(self) -> None:
        series = pd.Series(["pelayanan kurang"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "pelayanan kurang"

    def test_multiple_sentences(self) -> None:
        series = pd.Series(["yang bagus dan ramah", "pelayanan yang sangat baik"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "bagus ramah"
        assert result.iloc[1] == "pelayanan baik"

    def test_output_column_name(self) -> None:
        series = pd.Series(["yang bagus"])
        result = remove_stopwords(series, output_column="text_filtered")
        assert result.name == "text_filtered"
