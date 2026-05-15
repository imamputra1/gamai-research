# tests/test_core_nlp_stemmer.py
from __future__ import annotations

import pandas as pd

from src.core_nlp.stemmer import stem_text


class TestStemming:
    def test_stemming_layan_variants(self) -> None:
        series = pd.Series(["pelayanan", "melayani", "dilayani"])
        result = stem_text(series)
        assert result.iloc[0] == "layan"
        assert result.iloc[1] == "layan"
        assert result.iloc[2] == "layan"

    def test_stemming_deterministic(self) -> None:
        series = pd.Series(["pelayanan kurang banget"])
        result1 = stem_text(series)
        result2 = stem_text(series)
        pd.testing.assert_series_equal(result1, result2)

    def test_stemming_common_words(self) -> None:
        series = pd.Series(["membantu", "pembantu", "dibantu"])
        result = stem_text(series)
        assert result.iloc[0] == "bantu"
        assert result.iloc[1] == "bantu"
        assert result.iloc[2] == "bantu"

    def test_empty_string_stemming(self) -> None:
        series = pd.Series([""])
        result = stem_text(series)
        assert result.iloc[0] == ""

    def test_output_column_name(self) -> None:
        series = pd.Series(["pelayanan"])
        result = stem_text(series, output_column="text_final")
        assert result.name == "text_final"
