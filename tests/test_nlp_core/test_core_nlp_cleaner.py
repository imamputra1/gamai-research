# tests/test_core_nlp_cleaner.py
from __future__ import annotations

import pandas as pd

from src.core_nlp.cleaner import clean_text


class TestCleaningCaseFolding:
    def test_lowercase_output(self) -> None:
        series = pd.Series(["BAIK", "Bagus", "SANGAT MEMUASKAN"])
        result = clean_text(series)
        assert result.iloc[0] == "baik"
        assert result.iloc[1] == "bagus"
        assert result.iloc[2] == "sangat memuaskan"

    def test_punctuation_replaced_with_space_not_removed(self) -> None:
        series = pd.Series(["bagus.pelayanan", "sangat;baik", "kurang,ramah"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus pelayanan"
        assert result.iloc[1] == "sangat baik"
        assert result.iloc[2] == "kurang ramah"

    def test_numbers_replaced_with_space(self) -> None:
        series = pd.Series(["pelayanan123", "1sangat2baik3"])
        result = clean_text(series)
        assert result.iloc[0] == "pelayanan"
        assert result.iloc[1] == "sangat baik"

    def test_emojis_and_symbols_removed(self) -> None:
        series = pd.Series(["bagus 😊", "sangat👍baik", "kurang@#$%ramah"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus"
        assert result.iloc[1] == "sangat baik"
        assert result.iloc[2] == "kurang ramah"

    def test_only_a_z_and_spaces_remain(self) -> None:
        series = pd.Series(["Pelayanan!!! yang@ #1 Bagus..."])
        result = clean_text(series)
        assert all(c.islower() or c.isspace() for c in result.iloc[0])
        assert result.iloc[0] == "pelayanan yang bagus"

    def test_multiple_spaces_collapsed(self) -> None:
        series = pd.Series(["bagus    sekali   !!!"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus sekali"

    def test_empty_string(self) -> None:
        series = pd.Series([""])
        result = clean_text(series)
        assert result.iloc[0] == ""

    def test_output_column_name(self) -> None:
        series = pd.Series(["BAIK"])
        result = clean_text(series, output_column="text_cleaned")
        assert result.name == "text_cleaned"
