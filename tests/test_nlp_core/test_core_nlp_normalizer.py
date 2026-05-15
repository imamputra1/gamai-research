# tests/test_core_nlp_normalizer.py
from __future__ import annotations

import pandas as pd

from src.core_nlp.constants import DEFAULT_SLANG_DICT
from src.core_nlp.normalizer import normalize_slang


class TestSlangNormalization:
    def test_default_slang_krg_bgt(self) -> None:
        series = pd.Series(["pelayanannya krg bgt"])
        result = normalize_slang(series)
        assert result.iloc[0] == "pelayanannya kurang banget"

    def test_word_boundary_prevents_partial_replacement(self) -> None:
        series = pd.Series(["bagusnya", "kerja", "juga"])
        result = normalize_slang(series, {"jg": "juga"})
        assert result.iloc[0] == "bagusnya"
        assert result.iloc[1] == "kerja"
        assert result.iloc[2] == "juga"

    def test_custom_slang_dict(self) -> None:
        series = pd.Series(["rs bagus", "bpjs lancar"])
        custom = {"rs": "rumah sakit", "bpjs": "jaminan kesehatan"}
        result = normalize_slang(series, custom)
        assert result.iloc[0] == "rumah sakit bagus"
        assert result.iloc[1] == "jaminan kesehatan lancar"

    def test_default_dict_completeness(self) -> None:
        assert "yg" in DEFAULT_SLANG_DICT
        assert "bgt" in DEFAULT_SLANG_DICT
        assert DEFAULT_SLANG_DICT["yg"] == "yang"
        assert DEFAULT_SLANG_DICT["bgt"] == "banget"

    def test_multiple_slangs_in_one_text(self) -> None:
        series = pd.Series(["yg pelayanannya krg bgt jg"])
        result = normalize_slang(series)
        assert result.iloc[0] == "yang pelayanannya kurang banget juga"

    def test_output_column_name(self) -> None:
        series = pd.Series(["krg"])
        result = normalize_slang(series, output_column="text_normalized")
        assert result.name == "text_normalized"
