# tests/test_core_nlp_orchestrator.py
from __future__ import annotations

import pandas as pd
import pytest

from src.core_nlp.orchestrator import NLPPreprocessorOrchestrator


class TestOrchestrator:
    def test_run_creates_all_columns(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2],
            "jawaban": [
                "Pelayanannya yg krg bgt.",
                "Rumah sakit ini sangat bagus dan ramah!",
            ],
        })
        orch = NLPPreprocessorOrchestrator(custom_stopwords=["rumah", "sakit"])
        result = orch.run(df, text_column="jawaban")

        assert "text_cleaned" in result.columns
        assert "text_normalized" in result.columns
        assert "text_filtered" in result.columns
        assert "text_final_preprocessed" in result.columns

    def test_run_preserves_original_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "jawaban": ["bagus"]})
        orch = NLPPreprocessorOrchestrator()
        result = orch.run(df, text_column="jawaban")
        assert "id" in result.columns
        assert "jawaban" in result.columns

    def test_run_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"jawaban": ["bagus"]})
        orch = NLPPreprocessorOrchestrator()
        original_columns = list(df.columns)
        _ = orch.run(df, text_column="jawaban")
        assert list(df.columns) == original_columns

    def test_run_series_returns_series(self) -> None:
        series = pd.Series(["Pelayanannya yg krg bgt."])
        orch = NLPPreprocessorOrchestrator()
        result = orch.run_series(series)
        assert isinstance(result, pd.Series)
        assert "layan" in result.iloc[0]

    def test_custom_slang_dict_injected(self) -> None:
        df = pd.DataFrame({"jawaban": ["ok banget"]})
        orch = NLPPreprocessorOrchestrator(slang_dict={"ok": "baik"})
        result = orch.run(df, text_column="jawaban")
        assert "baik" in result["text_normalized"].iloc[0]

    def test_custom_stopwords_injected(self) -> None:
        df = pd.DataFrame({"jawaban": ["rumah sakit bagus"]})
        orch = NLPPreprocessorOrchestrator(custom_stopwords=["rumah", "sakit"])
        result = orch.run(df, text_column="jawaban")
        assert result["text_filtered"].iloc[0] == "bagus"
