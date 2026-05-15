# tests/nlp_test/test_frequency_analyzer.py
from __future__ import annotations

import pandas as pd

from src.core_nlp.frequency_analyzer import get_all_ngram_levels, get_top_ngrams


class TestGetTopNgrams:
    def test_unigram_extraction(self) -> None:
        df = pd.DataFrame({
            "text_final_preprocessed": [
                "bagus sekali pelayanan",
                "bagus ramah dokter",
                "kurang bagus",
            ]
        })
        result = get_top_ngrams(df, "text_final_preprocessed", n=1, top_k=5)
        assert list(result.columns) == ["term", "frequency"]
        assert len(result) <= 5
        assert "bagus" in result["term"].values

    def test_bigram_extraction(self) -> None:
        df = pd.DataFrame({
            "text_final_preprocessed": [
                "dokter ramah sekali",
                "dokter ramah baik",
                "perawat baik",
            ]
        })
        result = get_top_ngrams(df, "text_final_preprocessed", n=2, top_k=5)
        assert "dokter ramah" in result["term"].values
        assert result["frequency"].iloc[0] >= 2

    def test_trigram_extraction(self) -> None:
        df = pd.DataFrame({
            "text_final_preprocessed": [
                "dokter sangat ramah sekali",
                "dokter sangat ramah baik",
                "perawat sangat baik",
            ]
        })
        result = get_top_ngrams(df, "text_final_preprocessed", n=3, top_k=5)
        assert "dokter sangat ramah" in result["term"].values

    def test_top_k_limits_results(self) -> None:
        df = pd.DataFrame({
            "text_final_preprocessed": ["satu dua tiga empat lima enam tujuh delapan sembilan sepuluh"]
        })
        result = get_top_ngrams(df, "text_final_preprocessed", n=1, top_k=5)
        assert len(result) == 5

    def test_empty_text_returns_empty(self) -> None:
        df = pd.DataFrame({"text_final_preprocessed": [""]})
        result = get_top_ngrams(df, "text_final_preprocessed", n=1, top_k=5)
        assert len(result) == 0

    def test_single_char_tokens_ignored(self) -> None:
        df = pd.DataFrame({"text_final_preprocessed": ["a b c d e"]})
        result = get_top_ngrams(df, "text_final_preprocessed", n=1, top_k=10)
        assert len(result) == 0


class TestGetAllNgramLevels:
    def test_returns_all_levels(self) -> None:
        df = pd.DataFrame({
            "text_final_preprocessed": ["bagus sekali pelayanan dokter ramah"]
        })
        results = get_all_ngram_levels(df, "text_final_preprocessed", ngram_ranges=[1, 2], top_k=5)
        assert 1 in results
        assert 2 in results
        assert isinstance(results[1], pd.DataFrame)
        assert isinstance(results[2], pd.DataFrame)
