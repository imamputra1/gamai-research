# tests/nlp_test/test_aggregator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from src.core_nlp.aggregator import (
    build_llm_synthesis_prompt,
    build_summary_matrix,
    compute_overall_sentiment,
    merge_dataframes_by_index,
    persist_master_aggregated,
    persist_summary_matrix,
    prepare_dimension_subset,
)
from src.core_nlp.constants import (
    COL_OVERALL_LABEL,
    COL_OVERALL_SCORE,
    DEMOGRAPHIC_COLS,
    QUESTION_KEYS,
    QUESTION_LABELS,
    SENTIMENT_SCORE_MAP,
    SUFFIX_CONFIDENCE,
    SUFFIX_FINAL_TEXT,
    SUFFIX_SENTIMENT,
    SUMMARY_MATRIX_COLS,
)


@pytest.fixture
def mock_dimension_dfs() -> Dict[str, pd.DataFrame]:
    """Three aligned DataFrames simulating post-sentiment preprocessed outputs."""
    base_demo = {
        "Jenis Kelamin": ["Laki-laki", "Perempuan", "Laki-laki", "Perempuan"],
        "Usia": [25, 30, 22, 28],
        "Pendidikan terakhir": ["S1", "S2", "SMA", "S1"],
        "Pekerjaan": ["Pegawai", "Wiraswasta", "Pelajar", "Pegawai"],
        "Frekuensi kunjungan ke klinik": ["1x", "2x", ">5x", "3x"],
    }
    df_q1 = pd.DataFrame({
        **base_demo,
        f"q1_nakes{SUFFIX_SENTIMENT}": ["Positive", "Positive", "Neutral", "Negative"],
        f"q1_nakes{SUFFIX_CONFIDENCE}": [0.95, 0.88, 0.72, 0.91],
        f"q1_nakes{SUFFIX_FINAL_TEXT}": ["sangat baik", "memuaskan", "cukup", "kurang"],
    })
    df_q2 = pd.DataFrame({
        **base_demo,
        f"q2_proses{SUFFIX_SENTIMENT}": ["Positive", "Neutral", "Positive", "Negative"],
        f"q2_proses{SUFFIX_CONFIDENCE}": [0.90, 0.65, 0.85, 0.80],
        f"q2_proses{SUFFIX_FINAL_TEXT}": ["cepat", "biasa", "efisien", "lambat"],
    })
    df_q3 = pd.DataFrame({
        **base_demo,
        f"q3_fasilitas{SUFFIX_SENTIMENT}": ["Neutral", "Positive", "Neutral", "Negative"],
        f"q3_fasilitas{SUFFIX_CONFIDENCE}": [0.70, 0.92, 0.60, 0.85],
        f"q3_fasilitas{SUFFIX_FINAL_TEXT}": ["standar", "nyaman", "cukup", "kotor"],
    })
    return {"q1_nakes": df_q1, "q2_proses": df_q2, "q3_fasilitas": df_q3}


class TestPrepareDimensionSubset:
    def test_extracts_correct_columns_with_demographics(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        df = prepare_dimension_subset(
            mock_dimension_dfs["q1_nakes"],
            "q1_nakes",
            DEMOGRAPHIC_COLS,
            SUFFIX_SENTIMENT,
            SUFFIX_CONFIDENCE,
            SUFFIX_FINAL_TEXT,
            include_demographics=True,
        )
        expected_cols = (
            DEMOGRAPHIC_COLS
            + [f"q1_nakes{SUFFIX_SENTIMENT}", f"q1_nakes{SUFFIX_CONFIDENCE}", f"q1_nakes{SUFFIX_FINAL_TEXT}"]
        )
        assert list(df.columns) == expected_cols
        assert len(df) == 4

    def test_extracts_without_demographics(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        df = prepare_dimension_subset(
            mock_dimension_dfs["q1_nakes"],
            "q1_nakes",
            DEMOGRAPHIC_COLS,
            SUFFIX_SENTIMENT,
            SUFFIX_CONFIDENCE,
            SUFFIX_FINAL_TEXT,
            include_demographics=False,
        )
        assert "Jenis Kelamin" not in df.columns
        assert f"q1_nakes{SUFFIX_SENTIMENT}" in df.columns

    def test_missing_column_raises(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        with pytest.raises(KeyError):
            prepare_dimension_subset(
                mock_dimension_dfs["q1_nakes"],
                "nonexistent",
                DEMOGRAPHIC_COLS,
                SUFFIX_SENTIMENT,
                SUFFIX_CONFIDENCE,
                SUFFIX_FINAL_TEXT,
            )


class TestMergeDataframesByIndex:
    def test_merges_three_dimensions(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        result = merge_dataframes_by_index(mock_dimension_dfs, list(mock_dimension_dfs.keys()))
        assert len(result) == 4
        # Demographics deduplicated (only from first df)
        assert list(result.columns).count("Jenis Kelamin") == 1
        assert list(result.columns).count("Usia") == 1
        # All sentiment triads present
        for key in mock_dimension_dfs:
            assert f"{key}{SUFFIX_SENTIMENT}" in result.columns
            assert f"{key}{SUFFIX_CONFIDENCE}" in result.columns
            assert f"{key}{SUFFIX_FINAL_TEXT}" in result.columns

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            merge_dataframes_by_index({}, [])

    def test_length_mismatch_raises(self) -> None:
        df_a = pd.DataFrame({"x": [1, 2]})
        df_b = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="mismatch"):
            merge_dataframes_by_index({"a": df_a, "b": df_b}, ["a", "b"])


class TestComputeOverallSentiment:
    def test_positive_dominant(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        merged = merge_dataframes_by_index(mock_dimension_dfs, list(mock_dimension_dfs.keys()))
        result = compute_overall_sentiment(
            merged,
            list(mock_dimension_dfs.keys()),
            SENTIMENT_SCORE_MAP,
            COL_OVERALL_SCORE,
            COL_OVERALL_LABEL,
        )
        # Row 0: Pos + Pos + Neu = (1 + 1 + 0) / 3 = 0.667 -> Positive
        assert result.loc[0, COL_OVERALL_SCORE] == pytest.approx(0.667, rel=1e-3)
        assert result.loc[0, COL_OVERALL_LABEL] == "Positive"
        # Row 3: Neg + Neg + Neg = -1.0 -> Negative
        assert result.loc[3, COL_OVERALL_SCORE] == -1.0
        assert result.loc[3, COL_OVERALL_LABEL] == "Negative"

    def test_neutral_boundary(self) -> None:
        df = pd.DataFrame({
            f"q1_nakes{SUFFIX_SENTIMENT}": ["Neutral"],
            f"q2_proses{SUFFIX_SENTIMENT}": ["Neutral"],
        })
        result = compute_overall_sentiment(
            df,
            ["q1_nakes", "q2_proses"],
            SENTIMENT_SCORE_MAP,
            COL_OVERALL_SCORE,
            COL_OVERALL_LABEL,
        )
        assert result.loc[0, COL_OVERALL_SCORE] == 0.0
        assert result.loc[0, COL_OVERALL_LABEL] == "Neutral"

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError):
            compute_overall_sentiment(
                df,
                ["missing"],
                SENTIMENT_SCORE_MAP,
                COL_OVERALL_SCORE,
                COL_OVERALL_LABEL,
            )


class TestBuildSummaryMatrix:
    def test_columns_match_schema(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        merged = merge_dataframes_by_index(mock_dimension_dfs, list(mock_dimension_dfs.keys()))
        scored = compute_overall_sentiment(
            merged,
            list(mock_dimension_dfs.keys()),
            SENTIMENT_SCORE_MAP,
            COL_OVERALL_SCORE,
            COL_OVERALL_LABEL,
        )
        result = build_summary_matrix(
            scored,
            list(mock_dimension_dfs.keys()),
            QUESTION_LABELS,
            SENTIMENT_SCORE_MAP,
            SUMMARY_MATRIX_COLS,
        )
        assert list(result.columns) == SUMMARY_MATRIX_COLS
        assert len(result) == 3
        assert result.loc[0, "Dimensi"] == "Tenaga Kesehatan"
        assert result.loc[1, "Dimensi"] == "Proses Pelayanan"
        assert result.loc[2, "Dimensi"] == "Fasilitas Fisik"

    def test_percentages_sum_100(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        merged = merge_dataframes_by_index(mock_dimension_dfs, list(mock_dimension_dfs.keys()))
        scored = compute_overall_sentiment(
            merged,
            list(mock_dimension_dfs.keys()),
            SENTIMENT_SCORE_MAP,
            COL_OVERALL_SCORE,
            COL_OVERALL_LABEL,
        )
        result = build_summary_matrix(
            scored,
            list(mock_dimension_dfs.keys()),
            QUESTION_LABELS,
            SENTIMENT_SCORE_MAP,
            SUMMARY_MATRIX_COLS,
        )
        for _, row in result.iterrows():
            total = row["Positive_Pct"] + row["Neutral_Pct"] + row["Negative_Pct"]
            assert total == 100.0

    def test_ngram_attachment(self, mock_dimension_dfs: Dict[str, pd.DataFrame]) -> None:
        merged = merge_dataframes_by_index(mock_dimension_dfs, list(mock_dimension_dfs.keys()))
        scored = compute_overall_sentiment(
            merged,
            list(mock_dimension_dfs.keys()),
            SENTIMENT_SCORE_MAP,
            COL_OVERALL_SCORE,
            COL_OVERALL_LABEL,
        )
        ngram_data = {
            "q1_nakes": {
                1: pd.DataFrame({"term": ["baik", "ramah", "cepat"], "freq": [10, 8, 5]}),
                2: pd.DataFrame({"term": ["sangat baik", "pelayanan ramah"], "freq": [5, 3]}),
                3: pd.DataFrame({"term": ["pelayanan sangat baik"], "freq": [2]}),
            }
        }
        result = build_summary_matrix(
            scored,
            list(mock_dimension_dfs.keys()),
            QUESTION_LABELS,
            SENTIMENT_SCORE_MAP,
            SUMMARY_MATRIX_COLS,
            ngram_data=ngram_data,
        )
        assert result.loc[0, "Top_3_Unigram"] == "baik, ramah, cepat"
        assert result.loc[0, "Top_3_Bigram"] == "sangat baik, pelayanan ramah"
        assert result.loc[0, "Top_3_Trigram"] == "pelayanan sangat baik"
        # Dimensions without ngram_data remain empty
        assert result.loc[1, "Top_3_Unigram"] == ""


class TestBuildLLMSynthesisPrompt:
    def test_contains_matrix_and_instructions(self) -> None:
        df = pd.DataFrame({
            "Dimensi": ["Test"],
            "N": [10],
            "Positive_Pct": [80.0],
        })
        prompt = build_llm_synthesis_prompt(df)
        assert "Asisten Peneliti S2" in prompt
        assert "TEMUAN UTAMA" in prompt
        assert "IMPLIKASI MANAJERIAL" in prompt
        assert "ASPEK YANG PERLU DIPERBAIKI" in prompt
        assert "Test" in prompt


class TestPersistMasterAggregated:
    def test_creates_csv_with_index(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        out = tmp_path / "master.csv"
        persist_master_aggregated(df, str(out))
        assert out.exists()
        loaded = pd.read_csv(out, index_col="respondent_idx")
        assert len(loaded) == 3


class TestPersistSummaryMatrix:
    def test_creates_excel(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "Dimensi": ["A", "B"],
            "N": [1, 2],
        })
        out = tmp_path / "summary.xlsx"
        persist_summary_matrix(df, str(out), "TestSheet")
        assert out.exists()
        loaded = pd.read_excel(out, sheet_name="TestSheet")
        assert len(loaded) == 2
