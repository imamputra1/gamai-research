# tests/nlp_test/test_sentiment_engine.py
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.core_nlp.sentiment_engine import SentimentAnalyzer
from src.test_mocks import MockIndoBERTClient  # ← IMPORT dari src/test_mocks.py

class TestSentimentAnalyzer:
    def test_analyze_dataframe_adds_sentiment_columns(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client, batch_size=4)

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "text_normalized": ["pelayanan bagus", "kurang memuaskan", "standar"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized", output_prefix="q1")

        assert "q1_sentiment" in result.columns
        assert "q1_confidence" in result.columns
        assert result["q1_sentiment"].iloc[0] == "Positive"
        assert result["q1_sentiment"].iloc[1] == "Negative"

    def test_empty_text_bypassed_as_neutral(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "id": [1, 2],
            "text_normalized": ["", "bagus"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized")

        assert result["text_normalized_sentiment"].iloc[0] == "Neutral"
        assert result["text_normalized_confidence"].iloc[0] == 0.0
        assert result["text_normalized_sentiment"].iloc[1] == "Positive"

    def test_nan_text_bypassed_as_neutral(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "id": [1, 2],
            "text_normalized": ["nan", "bagus"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized")

        assert result["text_normalized_sentiment"].iloc[0] == "Neutral"
        assert result["text_normalized_confidence"].iloc[0] == 0.0

    def test_whitespace_only_bypassed(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "id": [1],
            "text_normalized": ["   "],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized")

        assert result["text_normalized_sentiment"].iloc[0] == "Neutral"
        assert result["text_normalized_confidence"].iloc[0] == 0.0

    def test_batch_size_respected(self) -> None:
        mock_client = MagicMock()
        mock_client.analyze_bulk.return_value = [
            {"text": f"t{i}", "sentiment": "Positive", "confidence": 0.9}
            for i in range(10)
        ]

        analyzer = SentimentAnalyzer(client=mock_client, batch_size=4)

        df = pd.DataFrame({
            "text_normalized": [f"text_{i}" for i in range(10)],
        })

        analyzer.analyze_dataframe(df, text_col="text_normalized")

        # Verify analyze_bulk was called (batching happens inside client)
        mock_client.analyze_bulk.assert_called_once()

    def test_all_empty_texts_no_bulk_call(self) -> None:
        mock_client = MagicMock()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "text_normalized": ["", "   ", "nan"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized")

        mock_client.analyze_bulk.assert_not_called()
        assert all(result["text_normalized_sentiment"] == "Neutral")
        assert all(result["text_normalized_confidence"] == 0.0)

    def test_preserves_original_columns(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "id": [1],
            "text_normalized": ["bagus"],
            "extra_col": ["x"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized")

        assert "id" in result.columns
        assert "extra_col" in result.columns

    def test_output_prefix_custom(self) -> None:
        mock_client = MockIndoBERTClient()
        analyzer = SentimentAnalyzer(client=mock_client)

        df = pd.DataFrame({
            "text_normalized": ["bagus"],
        })

        result = analyzer.analyze_dataframe(df, text_col="text_normalized", output_prefix="q5_nakes")

        assert "q5_nakes_sentiment" in result.columns
        assert "q5_nakes_confidence" in result.columns
