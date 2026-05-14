# tests/test_indobert_client.py
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ai_integration.constants import (
    INDOBERT_BATCH_SIZE,
    INDOBERT_MODEL_NAME,
    SENTIMENT_LABEL_MAP,
)
from src.ai_integration.indobert_client import IndoBERTClient


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset singleton before each test."""
    IndoBERTClient.reset_instance()


def test_singleton_behavior() -> None:
    """Must return same instance on multiple instantiations."""
    client1: IndoBERTClient = IndoBERTClient()
    client2: IndoBERTClient = IndoBERTClient()
    assert client1 is client2


def test_lazy_loading_does_not_load_on_init() -> None:
    """Pipeline must be None after instantiation."""
    client: IndoBERTClient = IndoBERTClient()
    assert client._pipeline is None


def test_analyze_maps_labels_correctly() -> None:
    """Must map LABEL_0/1/2 to Negative/Neutral/Positive."""
    client: IndoBERTClient = IndoBERTClient()

    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.return_value = [{"label": "LABEL_2", "score": 0.95}]

    with patch.object(client, "_load_pipeline", return_value=mock_pipeline):
        result: Dict[str, Any] = client.analyze("Sangat memuaskan")

    assert result["sentiment"] == "Positive"
    assert result["confidence"] == 0.95


def test_analyze_bulk_respects_batch_size() -> None:
    """Must slice input into batches of configured size."""
    client: IndoBERTClient = IndoBERTClient(batch_size=2)

    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.return_value = [
        {"label": "LABEL_0", "score": 0.8},
        {"label": "LABEL_2", "score": 0.9},
    ]

    texts: List[str] = ["A", "B", "C", "D"]
    with patch.object(client, "_load_pipeline", return_value=mock_pipeline):
        results: List[Dict[str, Any]] = client.analyze_bulk(texts)

    assert len(results) == 4
    # 4 items / batch_size 2 = 2 calls
    assert mock_pipeline.call_count == 2


def test_analyze_dataframe_adds_columns() -> None:
    """Must add sentiment_label and sentiment_confidence columns."""
    client: IndoBERTClient = IndoBERTClient()

    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.return_value = [
        {"label": "LABEL_0", "score": 0.85},
        {"label": "LABEL_2", "score": 0.92},
    ]

    df: pd.DataFrame = pd.DataFrame(
        {"id": [1, 2], "review": ["Layanannya buruk", "Sangat memuaskan"]}
    )

    with patch.object(client, "_load_pipeline", return_value=mock_pipeline):
        result: pd.DataFrame = client.analyze_dataframe(df, text_column="review")

    assert "sentiment_label" in result.columns
    assert "sentiment_confidence" in result.columns
    assert result.loc[0, "sentiment_label"] == "Negative"
    assert result.loc[1, "sentiment_label"] == "Positive"


def test_label_mapping_completeness() -> None:
    """Must cover all expected IndoBERT labels."""
    assert SENTIMENT_LABEL_MAP["LABEL_0"] == "Negative"
    assert SENTIMENT_LABEL_MAP["LABEL_1"] == "Neutral"
    assert SENTIMENT_LABEL_MAP["LABEL_2"] == "Positive"


def test_default_parameters() -> None:
    """Must use constants as defaults."""
    client: IndoBERTClient = IndoBERTClient()
    assert client.model_name == INDOBERT_MODEL_NAME
    assert client.batch_size == INDOBERT_BATCH_SIZE


def test_cpu_device_forced() -> None:
    """Must force device=-1 (CPU) regardless of CUDA availability."""
    client: IndoBERTClient = IndoBERTClient()

    with patch("src.ai_integration.indobert_client.AutoTokenizer") as mock_tok, \
         patch("src.ai_integration.indobert_client.AutoModelForSequenceClassification") as mock_model, \
         patch("src.ai_integration.indobert_client.pipeline") as mock_pipe:

        mock_pipe.return_value = MagicMock()

        client._load_pipeline()

        # Verify pipeline called with device=-1
        call_kwargs: Dict[str, Any] = mock_pipe.call_args[1]
        assert call_kwargs["device"] == -1
