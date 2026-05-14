# src/ai_integration/indobert_client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.ai_integration.constants import (
    INDOBERT_BATCH_SIZE,
    INDOBERT_MAX_LENGTH,
    INDOBERT_MODEL_NAME,
    SENTIMENT_LABEL_MAP,
)


class IndoBERTClient:
    """Lazy-loading singleton for Indonesian sentiment analysis via IndoBERT.

    Optimized for CPU-only slim laptops:
        - torch CPU wheels (no CUDA bloat)
        - .safetensors format for fast/safe RAM loading
        - Batch processing with moderate batch_size to prevent throttling
        - Model (~400MB) loaded ONLY on first analyze() call
    """

    _instance: Optional["IndoBERTClient"] = None
    _pipeline: Any | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "IndoBERTClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = INDOBERT_MODEL_NAME,
        batch_size: int = INDOBERT_BATCH_SIZE,
        max_length: int = INDOBERT_MAX_LENGTH,
    ) -> None:
        if hasattr(self, "_initialized"):
            return

        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.max_length: int = max_length
        self._initialized: bool = True

    def _load_pipeline(self) -> Any:
        """Lazy-load the transformers pipeline on first use.

        Forces CPU device and safetensors format for lightweight execution.

        Returns:
            transformers.Pipeline: Configured sentiment analysis pipeline.
        """
        if self._pipeline is None:
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,  # CPU-only
                truncation=True,
                max_length=self.max_length,
                batch_size=self.batch_size,
                torch_dtype=torch.float32,
            )
        return self._pipeline

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text.

        Args:
            text: Input text in Indonesian.

        Returns:
            dict: {"text": str, "sentiment": str, "confidence": float}.
        """
        classifier = self._load_pipeline()
        result = classifier(text)[0]

        raw_label: str = str(result["label"])
        confidence: float = float(result["score"])

        return {
            "text": text,
            "sentiment": SENTIMENT_LABEL_MAP.get(raw_label, raw_label),
            "confidence": round(confidence, 4),
        }

    def analyze_bulk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of texts with CPU batch processing.

        Uses the pipeline's internal batching (batch_size set at init)
        to prevent CPU throttling / OOM on 16GB RAM slim laptops.

        Args:
            texts: List of input texts.

        Returns:
            list: List of sentiment dictionaries.
        """
        classifier = self._load_pipeline()
        results: List[Dict[str, Any]] = []

        # Pipeline handles internal batching; we chunk to control memory pressure
        for i in range(0, len(texts), self.batch_size):
            batch: List[str] = texts[i : i + self.batch_size]
            batch_results = classifier(batch)

            for text, res in zip(batch, batch_results):
                raw_label: str = str(res["label"])
                confidence: float = float(res["score"])
                results.append(
                    {
                        "text": text,
                        "sentiment": SENTIMENT_LABEL_MAP.get(raw_label, raw_label),
                        "confidence": round(confidence, 4),
                    }
                )

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = "sentiment",
    ) -> pd.DataFrame:
        """Add sentiment columns to an existing DataFrame.

        Args:
            df: Source dataframe.
            text_column: Column containing text to analyze.
            output_prefix: Prefix for new columns.

        Returns:
            pd.DataFrame: Copy with sentiment and confidence columns added.
        """
        texts: List[str] = df[text_column].astype(str).tolist()
        results: List[Dict[str, Any]] = self.analyze_bulk(texts)

        df_out: pd.DataFrame = df.copy()
        df_out[f"{output_prefix}_label"] = [r["sentiment"] for r in results]
        df_out[f"{output_prefix}_confidence"] = [r["confidence"] for r in results]

        return df_out

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._pipeline = None
