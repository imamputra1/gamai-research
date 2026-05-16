# src/ai_integration/indobert_client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

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
        - Batch processing with tqdm to prevent UI hang and CPU throttling
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
        self._initialized = True

    @property
    def classifier(self) -> Any:
        """Memuat model ke memori CPU hanya saat pertama kali dipanggil."""
        if self._pipeline is None:
            print(f"\n[INFO] Memuat model {self.model_name} ke CPU...")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1, # -1 secara absolut memaksa eksekusi di CPU
            )
        return self._pipeline

    def map_label(self, raw_label: str) -> str:
        """Standarisasi label dari berbagai jenis model sentimen."""
        lbl = str(raw_label).lower()
        if "pos" in lbl or "label_2" in lbl:
            return "Positive"
        if "neg" in lbl or "label_0" in lbl:
            return "Negative"
        return "Neutral"

    def analyze_bulk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Eksekusi sentimen dengan mekanisme Chunking dan Progress Bar."""
        if not texts:
            return []

        results: List[Dict[str, Any]] = []

        # Menggunakan tqdm untuk iterasi batching agar terminal tidak Hang
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Inferensi Sentimen"):
            batch = texts[i : i + self.batch_size]
            
            # Mencegah error crash jika ada teks kuesioner yang terlalu panjang
            batch_results = self.classifier(
                batch, 
                truncation=True, 
                max_length=self.max_length
            )

            for text, res in zip(batch, batch_results):
                results.append(
                    {
                        "text": text,
                        "sentiment": self.map_label(res["label"]),
                        "confidence": round(float(res["score"]), 4),
                    }
                )

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = "sentiment",
    ) -> pd.DataFrame:
        """Menambahkan kolom sentimen ke DataFrame asli."""
        texts: List[str] = df[text_column].astype(str).tolist()
        results: List[Dict[str, Any]] = self.analyze_bulk(texts)

        df_out: pd.DataFrame = df.copy()
        df_out[f"{output_prefix}_label"] = [r["sentiment"] for r in results]
        df_out[f"{output_prefix}_confidence"] = [r["confidence"] for r in results]

        return df_out

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (berguna untuk testing)."""
        cls._instance = None
        cls._pipeline = None
