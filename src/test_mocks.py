# src/test_mocks.py
from __future__ import annotations

from typing import Any, Dict, List


class MockIndoBERTClient:
    """Mock IndoBERTClient untuk testing tanpa load model 400MB.

    Dipindahkan ke src/ agar dapat di-reuse oleh semua test suite.
    """

    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1") -> None:
        self.model_name: str = model_name
        self._pipeline: Any | None = None

    def analyze(self, text: str) -> Dict[str, Any]:
        """Single text analysis mock."""
        return {
            "text": text,
            "sentiment": "Positive" if len(text.strip()) > 5 else "Neutral",
            "confidence": 0.92,
        }

    def analyze_bulk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch analysis mock."""
        return [
            {
                "text": t,
                "sentiment": "Positive" if "bagus" in t or "baik" in t else "Negative",
                "confidence": 0.9,
            }
            for t in texts
        ]

    @classmethod
    def reset_instance(cls) -> None:
        """No-op untuk kompatibilitas dengan IndoBERTClient interface."""
        pass
