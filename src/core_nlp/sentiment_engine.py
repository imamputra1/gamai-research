# src/core_nlp/sentiment_engine.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.ai_integration.indobert_client import IndoBERTClient


class SentimentAnalyzer:
    """Wrapper untuk analisis sentimen menggunakan IndoBERTClient.

    Menerima instance IndoBERTClient yang sudah dimuat (dependency injection)
    untuk mencegah inisialisasi ulang model 400MB per kolom.

    Attributes:
        client: Instance IndoBERTClient yang sudah di-load.
        batch_size: Ukuran batch untuk inferencing.
    """

    def __init__(
        self,
        client: Optional[IndoBERTClient] = None,
        batch_size: int = 16,
    ) -> None:
        self.client: IndoBERTClient = client if client is not None else IndoBERTClient()
        self.batch_size: int = batch_size

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        output_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """Eksekusi batch inferencing sentimen pada DataFrame.

        Membaca dari kolom cleaned/normalized (bukan final stemmed).
        Teks kosong/NaN di-bypass dengan label 'Neutral' dan confidence 0.0.

        Args:
            df: DataFrame dengan kolom teks hasil preprocessing.
            text_col: Nama kolom teks (text_cleaned atau text_normalized).
            output_prefix: Prefix untuk kolom output. Default = text_col.

        Returns:
            pd.DataFrame: DataFrame baru dengan kolom tambahan:
                {prefix}_sentiment (str) dan {prefix}_confidence (float).
        """
        prefix: str = output_prefix if output_prefix is not None else text_col

        texts: List[str] = df[text_col].astype(str).tolist()
        sentiments: List[str] = []
        confidences: List[float] = []

        # Bypass teks kosong/NaN untuk mencegah error PyTorch
        valid_texts: List[str] = []
        valid_indices: List[int] = []
        for idx, text in enumerate(texts):
            stripped = text.strip()
            if stripped and stripped.lower() != "nan":
                valid_texts.append(stripped)
                valid_indices.append(idx)

        # Batch inferencing pada teks valid
        if valid_texts:
            batch_results: List[Dict[str, Any]] = self.client.analyze_bulk(valid_texts)

            # Build result arrays aligned with original DataFrame
            result_map: Dict[int, Dict[str, Any]] = {
                valid_indices[i]: batch_results[i]
                for i in range(len(valid_indices))
            }

            for idx in range(len(texts)):
                if idx in result_map:
                    sentiments.append(str(result_map[idx]["sentiment"]))
                    confidences.append(float(result_map[idx]["confidence"]))
                else:
                    sentiments.append("Neutral")
                    confidences.append(0.0)
        else:
            # Semua teks kosong
            sentiments = ["Neutral"] * len(texts)
            confidences = [0.0] * len(texts)

        df_out: pd.DataFrame = df.copy()
        df_out[f"{prefix}_sentiment"] = sentiments
        df_out[f"{prefix}_confidence"] = confidences

        return df_out

    def analyze_series(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Analisis sentimen pada single pd.Series.

        Args:
            series: Series teks hasil preprocessing.

        Returns:
            pd.DataFrame: DataFrame dengan kolom sentiment dan confidence.
        """
        df_temp: pd.DataFrame = pd.DataFrame({"text": series})
        return self.analyze_dataframe(df_temp, text_col="text", output_prefix="text")
