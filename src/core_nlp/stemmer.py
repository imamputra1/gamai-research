# src/core_nlp/stemmer.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm


def stem_text(
    series: pd.Series,
    output_column: Optional[str] = None,
) -> pd.Series:
    """15.1.4: Stemming Bahasa Indonesia (Pemotongan Imbuhan).

    Mengembalikan kata bervariasi ke akar kata dasar menggunakan Sastrawi.
    Wajib dijalankan SETELAH stopwords removal.

    Kritis: tqdm progress tracking untuk melacak progres waktu nyata.

    Args:
        series: pd.Series containing filtered text.
        output_column: Optional name for the returned series.

    Returns:
        pd.Series: Stemmed text series.
    """
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed: List[str] = [
        stemmer.stem(text) for text in tqdm(series, desc="Stemming", unit="doc")
    ]

    result: pd.Series = pd.Series(stemmed, index=series.index)
    if output_column:
        result.name = output_column
    return result
