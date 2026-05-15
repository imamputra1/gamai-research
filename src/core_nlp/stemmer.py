# src/core_nlp/stemmer.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm


def stem_text(series: pd.Series, output_column: Optional[str] = None) -> pd.Series:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed: List[str] = [
        stemmer.stem(text) for text in tqdm(series, desc="Stemming", unit="doc")
    ]
    result: pd.Series = pd.Series(stemmed, index=series.index)
    if output_column:
        result.name = output_column
    return result
