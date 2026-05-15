# src/core_nlp/frequency_analyzer.py
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.core_nlp.constants import DEFAULT_NGRAM_MIN_DF, DEFAULT_TOKEN_PATTERN, DEFAULT_TOP_K



def get_top_ngrams(
   df: pd.DataFrame,
   text_col: str,
   n: int = 1,
   top_k: int = DEFAULT_TOP_K,
   min_df: int = DEFAULT_NGRAM_MIN_DF,
   token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> pd.DataFrame:
    texts: List[str] = df[text_col].astype(str).tolist()
    all_tokens = " ".join(texts).split()
    valid_tokens = [t for t in all_tokens if len(t) >= 2]
    if not valid_tokens:
        return pd.DataFrame(columns=["term", "frequency"])

    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        min_df=min_df,
        token_pattern=token_pattern,
        )
    sparse_matrix = vectorizer.fit_transform(texts)
    vocab: Dict[str, int] = vectorizer.vocabulary_ 
    
    freq_array = sparse_matrix.sum(axis=0).A1 
    term_freq: List[tuple[str, int]] = [
        (term, int(freq_array[idx]))
        for term, idx in vocab.items()
        ]
    term_freq.sort(key=lambda x: x[1], reverse=True)
    top_terms = term_freq[:top_k]

    return pd.DataFrame(top_terms, columns=["term", "frequency"])


def get_all_ngram_levels(
    df: pd.DataFrame,
    text_col: str,
    ngram_ranges: Optional[List[int]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[int, pd.DataFrame]:
    """Extract unigram, bigram, and trigram frequencies in one call.

    Args:
        df: DataFrame containing preprocessed text.
        text_col: Text column name.
        ngram_ranges: List of n values (default [1, 2, 3]).
        top_k: Top-k per n-gram level.

    Returns:
        Dict mapping n -> DataFrame[term, frequency].
    """
    ranges: List[int] = ngram_ranges if ngram_ranges is not None else [1, 2, 3]
    return {n: get_top_ngrams(df, text_col, n=n, top_k=top_k) for n in ranges}
