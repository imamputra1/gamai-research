# src/core_nlp/aggregator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def prepare_dimension_subset(
    df: pd.DataFrame,
    key: str,
    demographic_cols: List[str],
    suffix_sentiment: str,
    suffix_confidence: str,
    suffix_final: str,
    include_demographics: bool = False,
) -> pd.DataFrame:
    """Extract sentiment triad + optional demographics for a single dimension.

    Returns:
        pd.DataFrame: Subset containing only relevant columns.
    """
    sentiment_col: str = f"{key}{suffix_sentiment}"
    confidence_col: str = f"{key}{suffix_confidence}"
    final_col: str = f"{key}{suffix_final}"

    cols: List[str] = [sentiment_col, confidence_col, final_col]
    if include_demographics:
        cols = demographic_cols + cols

    missing: List[str] = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for key '{key}': {missing}")

    return df[cols].copy()


def merge_dataframes_by_index(
    dfs: Dict[str, pd.DataFrame],
    keys: List[str],
) -> pd.DataFrame:
    """Inner-join multiple DataFrames on integer index (respondent order).

    Args:
        dfs: Mapping question_key -> DataFrame.
        keys: Ordered list of keys to merge.

    Returns:
        pd.DataFrame: Merged master DataFrame with deduplicated demographics.

    Raises:
        ValueError: If dfs is empty or lengths mismatch.
    """
    if not dfs or not keys:
        raise ValueError("DataFrame dictionary and keys must not be empty")

    lengths: Dict[str, int] = {k: len(dfs[k]) for k in keys}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"DataFrame lengths mismatch: {lengths}")

    base_key: str = keys[0]
    base_df: pd.DataFrame = dfs[base_key].copy()

    for key in keys[1:]:
        right_df: pd.DataFrame = dfs[key].copy()
        overlap: List[str] = [c for c in right_df.columns if c in base_df.columns]
        if overlap:
            right_df = right_df.drop(columns=overlap)
        base_df = base_df.merge(right_df, left_index=True, right_index=True, how="inner")

    return base_df


def compute_overall_sentiment(
    df: pd.DataFrame,
    question_keys: List[str],
    score_map: Dict[str, int],
    col_overall_score: str,
    col_overall_label: str,
) -> pd.DataFrame:
    """Compute row-wise mean sentiment score and discretized label.

    Args:
        df: Merged master DataFrame.
        question_keys: Ordered list of dimension keys.
        score_map: Mapping sentiment string -> integer score.
        col_overall_score: Output column name for numeric score.
        col_overall_label: Output column name for discretized label.

    Returns:
        pd.DataFrame: New DataFrame with two appended overall columns.
    """
    sentiment_cols: List[str] = [f"{k}_sentiment" for k in question_keys]
    missing: List[str] = [c for c in sentiment_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing sentiment columns: {missing}")

    score_df: pd.DataFrame = df[sentiment_cols].replace(score_map)
    mean_scores: pd.Series = score_df.mean(axis=1)

    def _discretize(score: float) -> str:
        if score > 0.15:
            return "Positive"
        if score < -0.15:
            return "Negative"
        return "Neutral"

    labels: pd.Series = mean_scores.apply(_discretize)

    out: pd.DataFrame = df.copy()
    out[col_overall_score] = mean_scores
    out[col_overall_label] = labels
    return out


def _extract_top_terms(freq_df: pd.DataFrame, k: int) -> str:
    """Safely extract top-k terms from a frequency DataFrame."""
    if freq_df is None or freq_df.empty:
        return ""
    if "term" in freq_df.columns:
        terms: List[str] = freq_df.head(k)["term"].astype(str).tolist()
    else:
        terms = freq_df.head(k).index.astype(str).tolist()
    return ", ".join(terms)


def build_summary_matrix(
    df: pd.DataFrame,
    question_keys: List[str],
    question_labels: Dict[str, str],
    score_map: Dict[str, int],
    summary_matrix_cols: List[str],
    ngram_data: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """Construct qualitative summary matrix per dimension.

    Args:
        df: Master DataFrame with sentiment columns.
        question_keys: Ordered dimension keys.
        question_labels: Human-readable dimension labels.
        score_map: Sentiment-to-score mapping.
        summary_matrix_cols: Enforced column order for output.
        ngram_data: Optional col_key -> {n: freq_df} injection.

    Returns:
        pd.DataFrame: Summary matrix ready for Excel export.
    """
    rows: List[Dict[str, object]] = []

    for key in question_keys:
        label: str = question_labels.get(key, key)
        sentiment_col: str = f"{key}_sentiment"

        if sentiment_col not in df.columns:
            continue

        n: int = len(df)
        counts: pd.Series = df[sentiment_col].value_counts()
        pos: int = int(counts.get("Positive", 0))
        neu: int = int(counts.get("Neutral", 0))
        neg: int = int(counts.get("Negative", 0))

        row: Dict[str, object] = {
            "Dimensi": label,
            "N": n,
            "Positive_N": pos,
            "Positive_Pct": round(pos / n * 100, 2) if n else 0.0,
            "Neutral_N": neu,
            "Neutral_Pct": round(neu / n * 100, 2) if n else 0.0,
            "Negative_N": neg,
            "Negative_Pct": round(neg / n * 100, 2) if n else 0.0,
            "Top_3_Unigram": "",
            "Top_3_Bigram": "",
            "Top_3_Trigram": "",
        }

        if ngram_data and key in ngram_data:
            gram_map: Dict[int, pd.DataFrame] = ngram_data[key]
            row["Top_3_Unigram"] = _extract_top_terms(gram_map.get(1), 3)
            row["Top_3_Bigram"] = _extract_top_terms(gram_map.get(2), 3)
            row["Top_3_Trigram"] = _extract_top_terms(gram_map.get(3), 3)

        rows.append(row)

    return pd.DataFrame(rows, columns=summary_matrix_cols)


def build_llm_synthesis_prompt(summary_df: pd.DataFrame) -> str:
    """Construct academic narrative prompt for LLM consumption.

    Returns:
        str: Formatted prompt ready for OpenRouter injection.
    """
    matrix_text: str = summary_df.to_string(index=False)
    prompt: str = (
        "Anda adalah Asisten Peneliti S2 yang ahli dalam analisis data kualitatif. "
        "Berdasarkan matriks ringkasan hasil analisis sentimen dan n-gram berikut, "
        "susunlah narasi akademik dalam Bahasa Indonesia formal yang terstruktur menjadi:\n\n"
        "1. TEMUAN UTAMA: Deskripsikan pola dominan sentimen per dimensi pelayanan.\n"
        "2. IMPLIKASI MANAJERIAL: Berikan rekomendasi strategis berbasis data.\n"
        "3. ASPEK YANG PERLU DIPERBAIKI: Identifikasi dimensi dengan sentimen negatif "
        "atau netral yang perlu intervensi.\n\n"
        f"{matrix_text}\n\n"
        "Gunakan gaya penulisan tesis (pasif, objektif, dan terukur). "
        "Hindari opini subjektif tanpa dukungan data."
    )
    return prompt


def persist_master_aggregated(df: pd.DataFrame, output_path: str) -> None:
    """Serialize master DataFrame to CSV with respondent index preserved."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True, index_label="respondent_idx", encoding="utf-8-sig")


def persist_summary_matrix(
    df: pd.DataFrame,
    output_path: str,
    sheet_name: str,
) -> None:
    """Serialize summary matrix to Excel."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
