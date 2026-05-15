# src/core_nlp/mapping_output.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.core_nlp.constants import COL_TEXT_CLEANED, COL_TEXT_FINAL, COL_TEXT_FILTERED, COL_TEXT_NORMALIZED


def build_text_mapping(df: pd.DataFrame, text_column: str) -> Dict[str, Dict[str, str]]:
    """Build mapping from original text to each preprocessing stage output.

    Args:
        df: DataFrame containing preprocessing columns.
        text_column: Original raw text column name.

    Returns:
        Dict mapping original_text -> {stage: processed_text}.
    """
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        original: str = str(row[text_column])
        mapping[original] = {
            "cleaned": str(row.get(COL_TEXT_CLEANED, "")),
            "normalized": str(row.get(COL_TEXT_NORMALIZED, "")),
            "filtered": str(row.get(COL_TEXT_FILTERED, "")),
            "final": str(row.get(COL_TEXT_FINAL, "")),
        }
    return mapping


def save_text_mapping(
    mapping: Dict[str, Dict[str, str]],
    output_path: str,
) -> None:
    """Persist text mapping to JSON.

    Args:
        mapping: Mapping dict from build_text_mapping.
        output_path: Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_text_mapping(path: str) -> Dict[str, Dict[str, str]]:
    """Load persisted text mapping from JSON.

    Args:
        path: Source file path.

    Returns:
        Mapping dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, str]] = json.load(f)
    return data


def export_mapping_csv(
    df: pd.DataFrame,
    text_column: str,
    output_path: str,
) -> None:
    """Export mapping as flat CSV for human inspection.

    Columns: original, cleaned, normalized, filtered, final.

    Args:
        df: DataFrame containing preprocessing columns.
        text_column: Original raw text column name.
        output_path: Destination CSV path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "original": df[text_column],
        "cleaned": df.get(COL_TEXT_CLEANED, ""),
        "normalized": df.get(COL_TEXT_NORMALIZED, ""),
        "filtered": df.get(COL_TEXT_FILTERED, ""),
        "final": df.get(COL_TEXT_FINAL, ""),
    })
    out_df.to_csv(output_path, index=False, encoding="utf-8")


def get_unique_tokens(df: pd.DataFrame, stage: str = "final") -> List[str]:
    """Extract unique tokens from a preprocessing stage.

    Args:
        df: DataFrame containing preprocessing columns.
        stage: Column key ('cleaned', 'normalized', 'filtered', 'final').

    Returns:
        Sorted list of unique tokens.
    """
    col_map = {
        "cleaned": COL_TEXT_CLEANED,
        "normalized": COL_TEXT_NORMALIZED,
        "filtered": COL_TEXT_FILTERED,
        "final": COL_TEXT_FINAL,
    }
    col = col_map.get(stage, COL_TEXT_FINAL)
    # PERBAIKAN: Gunakan list comprehension untuk menyaring string kosong
    all_text = " ".join(df[col].astype(str).replace("nan", "")).split()
    return sorted(list(set(token for token in all_text if token.strip())))
