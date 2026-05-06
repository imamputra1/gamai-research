# src/utils/schema_alignment.py
import re
from typing import Dict, List

import pandas as pd


def sanitize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names by stripping whitespace and normalizing spaces.

    Args:
        df: Raw dataframe with potentially dirty headers.

    Returns:
        pd.DataFrame: Copy with sanitized column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[\n\r\t]+", " ", regex=True)
        .str.replace(r",+$", "", regex=True)
    )
    return df


def align_column_order(
    df: pd.DataFrame,
    schema_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """Reorder columns according to schema_dict regex patterns.

    Unmatched columns are dropped. Each pattern may match multiple columns.
    Columns are collected in the order patterns are defined.

    Args:
        df: Dataframe with sanitized headers.
        schema_dict: Mapping of category -> list of regex patterns.

    Returns:
        pd.DataFrame: Reordered dataframe containing only matched columns.

    Raises:
        KeyError: If any ordered column is missing from the dataframe.
    """
    ordered_cols: List[str] = []
    matched_set: set[str] = set()

    for _category, patterns in schema_dict.items():
        for pattern in patterns:
            for col in df.columns:
                if col in matched_set:
                    continue
                if re.search(pattern, col, re.IGNORECASE):
                    ordered_cols.append(col)
                    matched_set.add(col)

    missing = set(ordered_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Schema columns not found in dataframe: {missing}")

    return df[ordered_cols].copy()


def enforce_likert_types(
    df: pd.DataFrame,
    likert_cols: List[str],
    likert_min: int = 1,
    likert_max: int = 5,
) -> pd.DataFrame:
    """Map qualitative Likert text to numeric values and coerce to numeric range.

    Args:
        df: Dataframe containing Likert columns.
        likert_cols: List of column names identified as Likert scale.
        likert_min: Minimum valid scale value.
        likert_max: Maximum valid scale value.

    Returns:
        pd.DataFrame: Copy with Likert columns cast to numeric and clipped.
    """
    df = df.copy()

    text_map: Dict[str, int] = {
        "sangat setuju": 5,
        "setuju": 4,
        "netral": 3,
        "tidak setuju": 2,
        "sangat tidak setuju": 1,
        "sangat baik": 5,
        "baik": 4,
        "cukup baik": 4,
        "cukup": 3,
        "kurang": 2,
        "kurang baik": 2,
        "sangat kurang": 1,
        "sangat puas": 5,
        "puas": 4,
        "tidak puas": 2,
        "sangat tidak puas": 1,
    }

    for col in likert_cols:
        if col not in df.columns:
            continue

        lowered = df[col].astype(str).str.lower().str.strip()
        mapped = lowered.map(text_map)
        df[col] = mapped.fillna(df[col])

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].clip(likert_min, likert_max)

    return df


def impute_missing_values(
    df: pd.DataFrame,
    demo_cols: List[str],
    likert_cols: List[str],
    teks_cols: List[str],
) -> pd.DataFrame:
    """Impute missing values with category-appropriate strategies.

    Demo: mode fill. Likert: median fill. Teks: dummy string fill.

    Args:
        df: Dataframe potentially containing NaN values.
        demo_cols: Demographic column names.
        likert_cols: Likert-scale column names.
        teks_cols: Open-ended text column names.

    Returns:
        pd.DataFrame: Copy with zero missing values.
    """
    df = df.copy()

    for col in demo_cols:
        if col in df.columns and df[col].isna().any():
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Tidak diketahui"
            df[col] = df[col].fillna(fill_val)

    for col in likert_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            fill_val = median_val if pd.notna(median_val) else 3.0
            df[col] = df[col].fillna(fill_val)

    teks_defaults: Dict[str, str] = {"default": "Pelayanan sudah memadai."}
    for col in teks_cols:
        if col in df.columns and df[col].isna().any():
            default = teks_defaults.get(col, teks_defaults["default"])
            df[col] = df[col].fillna(default)

    return df
