# src/core_hypothesis/evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.core_hypothesis.constants import (
    COL_B,
    COL_CONST,
    COL_HUBUNGAN,
    COL_KODE,
    COL_KOEFISIEN,
    COL_KEPUTUSAN,
    COL_P_VALUE,
    COL_P_VALUE_OUT,
    COL_T_STAT,
    COL_T_STATISTIC,
    COL_VARIABEL,
    DEFAULT_HYPOTHESIS_MAP,
    LABEL_DITERIMA,
    LABEL_DITOLAK,
    LABEL_NA,
    SHEET_COEFFICIENTS,
)


def load_coefficients(input_path: Path, sheet_name: str = SHEET_COEFFICIENTS) -> pd.DataFrame:
    """Load coefficient table from an Excel file.

    Args:
        input_path: Path to the Excel file.
        sheet_name: Target sheet name.

    Returns:
        pd.DataFrame: Coefficient table.
    """
    return pd.read_excel(input_path, sheet_name=sheet_name)


def filter_predictors(
    df: pd.DataFrame, exclude_var: str = COL_CONST
) -> pd.DataFrame:
    """Filter out intercept row; keep only predictor variables.

    Args:
        df: Coefficient dataframe.
        exclude_var: Variable name to exclude (default: const).

    Returns:
        pd.DataFrame: Predictor rows only.
    """
    return df[df[COL_VARIABEL] != exclude_var].copy()


def map_hypothesis_labels(
    df: pd.DataFrame,
    mapping: Dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Map variable names to formal hypothesis codes.

    Args:
        df: Predictor dataframe.
        mapping: Hypothesis mapping dictionary. Defaults to
            DEFAULT_HYPOTHESIS_MAP.

    Returns:
        pd.DataFrame: With Kode and Hubungan Variabel columns.
    """
    hypo_map: Dict[str, tuple[str, str]] = mapping or DEFAULT_HYPOTHESIS_MAP

    df = df.copy()
    df[COL_KODE] = df[COL_VARIABEL].map(lambda v: hypo_map.get(v, ("", ""))[0])
    df[COL_HUBUNGAN] = df[COL_VARIABEL].map(
        lambda v: hypo_map.get(v, ("", ""))[1]
    )
    return df


def evaluate_decisions(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Evaluate hypotheses using strict p < alpha and B > 0 criteria.

    Args:
        df: Predictor dataframe with p_value and B columns.
        alpha: Significance threshold (e.g. 0.05).

    Returns:
        pd.DataFrame: Dataframe with Keputusan column.
    """

    def _decide(row: pd.Series) -> str:
        if row[COL_VARIABEL] == COL_CONST:
            return LABEL_NA
        if row[COL_P_VALUE] < alpha and row[COL_B] > 0:
            return LABEL_DITERIMA
        return LABEL_DITOLAK

    df = df.copy()
    df[COL_KEPUTUSAN] = df.apply(_decide, axis=1)
    return df


def compile_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compile final summary with selected columns and rounded values.

    Args:
        df: Evaluated predictor dataframe.

    Returns:
        pd.DataFrame: Final summary (shape: N x 6).
    """
    df = df.copy()
    df[COL_KOEFISIEN] = df[COL_B].round(4)
    df[COL_T_STATISTIC] = df[COL_T_STAT].round(4)
    df[COL_P_VALUE_OUT] = df[COL_P_VALUE].round(4)

    return df[
        [
            COL_KODE,
            COL_HUBUNGAN,
            COL_KOEFISIEN,
            COL_T_STATISTIC,
            COL_P_VALUE_OUT,
            COL_KEPUTUSAN,
        ]
    ]


def export_evaluasi_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Export final hypothesis evaluation to Excel.

    Args:
        df: Compiled summary dataframe.
        output_path: Target Excel file path.
    """
    df.to_excel(output_path, index=False)
