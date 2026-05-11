# src/core_hypothesis/evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_coefficients_substruktur1(input_path: Path) -> pd.DataFrame:
    """Load Coefficients sheet from hasil_regresi_1.xlsx.

    Returns:
        pd.DataFrame: Coefficients table (shape: K+1 x C).
    """
    return pd.read_excel(input_path, sheet_name="Coefficients")


def filter_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out intercept row; keep only predictor variables.

    Returns:
        pd.DataFrame: Predictor rows only.
    """
    return df[df["Variabel"] != "const"].copy()


def map_hypothesis_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map variable names to formal hypothesis codes.

    Returns:
        pd.DataFrame: With Kode and Hubungan Variabel columns.
    """
    mapping: dict[str, tuple[str, str]] = {
        "People": ("H1", "People → Experiential Value"),
        "Process": ("H2", "Process → Experiential Value"),
        "Physical_Evidence": ("H3", "Physical Evidence → Experiential Value"),
    }

    df = df.copy()
    df["Kode"] = df["Variabel"].map(lambda v: mapping.get(v, ("", ""))[0])
    df["Hubungan Variabel"] = df["Variabel"].map(
        lambda v: mapping.get(v, ("", ""))[1]
    )
    return df


def compile_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compile final summary with selected columns and rounded values.

    Returns:
        pd.DataFrame: Final summary (shape: 3 x 5).
    """
    df = df.copy()
    df["Koefisien (B)"] = df["B"].round(4)
    df["t-Statistic"] = df["t_stat"].round(4)
    df["p-Value"] = df["p_value"].round(4)

    return df[["Kode", "Hubungan Variabel", "Koefisien (B)", "t-Statistic", "p-Value", "Hipotesis"]].rename(
        columns={"Hipotesis": "Keputusan"}
    )


def export_ringkasan_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Export final hypothesis summary to Excel.

    Args:
        df: Compiled summary dataframe.
        output_path: Target Excel file path.
    """
    df.to_excel(output_path, index=False)
