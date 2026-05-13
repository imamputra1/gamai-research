# src/core_mediation/decomposition_core.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.core_mediation.constants import (
    COL_B,
    COL_VARIABEL,
)


def load_coefficient_table(filepath: Path, sheet_name: str) -> pd.DataFrame:
    """Load a coefficient table from an Excel workbook.

    Args:
        filepath: Path to the .xlsx file.
        sheet_name: Target sheet name.

    Returns:
        pd.DataFrame: Coefficient table.
    """
    return pd.read_excel(filepath, sheet_name=sheet_name)


def extract_coefficient(
    df: pd.DataFrame,
    variable: str,
    value_col: str = COL_B,
    variable_col: str = COL_VARIABEL,
) -> float:
    """Extract a single unstandardized coefficient for a variable.

    Args:
        df: Coefficient dataframe.
        variable: Target variable name.
        value_col: Column containing coefficient values.
        variable_col: Column containing variable names.

    Returns:
        float: Coefficient value.

    Raises:
        KeyError: If variable is not found in the table.
    """
    mask: pd.Series = df[variable_col] == variable
    if not mask.any():
        raise KeyError(
            f"Variable '{variable}' not found in coefficient table. "
            f"Available: {df[variable_col].tolist()}."
        )
    return float(df.loc[mask, value_col].iloc[0])


def compute_effect_decomposition(
    antecedent_coeffs: pd.DataFrame,
    full_model_coeffs: pd.DataFrame,
    predictors: List[str],
    mediator: str,
) -> pd.DataFrame:
    """Compute direct, indirect, total effects and VAF for each predictor.

    Args:
        antecedent_coeffs: Coefficients from Sub-Structure 1 (X -> M).
        full_model_coeffs: Coefficients from Sub-Structure 2 full model (X,M -> Y).
        predictors: List of independent variable names.
        mediator: Mediator variable name.

    Returns:
        pd.DataFrame: Decomposition table (shape: n_predictors x 7).
    """
    b_path: float = extract_coefficient(full_model_coeffs, mediator)

    rows: List[Dict[str, Any]] = []
    for predictor in predictors:
        a_path: float = extract_coefficient(antecedent_coeffs, predictor)
        c_prime: float = extract_coefficient(full_model_coeffs, predictor)

        indirect: float = a_path * b_path
        direct: float = c_prime
        total: float = direct + indirect

        vaf: float | None = (indirect / total) if total != 0.0 else None

        rows.append(
            {
                "Variabel": predictor,
                "a_X_ke_M": round(a_path, 4),
                "b_M_ke_Y": round(b_path, 4),
                "Direct_Effect": round(direct, 4),
                "Indirect_Effect": round(indirect, 4),
                "Total_Effect": round(total, 4),
                "VAF": round(vaf, 4) if vaf is not None else "N/A",
            }
        )

    return pd.DataFrame(rows)


def export_decomposition_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Export decomposition results to Excel.

    Args:
        df: Decomposition dataframe.
        output_path: Target Excel file path.
    """
    df.to_excel(output_path, index=False)
