# src/core_mediation/decomposition_core.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from src.core_mediation.constants import (
    COL_B,
    COL_PREDIKTOR,
    COL_SIGNIFIKANSI_BOOTSTRAP,
    COL_STATUS,
    COL_VARIABEL,
    COL_VAF_PERSEN,
    LABEL_SIGNIFIKAN,
    LABEL_TIDAK_DIHITUNG,
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


def load_bootstrap_results(filepath: Path) -> pd.DataFrame:
    """Load mediation bootstrap results for significance lookup.

    Args:
        filepath: Path to mediasi_bootstrap_H4.xlsx.

    Returns:
        pd.DataFrame: Bootstrap CI table.
    """
    return pd.read_excel(filepath)



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


def _build_significance_map(bootstrap_df: pd.DataFrame) -> Dict[str, str]:
    """Map predictor names to their bootstrap significance status.

    Args:
        bootstrap_df: Dataframe from mediasi_bootstrap_H4.xlsx.

    Returns:
        Dictionary: predictor -> status string.
    """
    return {
        str(row[COL_PREDIKTOR]): str(row[COL_STATUS])
        for _, row in bootstrap_df.iterrows()
    }


def compute_effect_decomposition(
    antecedent_coeffs: pd.DataFrame,
    full_model_coeffs: pd.DataFrame,
    predictors: List[str],
    mediator: str,
    bootstrap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute direct, indirect, total effects and VAF for each predictor.

    VAF (Variance Accounted For) is expressed as a percentage and is ONLY
    computed when the bootstrap indirect effect is significant. If the
    bootstrap result is absent or non-significant, VAF is marked as
    "Tidak Dihitung".

    Args:
        antecedent_coeffs: Coefficients from Sub-Structure 1 (X -> M).
        full_model_coeffs: Coefficients from Sub-Structure 2 full model.
        predictors: List of independent variable names.
        mediator: Mediator variable name.
        bootstrap_df: Optional bootstrap results for significance gating.

    Returns:
        pd.DataFrame: Decomposition table (shape: n_predictors x 8).
    """
    b_path: float = extract_coefficient(full_model_coeffs, mediator)

    signif_map: Dict[str, str] = (
        _build_significance_map(bootstrap_df) if bootstrap_df is not None else {}
    )

    rows: List[Dict[str, Any]] = []
    for predictor in predictors:
        a_path: float = extract_coefficient(antecedent_coeffs, predictor)
        c_prime: float = extract_coefficient(full_model_coeffs, predictor)

        indirect: float = a_path * b_path
        direct: float = c_prime
        total: float = direct + indirect

        status: str = signif_map.get(predictor, "N/A")
        is_signifikan: bool = status == LABEL_SIGNIFIKAN

        if is_signifikan and total != 0.0:
            vaf_persen: Any = round((indirect / total) * 100, 2)
        else:
            vaf_persen = LABEL_TIDAK_DIHITUNG

        rows.append(
            {
                "Variabel": predictor,
                "a_X_ke_M": round(a_path, 4),
                "b_M_ke_Y": round(b_path, 4),
                "Direct_Effect": round(direct, 4),
                "Indirect_Effect": round(indirect, 4),
                "Total_Effect": round(total, 4),
                COL_SIGNIFIKANSI_BOOTSTRAP: status,
                COL_VAF_PERSEN: vaf_persen,
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
