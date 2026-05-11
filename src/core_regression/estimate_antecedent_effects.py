# src/core_regression/estimate_antecedent_effects.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def export_antecedent_effects_excel(
    model_fit: dict[str, Any],
    coeff_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export Sub-Structure 1 (X -> M) results to Excel.

    Args:
        model_fit: Dictionary of fit metrics.
        coeff_df: DataFrame of coefficients.
        output_path: Target Excel file path.
    """
    summary_df: pd.DataFrame = pd.DataFrame([model_fit])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Model Summary", index=False)
        coeff_df.to_excel(writer, sheet_name="Coefficients", index=False)
