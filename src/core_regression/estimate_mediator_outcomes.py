# src/core_regression/estimate_mediator_outcomes.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

from src.core_regression._core import (
    build_design_matrix,
    evaluate_hypotheses,
    extract_model_fit,
    extract_unstandardized_coefficients,
    fit_ols_hc3,
)


def build_base_matrix(
    df: pd.DataFrame, predictor_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Construct base design matrix (X only, no mediator).

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.

    Returns:
        tuple: (X_base with const, column names).
    """
    return build_design_matrix(df, predictor_cols)


def build_full_matrix(
    df: pd.DataFrame, predictor_cols: list[str], mediator_col: str
) -> tuple[np.ndarray, list[str]]:
    """Construct full design matrix (X + M).

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.
        mediator_col: Mediator column name.

    Returns:
        tuple: (X_full with const, column names).
    """
    full_cols: list[str] = predictor_cols + [mediator_col]
    return build_design_matrix(df, full_cols)


def fit_base_model(
    df: pd.DataFrame, predictor_cols: list[str], target_col: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS base model: X -> Y without mediator.

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.
        target_col: Dependent variable column name.

    Returns:
        RegressionResultsWrapper: Fitted base model.
    """
    X_base, _ = build_base_matrix(df, predictor_cols)
    y: np.ndarray = df[target_col].values
    return fit_ols_hc3(X_base, y)


def fit_full_model(
    df: pd.DataFrame,
    predictor_cols: list[str],
    mediator_col: str,
    target_col: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS full model: X + M -> Y.

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.
        mediator_col: Mediator column name.
        target_col: Dependent variable column name.

    Returns:
        RegressionResultsWrapper: Fitted full model.
    """
    X_full, _ = build_full_matrix(df, predictor_cols, mediator_col)
    y: np.ndarray = df[target_col].values
    return fit_ols_hc3(X_full, y)


def compute_delta_r2(
    res_base: sm.regression.linear_model.RegressionResultsWrapper,
    res_full: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict[str, Any]:
    """Compute incremental R² and nested F-test.

    Args:
        res_base: Base model results.
        res_full: Full model results.

    Returns:
        dict: delta_r2, f_stat, f_pvalue, r2_base, r2_full.
    """
    r2_base: float = float(res_base.rsquared)
    r2_full: float = float(res_full.rsquared)
    delta_r2: float = r2_full - r2_base

    anova_res: pd.DataFrame = anova_lm(res_base, res_full)
    f_stat: float = float(anova_res["F"].iloc[-1])
    f_pvalue: float = float(anova_res["Pr(>F)"].iloc[-1])

    return {
        "r2_base": r2_base,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
        "f_statistic": f_stat,
        "f_pvalue": f_pvalue,
    }


def extract_full_coefficients(
    res_full: sm.regression.linear_model.RegressionResultsWrapper,
    col_names: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Extract coefficients from full model with hypothesis evaluation.

    Args:
        res_full: Full model results.
        col_names: Column names.
        alpha: Significance level.

    Returns:
        pd.DataFrame: Full model coefficients with Hipotesis column.
    """
    unstd_df: pd.DataFrame = extract_unstandardized_coefficients(
        res_full, col_names, alpha
    )
    return evaluate_hypotheses(unstd_df, alpha)


def export_mediator_outcomes_excel(
    model_fit_base: dict[str, Any],
    model_fit_full: dict[str, Any],
    delta_r2: dict[str, Any],
    coeff_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export comprehensive Sub-Structure 2 results to Excel.

    Args:
        model_fit_base: Base model fit metrics.
        model_fit_full: Full model fit metrics.
        delta_r2: Delta R² metrics.
        coeff_df: Full model coefficients.
        output_path: Target Excel file path.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([model_fit_base]).to_excel(
            writer, sheet_name="Model Summary Base", index=False
        )
        pd.DataFrame([model_fit_full]).to_excel(
            writer, sheet_name="Model Summary Full", index=False
        )
        pd.DataFrame([delta_r2]).to_excel(
            writer, sheet_name="Delta R2 Mediator", index=False
        )
        coeff_df.to_excel(writer, sheet_name="Coefficients Full", index=False)


def export_delta_r2_excel(
    delta_r2: dict[str, Any], output_path: Path
) -> None:
    """Export delta R² results to standalone Excel.

    Args:
        delta_r2: Delta R² metrics.
        output_path: Target Excel file path.
    """
    pd.DataFrame([delta_r2]).to_excel(output_path, index=False)
