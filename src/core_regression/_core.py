# src/core_regression/_core.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.core_regression.constants import (
    DEFAULT_INDICATOR_PREFIX_MAP,
    MASTER_CSV_FILENAME,
)


def load_master_data(
    report_master_dir: Path,
    indicator_prefix_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load df_report_master.csv and aggregate latent variable means.

    Args:
        report_master_dir: Directory containing the master CSV.
        indicator_prefix_map: Mapping of latent variable names to column
            prefixes. Defaults to DEFAULT_INDICATOR_PREFIX_MAP.

    Returns:
        pd.DataFrame: DataFrame with latent variable columns appended.
    """
    target: Path = report_master_dir / MASTER_CSV_FILENAME
    if not target.exists():
        raise FileNotFoundError(f"Master data not found: {target}")

    df: pd.DataFrame = pd.read_csv(target)
    prefix_map: dict[str, str] = indicator_prefix_map or DEFAULT_INDICATOR_PREFIX_MAP

    for latent, prefix in prefix_map.items():
        indicators: list[str] = [c for c in df.columns if c.startswith(prefix)]
        if latent not in df.columns and indicators:
            df[latent] = df[indicators].mean(axis=1)

    return df


def standardize_variables(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply Z-score standardization to specified columns.

    Args:
        df: Source dataframe.
        cols: Columns to standardize.

    Returns:
        pd.DataFrame: New dataframe with standardized columns.
    """
    df_out: pd.DataFrame = df.copy()
    df_out[cols] = stats.zscore(df_out[cols], nan_policy="omit")
    return df_out


def build_design_matrix(
    df: pd.DataFrame, predictor_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Construct design matrix with intercept.

    Args:
        df: Source dataframe.
        predictor_cols: Predictor column names.

    Returns:
        tuple: (X matrix with const (shape: N x K+1), column names).
    """
    X: np.ndarray = sm.add_constant(df[predictor_cols].values, has_constant="add")
    col_names: list[str] = ["const"] + predictor_cols
    return X, col_names


def fit_ols_hc3(
    X: np.ndarray, y: np.ndarray
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with HC3 heteroscedasticity-consistent covariance.

    Args:
        X: Design matrix (shape: N x K).
        y: Target vector (shape: N,).

    Returns:
        RegressionResultsWrapper: Fitted model results.
    """
    model: sm.OLS = sm.OLS(y, X)
    return model.fit(cov_type="HC3")


def extract_model_fit(
    results: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict[str, Any]:
    """Extract goodness-of-fit metrics.

    Args:
        results: Fitted OLS results.

    Returns:
        dict: R², Adj R², F-stat, F-pvalue, n_obs, df.
    """
    return {
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "f_statistic": float(results.fvalue),
        "f_pvalue": float(results.f_pvalue),
        "n_observations": int(results.nobs),
        "df_model": int(results.df_model),
        "df_residuals": int(results.df_resid),
    }


def extract_unstandardized_coefficients(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    col_names: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Extract unstandardized B, Robust SE, t, p, and confidence interval.

    Args:
        results: Fitted OLS results.
        col_names: Column names corresponding to parameters.
        alpha: Significance level (e.g. 0.05 for 95% CI).

    Returns:
        pd.DataFrame: Coefficient table (shape: K+1 x 7).
    """
    params: np.ndarray = results.params
    bse: np.ndarray = results.bse
    tvalues: np.ndarray = results.tvalues
    pvalues: np.ndarray = results.pvalues
    ci: np.ndarray = results.conf_int(alpha=alpha)

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(col_names):
        rows.append(
            {
                "Variabel": name,
                "B": float(params[i]),
                "Robust_SE": float(bse[i]),
                "t_stat": float(tvalues[i]),
                "p_value": float(pvalues[i]),
                "CI_lower_95": float(ci[i, 0]),
                "CI_upper_95": float(ci[i, 1]),
            }
        )
    return pd.DataFrame(rows)


def extract_standardized_beta(
    df_std: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Compute standardized beta via OLS on z-scored data.

    Args:
        df_std: Standardized dataframe.
        predictor_cols: Predictor column names.
        target_col: Target column name.

    Returns:
        pd.DataFrame: Standardized coefficients (shape: K+1 x 2).
    """
    X_std, col_names = build_design_matrix(df_std, predictor_cols)
    y_std: np.ndarray = stats.zscore(df_std[target_col].values, nan_policy="omit")
    results_std: sm.regression.linear_model.RegressionResultsWrapper = fit_ols_hc3(
        X_std, y_std
    )

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(col_names):
        rows.append(
            {
                "Variabel": name,
                "Beta_Standardized": float(results_std.params[i]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_hypotheses(coeff_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Evaluate H1, H2, H3: p < alpha AND B > 0 for predictors.

    Args:
        coeff_df: Coefficient dataframe.
        alpha: Significance threshold.

    Returns:
        pd.DataFrame: Coefficient table with Hipotesis column.
    """

    def _eval(row: pd.Series) -> str:
        if row["Variabel"] == "const":
            return "N/A"
        if row["p_value"] < alpha and row["B"] > 0:
            return "Diterima"
        return "Ditolak"

    coeff_df = coeff_df.copy()
    coeff_df["Hipotesis"] = coeff_df.apply(_eval, axis=1)
    return coeff_df


def merge_coefficient_tables(
    unstd_df: pd.DataFrame, std_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge unstandardized and standardized coefficient tables.

    Args:
        unstd_df: Unstandardized coefficients.
        std_df: Standardized coefficients.

    Returns:
        pd.DataFrame: Merged table (shape: K+1 x 9).
    """
    return unstd_df.merge(std_df, on="Variabel", how="left")
