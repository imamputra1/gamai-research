from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def load_master_data(report_master_dir: Path) -> pd.DataFrame:
    """Load df_report_master.csv and aggregate latent variable means.

    Returns:
        pd.DataFrame: DataFrame with latent variable columns appended.
    """
    target = report_master_dir / "df_report_master.csv"
    if not target.exists():
        raise FileNotFoundError(f"Master data not found: {target}")

    df = pd.read_csv(target)

    mapping: dict[str, list[str]] = {
        "People": [c for c in df.columns if c.startswith("X1_")],
        "Process": [c for c in df.columns if c.startswith("X2_")],
        "Physical_Evidence": [c for c in df.columns if c.startswith("X3_")],
        "Experience_Value": [c for c in df.columns if c.startswith("M_")],
        "Minat_Kunjungan": [c for c in df.columns if c.startswith("Y_")],
    }

    for latent, indicators in mapping.items():
        if latent not in df.columns and indicators:
            df[latent] = df[indicators].mean(axis=1)

    return df


def standardize_variables(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply Z-score standardization to specified columns.

    Returns:
        pd.DataFrame: New dataframe with standardized columns.
    """
    df_out = df.copy()
    df_out[cols] = stats.zscore(df_out[cols], nan_policy="omit")
    return df_out


def build_design_matrix(
    df: pd.DataFrame, predictor_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Construct design matrix with intercept.

    Returns:
        tuple: (X matrix with const, column names including 'const').
    """
    X = sm.add_constant(df[predictor_cols].values, has_constant="add")
    col_names = ["const"] + predictor_cols
    return X, col_names


def fit_ols_hc3(
    X: np.ndarray, y: np.ndarray
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with HC3 heteroscedasticity-consistent covariance.

    Returns:
        RegressionResultsWrapper: Fitted model results.
    """
    model = sm.OLS(y, X)
    return model.fit(cov_type="HC3")


def extract_model_fit(
    results: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict[str, Any]:
    """Extract goodness-of-fit metrics.

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
    """Extract unstandardized B, Robust SE, t, p, and 95% CI.

    Returns:
        pd.DataFrame: Coefficient table (shape: K+1 x 7).
    """
    params = results.params
    bse = results.bse
    tvalues = results.tvalues
    pvalues = results.pvalues
    ci = results.conf_int(alpha=alpha)

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

    Returns:
        pd.DataFrame: Standardized coefficients (shape: K+1 x 2).
    """
    X_std, col_names = build_design_matrix(df_std, predictor_cols)
    y_std = stats.zscore(df_std[target_col].values, nan_policy="omit")
    results_std = fit_ols_hc3(X_std, y_std)

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

    Returns:
        pd.DataFrame: Merged table (shape: K+1 x 9).
    """
    return unstd_df.merge(std_df, on="Variabel", how="left")


def export_substruktur1_excel(
    model_fit: dict[str, Any],
    coeff_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export results to Excel with Model Summary and Coefficients sheets.

    Args:
        model_fit: Dictionary of fit metrics.
        coeff_df: DataFrame of coefficients.
        output_path: Target Excel file path.
    """
    summary_df = pd.DataFrame([model_fit])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Model Summary", index=False)
        coeff_df.to_excel(writer, sheet_name="Coefficients", index=False)
