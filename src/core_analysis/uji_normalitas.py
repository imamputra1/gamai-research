# src/core_analysis/09_6_1_uji_normalitas.py
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import boxcox, shapiro

from src.utils import setup_logger

logger = setup_logger("uji_normalitas")

ALPHA: float = 0.05


def load_master_report(path: Path) -> pd.DataFrame:
    """Load locked master dataset.

    Args:
        path: Path to df_report_master.csv.

    Returns:
        pd.DataFrame: Master dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Master report not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Master report loaded: {df.shape}")
    return df


def extract_super_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5 super-feature means from 25 Likert indicators.

    Args:
        df: Master dataframe (shape: 100, 35).

    Returns:
        pd.DataFrame: Super-feature dataframe (shape: 100, 5).
    """
    df_likert: pd.DataFrame = df.iloc[:, 5:30].copy()

    features: dict[str, pd.Series] = {
        "People": df_likert.iloc[:, 0:5].mean(axis=1),
        "Process": df_likert.iloc[:, 5:10].mean(axis=1),
        "Physical_Evidence": df_likert.iloc[:, 10:15].mean(axis=1),
        "Experience_Value": df_likert.iloc[:, 15:20].mean(axis=1),
        "Minat_Kunjungan_Ulang": df_likert.iloc[:, 20:25].mean(axis=1),
    }

    return pd.DataFrame(features)


def fit_ols_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS regression with constant term.

    Args:
        X: Independent variables dataframe.
        y: Dependent variable series.

    Returns:
        RegressionResultsWrapper: Fitted OLS model.
    """
    X_sm: pd.DataFrame = sm.add_constant(X)
    model: sm.regression.linear_model.RegressionResultsWrapper = sm.OLS(
        y, X_sm
    ).fit()
    return model


def apply_boxcox_if_needed(
    residuals: np.ndarray,
    y: pd.Series,
    X: pd.DataFrame,
) -> tuple[np.ndarray, sm.regression.linear_model.RegressionResultsWrapper, str]:
    """Apply Box-Cox transformation if residuals fail Shapiro-Wilk.

    Args:
        residuals: Original OLS residuals.
        y: Original dependent variable.
        X: Independent variables.

    Returns:
        tuple: (residuals, model, transformation_note).
    """
    stat: float
    p_value: float
    stat, p_value = shapiro(residuals)

    if p_value > ALPHA:
        logger.info(
            f"Shapiro-Wilk (raw): stat={stat:.4f}, p={p_value:.4f} -> Normal"
        )
        return residuals, fit_ols_model(X, y), "None (raw data)"

    logger.warning(
        f"Shapiro-Wilk (raw): stat={stat:.4f}, p={p_value:.4f} -> Not Normal. "
        f"Applying Box-Cox transformation..."
    )

    # Box-Cox requires strictly positive values
    y_shifted: pd.Series = y + 1.0
    y_boxcox: np.ndarray
    lambda_val: float
    y_boxcox, lambda_val = boxcox(y_shifted)

    model_bc: sm.regression.linear_model.RegressionResultsWrapper = fit_ols_model(
        X, pd.Series(y_boxcox, index=y.index)
    )
    residuals_bc: np.ndarray = model_bc.resid

    stat_bc: float
    p_bc: float
    stat_bc, p_bc = shapiro(residuals_bc)

    if p_bc > ALPHA:
        logger.info(
            f"Box-Cox (lambda={lambda_val:.4f}): stat={stat_bc:.4f}, "
            f"p={p_bc:.4f} -> Normal"
        )
        return residuals_bc, model_bc, f"Box-Cox (lambda={lambda_val:.4f})"
    else:
        logger.warning(
            f"Box-Cox still not normal: stat={stat_bc:.4f}, p={p_bc:.4f}. "
            f"Proceeding with raw residuals for reporting."
        )
        return residuals, fit_ols_model(X, y), "None (transformation failed)"


def export_qq_plot(
    residuals: np.ndarray,
    output_path: Path,
    title: str = "Normal Q-Q Plot of OLS Residuals",
) -> None:
    """Generate and save QQ plot.

    Args:
        residuals: Residual array.
        output_path: Destination file path.
        title: Plot title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"QQ Plot exported: {output_path}")


def run_uji_normalitas(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_path: Path = Path("reports/figures/qq_plot_residuals.png"),
) -> dict[str, Any]:
    """Execute full normality test pipeline: OLS, Shapiro-Wilk, QQ-Plot.

    Args:
        master_path: Path to locked master dataset.
        output_path: Path for QQ plot PNG.

    Returns:
        dict: Test results with stat, p_value, status, and transformation note.
    """
    df = load_master_report(master_path)
    df_features = extract_super_features(df)

    X: pd.DataFrame = df_features[
        ["People", "Process", "Physical_Evidence", "Experience_Value"]
    ]
    y: pd.Series = df_features["Minat_Kunjungan_Ulang"]

    # Initial OLS fit
    model = fit_ols_model(X, y)
    residuals_raw: np.ndarray = model.resid

    # Auto-transform if needed
    residuals_final: np.ndarray
    model_final: sm.regression.linear_model.RegressionResultsWrapper
    transform_note: str
    residuals_final, model_final, transform_note = apply_boxcox_if_needed(
        residuals_raw, y, X
    )

    # Final Shapiro-Wilk
    stat: float
    p_value: float
    stat, p_value = shapiro(residuals_final)
    status: str = "Normal" if p_value > ALPHA else "Tidak Normal"

    logger.info(f"Final Shapiro-Wilk: stat={stat:.4f}, p={p_value:.4f} -> {status}")
    logger.info(f"Transformation applied: {transform_note}")

    title_suffix: str = (
        f" ({transform_note})" if transform_note != "None (raw data)" else ""
    )
    export_qq_plot(
        residuals_final,
        output_path,
        title=f"Normal Q-Q Plot of OLS Residuals{title_suffix}",
    )

    logger.info("Uji normalitas completed successfully.")

    return {
        "shapiro_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "status": status,
        "transformation": transform_note,
    }


if __name__ == "__main__":
    run_uji_normalitas()
