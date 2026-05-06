# src/core_analysis/09_6_3_uji_hetero.py
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd

from src.utils import setup_logger

logger = setup_logger("uji_hetero")

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


def extract_ols_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract predictor matrix X and dependent vector y.

    Args:
        df: Master dataframe (shape: 100, 35).

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y).
    """
    df_likert: pd.DataFrame = df.iloc[:, 5:30].copy()

    X: pd.DataFrame = pd.DataFrame(
        {
            "People": df_likert.iloc[:, 0:5].mean(axis=1),
            "Process": df_likert.iloc[:, 5:10].mean(axis=1),
            "Physical_Evidence": df_likert.iloc[:, 10:15].mean(axis=1),
            "Experience_Value": df_likert.iloc[:, 15:20].mean(axis=1),
        }
    )
    y: pd.Series = df_likert.iloc[:, 20:25].mean(axis=1)
    return X, y


def fit_ols_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS regression with constant term.

    Args:
        X: Independent variables.
        y: Dependent variable.

    Returns:
        RegressionResultsWrapper: Fitted model.
    """
    X_sm: pd.DataFrame = sm.add_constant(X)
    model: sm.regression.linear_model.RegressionResultsWrapper = sm.OLS(
        y, X_sm
    ).fit()
    return model


def run_breusch_pagan(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[float, float, str]:
    """Execute Breusch-Pagan test for heteroskedasticity.

    Args:
        model: Fitted OLS model.

    Returns:
        tuple[float, float, str]: (lm_stat, p_value, status).
    """
    bp_test = smd.het_breuschpagan(model.resid, model.model.exog)
    lm_stat: float = float(bp_test[0])
    p_value: float = float(bp_test[1])
    status: str = "Homoskedastik" if p_value > ALPHA else "Heteroskedastik"

    logger.info(
        f"Breusch-Pagan: LM={lm_stat:.4f}, p={p_value:.4f} -> {status}"
    )
    return lm_stat, p_value, status


def run_glejser(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[pd.Series, str]:
    """Execute Glejser test using absolute residuals.

    Args:
        model: Fitted OLS model.

    Returns:
        tuple[pd.Series, str]: (p_values_series, status).
    """
    abs_resid: np.ndarray = np.abs(model.resid)
    model_glejser: sm.regression.linear_model.RegressionResultsWrapper = sm.OLS(
        abs_resid, model.model.exog
    ).fit()

    glejser_pvalues: pd.Series = model_glejser.pvalues.iloc[1:]  # skip const
    status: str = (
        "Homoskedastik"
        if (glejser_pvalues > ALPHA).all()
        else "Heteroskedastik"
    )

    for var, p in glejser_pvalues.items():
        logger.info(f"  Glejser {var}: p={p:.4f}")
    logger.info(f"Glejser status: {status}")

    return glejser_pvalues, status


def export_scatterplot(
    fitted: pd.Series,
    residuals: pd.Series,
    output_path: Path,
) -> None:
    """Generate and save residual vs fitted scatterplot.

    Args:
        fitted: Fitted values.
        residuals: Residuals.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        fitted,
        residuals,
        alpha=0.7,
        color="blue",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Values (Fitted)")
    ax.set_ylabel("Residuals")
    ax.set_title("Scatterplot Uji Heteroskedastisitas")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Scatterplot exported: {output_path}")


def export_results_table(
    bp_stat: float,
    bp_p: float,
    bp_status: str,
    glejser_pvalues: pd.Series,
    glejser_status: str,
    output_path: Path,
) -> None:
    """Export heteroskedasticity test results to Excel.

    Args:
        bp_stat: Breusch-Pagan LM statistic.
        bp_p: Breusch-Pagan p-value.
        bp_status: Breusch-Pagan status.
        glejser_pvalues: Glejser p-values per predictor.
        glejser_status: Glejser overall status.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = [
        {
            "Uji": "Breusch-Pagan",
            "Statistik": round(bp_stat, 4),
            "p-value": round(bp_p, 4),
            "Keterangan": bp_status,
        },
        {
            "Uji": "Glejser (Overall)",
            "Statistik": None,
            "p-value": None,
            "Keterangan": glejser_status,
        },
    ]

    for var, p in glejser_pvalues.items():
        rows.append(
            {
                "Uji": f"Glejser ({var})",
                "Statistik": None,
                "p-value": round(p, 4),
                "Keterangan": (
                    "Homoskedastik" if p > ALPHA else "Heteroskedastik"
                ),
            }
        )

    df_results: pd.DataFrame = pd.DataFrame(rows)
    df_results.to_excel(output_path, index=False, sheet_name="Heteroskedastisitas")
    logger.info(f"Results table exported: {output_path}")


def run_uji_hetero(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    scatter_path: Path = Path("reports/figures/scatterplot_hetero.png"),
    table_path: Path = Path("reports/tables/tabel_heteroskedastisitas.xlsx"),
) -> dict[str, Any]:
    """Execute full heteroskedasticity test pipeline (Breusch-Pagan + Glejser + Visual).

    Args:
        master_path: Path to locked master dataset.
        scatter_path: Path for scatterplot PNG.
        table_path: Path for results Excel.

    Returns:
        dict: Test results summary.
    """
    df = load_master_report(master_path)
    X, y = extract_ols_matrices(df)

    model = fit_ols_model(X, y)

    bp_stat, bp_p, bp_status = run_breusch_pagan(model)
    glejser_pvalues, glejser_status = run_glejser(model)

    export_scatterplot(model.fittedvalues, model.resid, scatter_path)
    export_results_table(
        bp_stat, bp_p, bp_status, glejser_pvalues, glejser_status, table_path
    )

    overall_pass: bool = (
        bp_status == "Homoskedastik" and glejser_status == "Homoskedastik"
    )

    logger.info(
        f"Heteroskedastisitas: {'PASSED' if overall_pass else 'FAILED'} "
        f"(BP: {bp_status}, Glejser: {glejser_status})"
    )
    logger.info("Uji heteroskedastisitas completed successfully.")

    return {
        "breusch_pagan": {
            "statistic": round(bp_stat, 4),
            "p_value": round(bp_p, 4),
            "status": bp_status,
        },
        "glejser": {
            "p_values": {k: round(v, 4) for k, v in glejser_pvalues.items()},
            "status": glejser_status,
        },
        "overall": "PASSED" if overall_pass else "FAILED",
    }


if __name__ == "__main__":
    run_uji_hetero()
