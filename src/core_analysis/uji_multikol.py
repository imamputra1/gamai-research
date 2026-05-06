from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils import setup_logger

logger = setup_logger("uji_multikol")

VIF_THRESHOLD: float = 10.0
TOLERANCE_THRESHOLD: float = 0.10


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


def extract_predictor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 4 predictor super-features for VIF computation.

    Args:
        df: Master dataframe (shape: 100, 35).

    Returns:
        pd.DataFrame: Predictor matrix (shape: 100, 4).
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
    return X


def compute_vif_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Tolerance and VIF for each predictor with constant term.

    Args:
        X: Predictor dataframe (no constant).

    Returns:
        pd.DataFrame: VIF matrix with tolerance, VIF, and status.
    """
    X_with_const: pd.DataFrame = sm.add_constant(X)

    rows: list[dict[str, Any]] = []

    for i in range(1, X_with_const.shape[1]):
        vif_val: float = float(
            variance_inflation_factor(X_with_const.values, i)
        )
        tol_val: float = 1.0 / vif_val
        var_name: str = str(X_with_const.columns[i])

        status: str = (
            "Bebas Multikolinearitas"
            if (vif_val <= VIF_THRESHOLD and tol_val > TOLERANCE_THRESHOLD)
            else "Gejala Multikolinearitas"
        )

        rows.append(
            {
                "Variabel": var_name,
                "Tolerance": round(tol_val, 4),
                "VIF": round(vif_val, 4),
                "Keterangan": status,
            }
        )

    return pd.DataFrame(rows)


def export_vif_table(df: pd.DataFrame, output_path: Path) -> None:
    """Export VIF matrix to Excel.

    Args:
        df: VIF dataframe.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name="Multikolinearitas")
    logger.info(f"Tabel multikolinearitas exported: {output_path}")


def run_uji_multikol(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_path: Path = Path("reports/tables/tabel_multikolinearitas.xlsx"),
) -> pd.DataFrame:
    """Execute full multicollinearity test pipeline.

    Args:
        master_path: Path to locked master dataset.
        output_path: Path for VIF Excel output.

    Returns:
        pd.DataFrame: VIF matrix.
    """
    df = load_master_report(master_path)
    X = extract_predictor_matrix(df)

    df_vif = compute_vif_matrix(X)
    export_vif_table(df_vif, output_path)

    pass_count: int = int(
        (df_vif["Keterangan"] == "Bebas Multikolinearitas").sum()
    )
    total_count: int = len(df_vif)
    logger.info(
        f"Multikolinearitas: {pass_count}/{total_count} variabel bebas "
        f"(VIF <= {VIF_THRESHOLD}, TOL > {TOLERANCE_THRESHOLD})"
    )
    logger.info("Uji multikolinearitas completed successfully.")

    return df_vif


if __name__ == "__main__":
    run_uji_multikol()
