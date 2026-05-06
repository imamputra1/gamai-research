# src/core_analysis/09_4_uji_validitas.py
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import pearsonr

from src.utils import setup_logger

logger = setup_logger("uji_validitas")

R_TABEL: float = 0.1966  # Pearson critical value, df=98, alpha=0.05, two-tailed


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


def build_construct_mapping(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map Likert columns to 5 constructs via positional slicing on master df.

    Args:
        df: Master dataframe (shape: 100, 35).

    Returns:
        dict[str, list[str]]: Construct -> list of column names.
    """
    mapping: dict[str, list[str]] = {
        "People": df.columns[5:10].tolist(),
        "Process": df.columns[10:15].tolist(),
        "Physical_Evidence": df.columns[15:20].tolist(),
        "Experience_Value": df.columns[20:25].tolist(),
        "Minat_Kunjung": df.columns[25:30].tolist(),
    }
    return mapping


def compute_validity_matrix(
    df: pd.DataFrame,
    mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute Pearson Item-Total correlation per indicator.

    Total score = sum of all items in the construct (not mean).

    Args:
        df: Master dataframe.
        mapping: Construct-to-indicators mapping.

    Returns:
        pd.DataFrame: Validity matrix with r_hitung, r_tabel, p_value, status.
    """
    rows: list[dict[str, Any]] = []

    for construct, items in mapping.items():
        total_score: pd.Series = df[items].sum(axis=1)

        for item in items:
            r_hitung: float
            p_value: float
            r_hitung, p_value = pearsonr(df[item], total_score)

            status: str = (
                "Valid"
                if (r_hitung > R_TABEL and p_value < 0.05)
                else "Tidak Valid"
            )

            rows.append(
                {
                    "Konstruk": construct,
                    "Item/Indikator": item,
                    "r_hitung": round(r_hitung, 4),
                    "r_tabel": R_TABEL,
                    "p_value": round(p_value, 4),
                    "Keterangan": status,
                }
            )

    df_validity: pd.DataFrame = pd.DataFrame(rows)
    return df_validity


def export_validity_table(df: pd.DataFrame, output_path: Path) -> None:
    """Export validity matrix to Excel.

    Args:
        df: Validity dataframe.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name="Validitas")
    logger.info(f"Tabel validitas exported: {output_path}")


def run_uji_validitas(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_path: Path = Path("reports/tables/tabel_validitas_pearson.xlsx"),
) -> pd.DataFrame:
    """Execute full validity test pipeline.

    Args:
        master_path: Path to locked master dataset.
        output_path: Path for validity Excel output.

    Returns:
        pd.DataFrame: Validity matrix.
    """
    df = load_master_report(master_path)

    mapping = build_construct_mapping(df)
    df_validity = compute_validity_matrix(df, mapping)

    export_validity_table(df_validity, output_path)

    valid_count: int = int((df_validity["Keterangan"] == "Valid").sum())
    total_count: int = len(df_validity)
    logger.info(
        f"Uji validitas: {valid_count}/{total_count} indikator Valid "
        f"(r_tabel = {R_TABEL})"
    )
    logger.info("Uji validitas completed successfully.")

    return df_validity


if __name__ == "__main__":
    run_uji_validitas()
