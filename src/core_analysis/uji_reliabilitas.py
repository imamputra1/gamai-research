# src/core_analysis/09_5_uji_reliabilitas.py
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import setup_logger

logger = setup_logger("uji_reliabilitas")

CRITICAL_THRESHOLD: float = 0.70


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


def calculate_cronbach_alpha(df_subset: pd.DataFrame) -> float:
    """Compute Cronbach's Alpha for internal consistency.

    Formula: alpha = (k / (k-1)) * (1 - sum(item_variances) / total_variance)

    Args:
        df_subset: DataFrame with k indicator columns.

    Returns:
        float: Alpha coefficient [0.0, 1.0].
    """
    k: int = df_subset.shape[1]
    item_variances: pd.Series = df_subset.var(axis=0, ddof=1)
    total_variance: float = float(df_subset.sum(axis=1).var(ddof=1))

    if total_variance == 0:
        return 0.0

    alpha: float = (k / (k - 1)) * (1 - (item_variances.sum() / total_variance))
    return alpha


def compute_reliability_matrix(
    df: pd.DataFrame,
    mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute Cronbach's Alpha per construct and assign status.

    Args:
        df: Master dataframe.
        mapping: Construct-to-indicators mapping.

    Returns:
        pd.DataFrame: Reliability matrix.
    """
    rows: list[dict[str, Any]] = []

    for construct, items in mapping.items():
        alpha: float = calculate_cronbach_alpha(df[items])

        if alpha >= CRITICAL_THRESHOLD:
            status: str = "Reliabel"
        elif alpha >= 0.60:
            status = "Dipertanyakan (Questionable)"
        else:
            status = "Tidak Reliabel"

        rows.append(
            {
                "Variabel/Konstruk": construct,
                "Jumlah Item (k)": len(items),
                "Cronbach's Alpha": round(alpha, 4),
                "Batas Kritis": CRITICAL_THRESHOLD,
                "Keterangan": status,
            }
        )

    df_reliability: pd.DataFrame = pd.DataFrame(rows)
    return df_reliability


def export_reliability_table(df: pd.DataFrame, output_path: Path) -> None:
    """Export reliability matrix to Excel.

    Args:
        df: Reliability dataframe.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name="Reliabilitas")
    logger.info(f"Tabel reliabilitas exported: {output_path}")


def run_uji_reliabilitas(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_path: Path = Path("reports/tables/tabel_reliabilitas_cronbach.xlsx"),
) -> pd.DataFrame:
    """Execute full reliability test pipeline.

    Args:
        master_path: Path to locked master dataset.
        output_path: Path for reliability Excel output.

    Returns:
        pd.DataFrame: Reliability matrix.
    """
    df = load_master_report(master_path)

    mapping = build_construct_mapping(df)
    df_reliability = compute_reliability_matrix(df, mapping)

    export_reliability_table(df_reliability, output_path)

    reliable_count: int = int(
        (df_reliability["Keterangan"] == "Reliabel").sum()
    )
    total_count: int = len(df_reliability)
    logger.info(
        f"Uji reliabilitas: {reliable_count}/{total_count} konstruk Reliabel "
        f"(threshold = {CRITICAL_THRESHOLD})"
    )
    logger.info("Uji reliabilitas completed successfully.")

    return df_reliability


if __name__ == "__main__":
    run_uji_reliabilitas()
