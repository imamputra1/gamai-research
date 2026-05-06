# src/core_analysis/09_3_qc_final_check.py
from pathlib import Path

import pandas as pd

from src.utils import setup_logger

logger = setup_logger("qc_final_check")

EXPECTED_SHAPE: tuple[int, int] = (100, 35)
LIKERT_DOMAIN: set[int] = {1, 2, 3, 4, 5}


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
    return df


def validate_dimensions(df: pd.DataFrame) -> None:
    """Assert absolute dimensions: 100 rows, 35 columns.

    Args:
        df: Master dataframe.

    Raises:
        AssertionError: If shape mismatch.
    """
    assert df.shape == EXPECTED_SHAPE, (
        f"Dimensi gagal. Diharapkan {EXPECTED_SHAPE}, ditemukan {df.shape}"
    )
    logger.info(f"Dimensi Target {EXPECTED_SHAPE} : PASSED (Valid)")


def validate_no_missing(df: pd.DataFrame) -> None:
    """Assert zero missing values across entire dataframe.

    Args:
        df: Master dataframe.

    Raises:
        AssertionError: If any NaN detected.
    """
    missing_count: int = int(df.isna().sum().sum())
    assert missing_count == 0, (
        f"Missing values check gagal. Ditemukan {missing_count} NaN."
    )
    logger.info(f"Missing Values Check (0) : PASSED ({missing_count} NaN terdeteksi)")


def validate_likert_domain(df: pd.DataFrame) -> set[int]:
    """Assert all Likert values are strictly within {1, 2, 3, 4, 5}.

    Args:
        df: Master dataframe.

    Returns:
        set[int]: Unique Likert values detected.

    Raises:
        AssertionError: If any value outside Likert domain.
    """
    likert_block: pd.DataFrame = df.iloc[:, 5:30]
    unique_values: set[int] = set(likert_block.values.flatten())

    assert unique_values.issubset(LIKERT_DOMAIN), (
        f"Likert domain check gagal. Nilai unik terdeteksi: {unique_values}. "
        f"Diharapkan subset dari {LIKERT_DOMAIN}."
    )
    logger.info(
        f"Likert Domain {LIKERT_DOMAIN} : PASSED "
        f"(Nilai unik terdeteksi: {unique_values})"
    )
    return unique_values


def write_audit_report(
    output_path: Path,
    shape: tuple[int, int],
    missing_count: int,
    unique_values: set[int],
) -> None:
    """Export data quality audit report to text file.

    Args:
        output_path: Destination file path.
        shape: Validated dataframe shape.
        missing_count: Validated missing value count.
        unique_values: Validated unique Likert values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: str = (
        "=== LAPORAN AUDIT KUALITAS DATA ===\n\n"
        f"Dimensi Target (100x35)    : PASSED (Valid) -> {shape}\n"
        f"Missing Values Check (0)   : PASSED ({missing_count} NaN terdeteksi)\n"
        f"Likert Domain {{1,2,3,4,5}} : PASSED (Nilai unik terdeteksi: {unique_values})\n"
        "\nSTATUS FINAL: DATA BERSIH 100% DAN SIAP UJI STATISTIK.\n"
    )

    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Audit report exported: {output_path}")


def run_qc_final_check(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_path: Path = Path("reports/data_quality_audit.txt"),
) -> None:
    """Execute full quality control final check pipeline.

    Args:
        master_path: Path to locked master dataset.
        output_path: Path for audit report text file.
    """
    df = load_master_report(master_path)

    validate_dimensions(df)

    validate_no_missing(df)

    unique_values = validate_likert_domain(df)

    write_audit_report(output_path, df.shape, 0, unique_values)

    logger.info("QC Final Check completed successfully.")


if __name__ == "__main__":
    run_qc_final_check()
