# src/core_synthesis/ingestion.py
import re
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from src.utils import setup_logger
from src.utils.schema_alignment import (
    sanitize_headers,
    align_column_order,
    enforce_likert_types,
    impute_missing_values,
)

logger = setup_logger("ingestion")

# Regex-based schema for Sri Husada dataset (5 demo + 25 likert + 5 teks)
_SCHEMA: dict[str, List[str]] = {
    "demo": [
        r"^Jenis\s+Kelamin",
        r"^Usia",
        r"^Pendidikan",
        r"^Pekerjaan",
        r"^Frekuensi",
    ],
    "likert": [
        r"^X[123]_\d+",
        r"^M_\d+",
        r"^Y_\d+",
    ],
    "teks": [
        r"^Menurut\s+Anda,\s+bagaimana\s+kualitas",
        r"^Bagaimana\s+pendapat\s+Anda\s+mengenai\s+proses",
        r"^Bagaimana\s+kesan\s+Anda\s+terhadap\s+fasilitas",
        r"^Bagaimana\s+kesan\s+Anda\s+secara\s+keseluruhan",
        r"^Apa\s+alasan\s+utama",
    ],
}


def load_seed_file(seed_dir: Path) -> pd.DataFrame:
    """Load seed file from directory. Supports .csv and .xlsx.

    Args:
        seed_dir: Directory containing seed file.

    Returns:
        pd.DataFrame: Raw seed dataframe.
    """
    candidates: list[Path] = sorted(
        [f for f in seed_dir.iterdir() if f.suffix.lower() in (".csv", ".xlsx")]
    )
    if not candidates:
        raise FileNotFoundError(f"No .csv or .xlsx found in {seed_dir}")

    seed_file = candidates[0]
    if seed_file.suffix.lower() == ".xlsx":
        df: pd.DataFrame = pd.read_excel(seed_file)
    else:
        df = pd.read_csv(seed_file)

    logger.info(f"Seed file loaded: {seed_file.name}")
    return df


def validate_dimensions(df: pd.DataFrame, expected_rows: int = 20, expected_cols: int = 35) -> None:
    """Validate exact dimensions after schema alignment (Timestamp dropped).

    Args:
        df: Aligned dataframe.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If dimensions do not match.
    """
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Data seed harus memiliki tepat {expected_rows} baris. Ditemukan: {df.shape[0]}"
        )
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Data seed harus memiliki tepat {expected_cols} kolom. Ditemukan: {df.shape[1]}"
        )


def validate_no_missing(df: pd.DataFrame) -> None:
    """Validate zero missing values across entire dataframe.

    Args:
        df: Dataframe to validate.

    Raises:
        ValueError: If any NaN/Null detected.
    """
    missing_count: int = int(df.isna().sum().sum())
    if missing_count > 0:
        raise ValueError(
            f"Data seed mengandung {missing_count} missing value. Perbaiki data mentah."
        )


def extract_column_categories(df: pd.DataFrame, schema: dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    """Identify columns by category using regex patterns from schema.

    Args:
        df: Aligned dataframe.
        schema: Schema dictionary with demo/likert/teks regex patterns.

    Returns:
        Tuple[List[str], List[str], List[str]]: (demo_cols, likert_cols, teks_cols).
    """
    demo_cols: List[str] = []
    likert_cols: List[str] = []
    teks_cols: List[str] = []
    assigned: set[str] = set()

    for col in df.columns:
        matched = False
        for pattern in schema.get("likert", []):
            if re.search(pattern, col, re.IGNORECASE) and col not in assigned:
                likert_cols.append(col)
                assigned.add(col)
                matched = True
                break
        if matched:
            continue

        for pattern in schema.get("demo", []):
            if re.search(pattern, col, re.IGNORECASE) and col not in assigned:
                demo_cols.append(col)
                assigned.add(col)
                matched = True
                break
        if matched:
            continue

        for pattern in schema.get("teks", []):
            if re.search(pattern, col, re.IGNORECASE) and col not in assigned:
                teks_cols.append(col)
                assigned.add(col)
                break

    return demo_cols, likert_cols, teks_cols


def run_schema_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Execute full schema alignment: drop metadata, sanitize, reorder, type-cast, impute.

    Args:
        df: Raw seed dataframe.

    Returns:
        pd.DataFrame: Clean, aligned dataframe ready for slicing.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    df = sanitize_headers(df)
    df = align_column_order(df, _SCHEMA)

    demo_cols, likert_cols, teks_cols = extract_column_categories(df, _SCHEMA)

    df = enforce_likert_types(df, likert_cols)
    df = impute_missing_values(df, demo_cols, likert_cols, teks_cols)

    return df


def slice_dataframes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Segregate aligned dataframe into demo, likert, and teks sub-datasets.

    Args:
        df: Aligned dataframe (shape: 20, 35).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_demo (20, 5), df_likert (20, 25), df_teks (20, 5).
    """
    demo_cols, likert_cols, teks_cols = extract_column_categories(df, _SCHEMA)

    df_demo: pd.DataFrame = df[demo_cols].copy()
    df_likert: pd.DataFrame = df[likert_cols].copy()
    df_teks: pd.DataFrame = df[teks_cols].copy()

    return df_demo, df_likert, df_teks


def export_dataframes(
    df_demo: pd.DataFrame,
    df_likert: pd.DataFrame,
    df_teks: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export three dataframes to CSV without index.

    Args:
        df_demo: Demographic dataframe.
        df_likert: Likert-scale dataframe.
        df_teks: Open-ended text dataframe.
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_demo.to_csv(output_dir / "df_demo.csv", index=False)
    df_likert.to_csv(output_dir / "df_likert.csv", index=False)
    df_teks.to_csv(output_dir / "df_teks.csv", index=False)


def run_ingestion(
    seed_dir: Path = Path("data/01_raw_seed"),
    output_dir: Path = Path("data/02_processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute full ingestion pipeline with schema alignment.

    Args:
        seed_dir: Source directory for seed file.
        output_dir: Destination directory for processed CSVs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Processed demo, likert, and teks dataframes.
    """
    df: pd.DataFrame = load_seed_file(seed_dir)

    df = run_schema_alignment(df)

    validate_dimensions(df, expected_rows=20, expected_cols=35)
    validate_no_missing(df)

    df_demo, df_likert, df_teks = slice_dataframes(df)

    export_dataframes(df_demo, df_likert, df_teks, output_dir)

    logger.info(f"df_demo shape: {df_demo.shape} (expected: 20x5)")
    logger.info(f"df_likert shape: {df_likert.shape} (expected: 20x25)")
    logger.info(f"df_teks shape: {df_teks.shape} (expected: 20x5)")
    logger.info("Ingestion completed successfully. No missing values found.")

    return df_demo, df_likert, df_teks


if __name__ == "__main__":
    run_ingestion()
