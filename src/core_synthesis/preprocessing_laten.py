# src/core_synthesis/02_preprocessing_laten.py
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils import setup_logger

logger = setup_logger("preprocessing_laten")


def load_likert_data(path: Path) -> pd.DataFrame:
    """Load cleaned Likert dataframe from CSV.

    Args:
        path: Path to df_likert.csv.

    Returns:
        pd.DataFrame: Likert-scale data.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Likert file not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Likert data loaded: {df.shape}")
    return df


def validate_likert_shape(df: pd.DataFrame) -> None:
    """Validate exact Likert dimensions.

    Args:
        df: Likert dataframe.

    Raises:
        ValueError: If shape is not (20, 25).
    """
    if df.shape != (20, 25):
        raise ValueError(
            f"df_likert shape harus (20, 25). Ditemukan: {df.shape}"
        )


def build_construct_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map Likert columns to 5 latent constructs via positional slicing.

    Args:
        df: Likert dataframe with exactly 25 columns.

    Returns:
        Dict[str, List[str]]: Mapping construct name -> list of column names.
    """
    mapping: Dict[str, List[str]] = {
        "People": df.columns[0:5].tolist(),
        "Process": df.columns[5:10].tolist(),
        "Physical_Evidence": df.columns[10:15].tolist(),
        "Experience_Value": df.columns[15:20].tolist(),
        "Minat_Kunjung": df.columns[20:25].tolist(),
    }
    return mapping


def compute_latent_means(
    df: pd.DataFrame,
    mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """Compute row-wise mean per construct to reduce dimensionality.

    Args:
        df: Likert dataframe (shape: 20, 25).
        mapping: Construct-to-columns mapping.

    Returns:
        pd.DataFrame: Latent mean dataframe (shape: 20, 5).
    """
    df_latent: pd.DataFrame = pd.DataFrame(index=df.index)

    for construct_name, columns in mapping.items():
        df_latent[construct_name] = df[columns].mean(axis=1)

    return df_latent


def validate_latent_data(df: pd.DataFrame) -> None:
    """Validate latent dataframe has zero missing values and correct shape.

    Args:
        df: Latent dataframe.

    Raises:
        ValueError: If shape != (20, 5) or missing values exist.
    """
    if df.shape != (20, 5):
        raise ValueError(
            f"df_latent_seed shape harus (20, 5). Ditemukan: {df.shape}"
        )
    missing_count: int = int(df.isna().sum().sum())
    if missing_count > 0:
        raise ValueError(
            f"df_latent_seed mengandung {missing_count} missing value."
        )


def export_latent(df: pd.DataFrame, output_path: Path) -> None:
    """Export latent dataframe to CSV without index.

    Args:
        df: Latent dataframe.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Latent seed exported: {output_path}")


def run_preprocessing_laten(
    input_dir: Path = Path("data/02_processed"),
    output_dir: Path = Path("data/02_processed"),
) -> pd.DataFrame:
    """Execute full latent preprocessing pipeline.

    Args:
        input_dir: Directory containing df_likert.csv.
        output_dir: Directory for df_latent_seed.csv output.

    Returns:
        pd.DataFrame: Latent mean dataframe (shape: 20, 5).
    """
    likert_path = input_dir / "df_likert.csv"
    df_likert = load_likert_data(likert_path)

    validate_likert_shape(df_likert)

    mapping = build_construct_mapping(df_likert)
    logger.info(f"Construct mapping built: {list(mapping.keys())}")

    df_latent = compute_latent_means(df_likert, mapping)

    validate_latent_data(df_latent)

    output_path = output_dir / "df_latent_seed.csv"
    export_latent(df_latent, output_path)

    logger.info(f"df_latent_seed shape: {df_latent.shape} (expected: 20x5)")
    logger.info("Latent preprocessing completed successfully.")

    return df_latent


if __name__ == "__main__":
    run_preprocessing_laten()
