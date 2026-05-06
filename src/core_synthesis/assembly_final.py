# src/core_synthesis/09_assembly_final.py
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("assembly_final")

EXPECTED_COLS: int = 35


def load_component(path: Path) -> pd.DataFrame:
    """Load a single component dataframe.

    Args:
        path: Path to CSV file.

    Returns:
        pd.DataFrame: Loaded component.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Component file not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    return df


def assemble_dataset(
    df_demo: pd.DataFrame,
    df_likert: pd.DataFrame,
    df_teks: pd.DataFrame,
) -> pd.DataFrame:
    """Horizontally concatenate demo, likert, and text dataframes.

    Args:
        df_demo: Demographic dataframe.
        df_likert: Likert-scale dataframe.
        df_teks: Text response dataframe.

    Returns:
        pd.DataFrame: Assembled master dataframe.
    """
    df_final: pd.DataFrame = pd.concat([df_demo, df_likert, df_teks], axis=1)
    return df_final


def validate_assembly(df: pd.DataFrame, expected_rows: int, expected_cols: int) -> None:
    """Validate final assembled dataframe dimensions and integrity.

    Args:
        df: Assembled dataframe.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If shape mismatch or missing values detected.
    """
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Baris df_final harus {expected_rows}. Ditemukan: {df.shape[0]}"
        )
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Kolom df_final harus {expected_cols}. Ditemukan: {df.shape[1]}"
        )

    missing_count: int = int(df.isna().sum().sum())
    if missing_count > 0:
        raise ValueError(
            f"df_final mengandung {missing_count} missing value."
        )


def save_final_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Export assembled master dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_assembly_final(
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute final assembly pipeline: load, concat, validate, save.

    Args:
        synthetic_dir: Directory containing component CSVs and output destination.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)

    for m in range(1, m_datasets + 1):
        demo_path = synthetic_dir / f"df_demo_syn_{m}.csv"
        likert_path = synthetic_dir / f"df_likert_syn_{m}.csv"
        teks_path = synthetic_dir / f"df_teks_syn_{m}.csv"

        df_demo = load_component(demo_path)
        df_likert = load_component(likert_path)
        df_teks = load_component(teks_path)

        df_final = assemble_dataset(df_demo, df_likert, df_teks)

        validate_assembly(df_final, expected_rows=n_target, expected_cols=EXPECTED_COLS)

        output_path = synthetic_dir / f"df_final_syn_{m}.csv"
        save_final_dataset(df_final, output_path)

        logger.info(f"Assembling dataset m={m}... SUCCESS (Shape: {df_final.shape[0]}x{df_final.shape[1]})")

    logger.info(f"BLOK B COMPLETED: {m_datasets} Final Datasets ready for Statistical Modeling.")


if __name__ == "__main__":
    run_assembly_final()
