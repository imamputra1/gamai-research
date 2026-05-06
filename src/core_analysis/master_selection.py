# src/core_analysis/master_selection.py
from pathlib import Path

import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("master_selection")

EXPECTED_SHAPE: tuple[int, int] = (100, 35)


def resolve_target_dataset(synthetic_dir: Path, target_index: int) -> Path:
    """Resolve file path for the selected dataset index.

    Args:
        synthetic_dir: Directory containing df_final_syn_*.csv files.
        target_index: Dataset index m to select (1-based).

    Returns:
        Path: Resolved file path.

    Raises:
        FileNotFoundError: If resolved file does not exist.
    """
    target_path = synthetic_dir / f"df_final_syn_{target_index}.csv"
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset m={target_index} not found: {target_path}"
        )
    return target_path


def load_target_dataset(path: Path) -> pd.DataFrame:
    """Load the selected synthetic dataset for master reporting.

    Args:
        path: Path to df_final_syn_{m}.csv.

    Returns:
        pd.DataFrame: Master dataset (shape: 100, 35).

    Raises:
        AssertionError: If shape does not match expected dimensions.
    """
    df: pd.DataFrame = pd.read_csv(path)

    assert df.shape == EXPECTED_SHAPE, (
        f"Shape assertion failed. Expected {EXPECTED_SHAPE}, got {df.shape}"
    )

    logger.info(f"Target dataset loaded: {path.name} | Shape: {df.shape}")
    return df


def isolate_master_file(df: pd.DataFrame, output_dir: Path, target_index: int) -> Path:
    """Save master dataframe to isolated report directory with provenance metadata.

    Args:
        df: Validated master dataframe.
        output_dir: Destination directory.
        target_index: Source dataset index for audit trail.

    Returns:
        Path: Output file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "df_report_master.csv"
    df.to_csv(output_path, index=False)

    # Write provenance metadata
    meta_path = output_dir / "master_provenance.txt"
    meta_content = (
        f"source_dataset: df_final_syn_{target_index}.csv\n"
        f"shape: {df.shape}\n"
        f"rows: {df.shape[0]}\n"
        f"columns: {df.shape[1]}\n"
    )
    meta_path.write_text(meta_content, encoding="utf-8")

    logger.info(f"Master file isolated: {output_path}")
    return output_path


def run_master_selection(
    synthetic_dir: Path = Path("data/03_synthetic"),
    report_dir: Path = Path("data/04_report_master"),
    target_index: int | None = None,
) -> pd.DataFrame:
    """Execute master dataset selection and isolation for Chapter 4 reporting.

    Args:
        synthetic_dir: Directory containing df_final_syn_*.csv.
        report_dir: Isolated directory for df_report_master.csv.
        target_index: Dataset index m to select (1-based). If None, reads from
                      pipeline_config.yaml key 'report_master_index' (default 1).

    Returns:
        pd.DataFrame: Locked master dataframe.
    """
    resolved_index: int = target_index if target_index is not None else 1

    if target_index is None:
        config = load_config()
        resolved_index = config.get("pipeline", {}).get("report_master_index", 1)

    target_path = resolve_target_dataset(synthetic_dir, resolved_index)
    df_master = load_target_dataset(target_path)

    isolate_master_file(df_master, report_dir, resolved_index)

    logger.info(
        f"PONDASI EDA: Dataset m={resolved_index} (df_final_syn_{resolved_index}.csv) "
        f"berhasil diisolasi dan dikunci sebagai df_report_master.csv "
        f"untuk pelaporan visual Bab 4."
    )

    return df_master


if __name__ == "__main__":
    run_master_selection()
