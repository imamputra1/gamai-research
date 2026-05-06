# src/core_synthesis/06_post_processing_likert.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("post_processing_likert")


def load_continuous_likert(path: Path) -> pd.DataFrame:
    """Load continuous Likert synthetic dataset.

    Args:
        path: Path to df_likert_cont_syn_{m}.csv.

    Returns:
        pd.DataFrame: Continuous synthetic Likert data.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Continuous Likert file not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    return df


def discretize_likert(df: pd.DataFrame) -> pd.DataFrame:
    """Round, clip to [1, 5], and cast to integer.

    Args:
        df: Continuous Likert dataframe.

    Returns:
        pd.DataFrame: Discretized integer Likert dataframe.
    """
    df_rounded: pd.DataFrame = np.round(df)
    df_clipped: pd.DataFrame = np.clip(df_rounded, a_min=1, a_max=5)
    df_final: pd.DataFrame = df_clipped.astype(int)
    return df_final


def validate_discrete_bounds(df: pd.DataFrame) -> None:
    """Validate all values are strictly within {1, 2, 3, 4, 5}.

    Args:
        df: Discretized Likert dataframe.

    Raises:
        ValueError: If any value outside [1, 5] or non-integer detected.
    """
    global_min: int = int(df.min().min())
    global_max: int = int(df.max().max())

    if global_min < 1 or global_max > 5:
        raise ValueError(
            f"Likert bounds violated. Min: {global_min}, Max: {global_max}"
        )

    unique_vals = set(df.values.flatten())
    invalid = unique_vals - {1, 2, 3, 4, 5}
    if invalid:
        raise ValueError(f"Invalid Likert values detected: {invalid}")


def save_discrete_likert(df: pd.DataFrame, output_path: Path) -> None:
    """Export discretized Likert dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_post_processing_likert(
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute full post-processing: load, round, clip, validate, save.

    Args:
        synthetic_dir: Directory containing df_likert_cont_syn_*.csv
                       and output destination for df_likert_syn_*.csv.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)

    for m in range(1, m_datasets + 1):
        cont_path = synthetic_dir / f"df_likert_cont_syn_{m}.csv"
        df_cont = load_continuous_likert(cont_path)

        df_final = discretize_likert(df_cont)
        validate_discrete_bounds(df_final)

        output_path = synthetic_dir / f"df_likert_syn_{m}.csv"
        save_discrete_likert(df_final, output_path)

    global_min = int(df_final.min().min())
    global_max = int(df_final.max().max())
    logger.info(f"Rounding & clipping completed for {m_datasets} datasets")
    logger.info(f"Sanity check (last dataset) | Min: {global_min} | Max: {global_max}")
    logger.info("Post-processing Likert discretization completed successfully.")


if __name__ == "__main__":
    run_post_processing_likert()
