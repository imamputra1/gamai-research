# src/core_synthesis/04_hierarchical_sim_laten.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from src.utils import load_config, setup_logger

logger = setup_logger("hierarchical_sim_laten")

LATENT_COLS: list[str] = [
    "People",
    "Process",
    "Physical_Evidence",
    "Experience_Value",
    "Minat_Kunjung",
]


def load_distribution_params(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed mean vector and covariance matrix.

    Args:
        input_dir: Directory containing .npy artifacts.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean vector (5,), Covariance matrix (5, 5).

    Raises:
        FileNotFoundError: If either artifact is missing.
    """
    mean_path = input_dir / "mean_vector.npy"
    cov_path = input_dir / "cov_matrix.npy"

    if not mean_path.exists():
        raise FileNotFoundError(f"Mean vector not found: {mean_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariance matrix not found: {cov_path}")

    mean_vector: np.ndarray = np.load(mean_path)
    cov_matrix: np.ndarray = np.load(cov_path)

    logger.info(f"Distribution params loaded | mean: {mean_vector.shape}, cov: {cov_matrix.shape}")
    return mean_vector, cov_matrix


def generate_latent_dataset(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    n_target: int,
    random_state: int,
) -> pd.DataFrame:
    """Generate one synthetic latent dataset via multivariate normal.

    Args:
        mean_vector: Mean vector (shape: 5,).
        cov_matrix: Covariance matrix (shape: 5, 5).
        n_target: Number of synthetic respondents (N).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic latent data (shape: N, 5).
    """
    synthetic_data: np.ndarray = multivariate_normal.rvs(
        mean=mean_vector,
        cov=cov_matrix,
        size=n_target,
        random_state=random_state,
    )
    df: pd.DataFrame = pd.DataFrame(synthetic_data, columns=LATENT_COLS)
    return df


def validate_latent_shape(df: pd.DataFrame, expected_rows: int, expected_cols: int = 5) -> None:
    """Validate synthetic latent dataframe dimensions.

    Args:
        df: Synthetic dataframe.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If shape mismatch.
    """
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Latent synthetic shape harus ({expected_rows}, {expected_cols}). "
            f"Ditemukan: {df.shape}"
        )


def save_latent_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Export synthetic latent dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_hierarchical_sim_laten(
    input_dir: Path = Path("data/02_processed"),
    output_dir: Path = Path("data/03_synthetic"),
) -> list[pd.DataFrame]:
    """Execute full hierarchical latent simulation: load, loop, generate, validate, save.

    Args:
        input_dir: Directory containing .npy distribution parameters.
        output_dir: Directory for M synthetic latent CSV outputs.

    Returns:
        list[pd.DataFrame]: List of M generated dataframes.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)
    base_seed: int = pipeline_cfg.get("random_seed", 42)

    mean_vector, cov_matrix = load_distribution_params(input_dir)

    generated: list[pd.DataFrame] = []

    for m in range(1, m_datasets + 1):
        random_state = base_seed + m
        df_syn = generate_latent_dataset(mean_vector, cov_matrix, n_target, random_state)

        validate_latent_shape(df_syn, expected_rows=n_target)

        output_path = output_dir / f"df_latent_syn_{m}.csv"
        save_latent_dataset(df_syn, output_path)

        generated.append(df_syn)

        if m == 1:
            corr_matrix = df_syn.corr()
            logger.info(f"Correlation matrix (m=1):\n{corr_matrix.round(4)}")

    logger.info(f"Successfully generated {m_datasets} latent datasets")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Hierarchical latent simulation completed successfully.")

    return generated


if __name__ == "__main__":
    run_hierarchical_sim_laten()
