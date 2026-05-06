# src/core_synthesis/generator_setup.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.utils import load_config, setup_logger

logger = setup_logger("generator_setup")


def load_latent_seed(path: Path) -> pd.DataFrame:
    """Load latent seed dataframe from CSV.

    Args:
        path: Path to df_latent_seed.csv.

    Returns:
        pd.DataFrame: Latent mean data (shape: 20, 5).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Latent seed file not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Latent seed loaded: {df.shape}")
    return df


def validate_latent_shape(df: pd.DataFrame) -> None:
    """Validate latent seed dimensions.

    Args:
        df: Latent dataframe.

    Raises:
        ValueError: If shape is not (20, 5).
    """
    if df.shape != (20, 5):
        raise ValueError(
            f"df_latent_seed shape harus (20, 5). Ditemukan: {df.shape}"
        )


def extract_distribution_params(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean vector and covariance matrix from latent dataframe.

    Args:
        df: Latent dataframe (shape: N, k).

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean vector (k,), Covariance matrix (k, k).
    """
    values: np.ndarray = df.to_numpy(dtype=np.float64)
    mean_vector: np.ndarray = np.mean(values, axis=0)
    cov_matrix: np.ndarray = np.cov(values, rowvar=False)
    return mean_vector, cov_matrix


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix is Positive Semi-Definite via eigenvalue inspection.

    Args:
        matrix: Square symmetric matrix.
        tol: Tolerance for near-zero eigenvalues.

    Returns:
        bool: True if all eigenvalues >= -tol.
    """
    eigenvalues: np.ndarray = np.linalg.eigvals(matrix)
    min_eig: float = float(np.min(eigenvalues))
    logger.info(f"Minimum eigenvalue: {min_eig:.2e}")
    return min_eig >= -tol


def regularize_covariance(
    values: np.ndarray,
    cov_matrix: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Apply Ledoit-Wolf shrinkage and optional jitter if matrix is not PSD.

    Args:
        values: Raw latent values (shape: N, k).
        cov_matrix: Initial covariance matrix (shape: k, k).
        tol: PSD tolerance threshold.

    Returns:
        np.ndarray: Robust covariance matrix guaranteed PSD.
    """
    if is_psd(cov_matrix, tol):
        logger.info("PSD Check: PASS")
        return cov_matrix

    logger.warning("PSD Check: FAILED, applying Ledoit-Wolf Shrinkage")
    lw = LedoitWolf()
    lw.fit(values)
    cov_matrix = lw.covariance_

    if not is_psd(cov_matrix, tol):
        logger.warning("PSD Check: STILL FAILED, applying diagonal jitter")
        k = cov_matrix.shape[0]
        cov_matrix = cov_matrix + np.eye(k) * 1e-6

    logger.info("PSD Check: RECOVERED")
    return cov_matrix


def save_artifacts(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    output_dir: Path,
) -> None:
    """Save distribution parameters as .npy binary files.

    Args:
        mean_vector: Mean vector (shape: k,).
        cov_matrix: Covariance matrix (shape: k, k).
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_path = output_dir / "mean_vector.npy"
    cov_path = output_dir / "cov_matrix.npy"

    np.save(mean_path, mean_vector)
    np.save(cov_path, cov_matrix)

    logger.info(f"Mean vector saved: {mean_path} (shape: {mean_vector.shape})")
    logger.info(f"Covariance matrix saved: {cov_path} (shape: {cov_matrix.shape})")


def run_generator_setup(
    input_dir: Path = Path("data/02_processed"),
    output_dir: Path = Path("data/02_processed"),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute full generator setup: extract, validate, regularize, save.

    Args:
        input_dir: Directory containing df_latent_seed.csv.
        output_dir: Directory for .npy artifact output.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Mean vector, covariance matrix, M datasets.
    """
    config: dict[str, Any] = load_config()
    m_datasets: int = config["pipeline"]["m_datasets"]

    latent_path = input_dir / "df_latent_seed.csv"
    df_latent = load_latent_seed(latent_path)
    validate_latent_shape(df_latent)

    mean_vector, cov_matrix = extract_distribution_params(df_latent)
    cov_matrix = regularize_covariance(df_latent.to_numpy(dtype=np.float64), cov_matrix)

    save_artifacts(mean_vector, cov_matrix, output_dir)

    logger.info(f"Mean vector shape: {mean_vector.shape} (expected: (5,))")
    logger.info(f"Covariance matrix shape: {cov_matrix.shape} (expected: (5, 5))")
    logger.info(f"M target datasets: {m_datasets}")
    logger.info("Generator setup completed successfully.")

    return mean_vector, cov_matrix, m_datasets


if __name__ == "__main__":
    run_generator_setup()
