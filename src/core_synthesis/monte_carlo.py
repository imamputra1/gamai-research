# src/core_synthesis/monte_carlo.py
from typing import Any

import numpy as np
import pandas as pd


def synthesize_likert_monte_carlo(
    seed_df: pd.DataFrame,
    n_target: int,
    random_seed: int,
    likert_min: int = 1,
    likert_max: int = 5,
) -> pd.DataFrame:
    """Generate synthetic Likert-scale data via Monte Carlo from empirical distribution.

    Args:
        seed_df: Source DataFrame with Likert columns.
        n_target: Target row count (N).
        random_seed: Absolute seed for reproducibility.
        likert_min: Minimum scale value.
        likert_max: Maximum scale value.

    Returns:
        pd.DataFrame: Synthetic data of shape (n_target, n_cols).
    """
    rng = np.random.default_rng(random_seed)
    synthetic_data: dict[str, np.ndarray] = {}

    for col in seed_df.columns:
        probs = seed_df[col].value_counts(normalize=True).sort_index()
        values = rng.choice(
            probs.index.to_numpy(),
            size=n_target,
            p=probs.to_numpy(),
        )
        synthetic_data[col] = values

    syn_df = pd.DataFrame(synthetic_data)
    syn_df = syn_df.clip(likert_min, likert_max).astype(int)
    return syn_df


def compute_correlation_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute Pearson correlation matrix.

    Args:
        df: Numeric DataFrame.

    Returns:
        np.ndarray: Correlation matrix of shape (n_cols, n_cols).
    """
    return df.corr(method="pearson").to_numpy()


def validate_correlation_preservation(
    seed_corr: np.ndarray,
    syn_corr: np.ndarray,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    """Validate absolute deviation between seed and synthetic correlation matrices.

    Args:
        seed_corr: Seed correlation matrix (shape: k, k).
        syn_corr: Synthetic correlation matrix (shape: k, k).
        tolerance: Maximum allowed absolute deviation.

    Returns:
        dict[str, Any]: {'is_valid': bool, 'max_deviation': float, 'mean_deviation': float}.
    """
    diff = np.abs(seed_corr - syn_corr)
    np.fill_diagonal(diff, 0.0)

    max_dev = float(np.nanmax(diff))
    mean_dev = float(np.nanmean(diff))

    return {
        "is_valid": max_dev <= tolerance,
        "max_deviation": max_dev,
        "mean_deviation": mean_dev,
    }
