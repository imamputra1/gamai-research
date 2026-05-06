# src/core_synthesis/05_hierarchical_sim_likert.py
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("hierarchical_sim_likert")


def load_likert_seed(path: Path) -> pd.DataFrame:
    """Load original Likert seed to extract indicator standard deviations.

    Args:
        path: Path to df_likert.csv.

    Returns:
        pd.DataFrame: Original Likert dataframe (shape: 20, 25).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Likert seed not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Likert seed loaded: {df.shape}")
    return df


def build_construct_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map Likert columns to 5 latent constructs via positional slicing.

    Args:
        df: Likert dataframe with exactly 25 columns.

    Returns:
        Dict[str, List[str]]: Mapping construct name -> ordered column names.
    """
    mapping: Dict[str, List[str]] = {
        "People": df.columns[0:5].tolist(),
        "Process": df.columns[5:10].tolist(),
        "Physical_Evidence": df.columns[10:15].tolist(),
        "Experience_Value": df.columns[15:20].tolist(),
        "Minat_Kunjung": df.columns[20:25].tolist(),
    }
    return mapping


def extract_indicator_std(df: pd.DataFrame) -> pd.Series:
    """Compute per-indicator standard deviation from original seed.

    Args:
        df: Original Likert dataframe.

    Returns:
        pd.Series: Standard deviations indexed by column name.
    """
    std_series: pd.Series = df.std(ddof=1)
    logger.info(f"Indicator std extracted for {len(std_series)} columns")
    return std_series


def load_latent_dataset(path: Path) -> pd.DataFrame:
    """Load a single synthetic latent dataset.

    Args:
        path: Path to df_latent_syn_{m}.csv.

    Returns:
        pd.DataFrame: Latent synthetic dataframe (shape: N, 5).
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df


def expand_latent_to_likert(
    df_latent: pd.DataFrame,
    construct_mapping: Dict[str, List[str]],
    indicator_std: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Expand 5 latent scores into 25 continuous Likert indicators with noise.

    Args:
        df_latent: Synthetic latent dataframe (shape: N, 5).
        construct_mapping: Construct-to-indicators mapping.
        indicator_std: Per-indicator standard deviations from seed.
        random_state: Seed for noise reproducibility.

    Returns:
        pd.DataFrame: Continuous Likert synthetic data (shape: N, 25).
    """
    rng = np.random.default_rng(random_state)
    df_cont: pd.DataFrame = pd.DataFrame(index=df_latent.index)

    for construct_name, indicators in construct_mapping.items():
        latent_scores: pd.Series = df_latent[construct_name]
        for ind in indicators:
            sigma: float = float(indicator_std[ind])
            noise: np.ndarray = rng.normal(loc=0.0, scale=sigma, size=len(df_latent))
            df_cont[ind] = latent_scores.to_numpy() + noise

    return df_cont


def save_continuous_likert(df: pd.DataFrame, output_path: Path) -> None:
    """Export continuous Likert dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def validate_continuous_shape(df: pd.DataFrame, expected_rows: int, expected_cols: int = 25) -> None:
    """Validate continuous Likert dataframe dimensions.

    Args:
        df: Dataframe to validate.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If shape mismatch.
    """
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Continuous Likert shape harus ({expected_rows}, {expected_cols}). "
            f"Ditemukan: {df.shape}"
        )


def run_hierarchical_sim_likert(
    processed_dir: Path = Path("data/02_processed"),
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute full hierarchical Level 2 simulation: expand latent to 25 indicators.

    Args:
        processed_dir: Directory containing df_likert.csv.
        synthetic_dir: Directory containing df_latent_syn_*.csv and output destination.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)
    base_seed: int = pipeline_cfg.get("random_seed", 42)

    likert_path = processed_dir / "df_likert.csv"
    df_likert_seed = load_likert_seed(likert_path)

    construct_mapping = build_construct_mapping(df_likert_seed)
    indicator_std = extract_indicator_std(df_likert_seed)

    for m in range(1, m_datasets + 1):
        latent_path = synthetic_dir / f"df_latent_syn_{m}.csv"
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent dataset missing: {latent_path}")

        df_latent = load_latent_dataset(latent_path)

        random_state = base_seed + m
        df_cont = expand_latent_to_likert(
            df_latent=df_latent,
            construct_mapping=construct_mapping,
            indicator_std=indicator_std,
            random_state=random_state,
        )

        validate_continuous_shape(df_cont, expected_rows=n_target)

        output_path = synthetic_dir / f"df_likert_cont_syn_{m}.csv"
        save_continuous_likert(df_cont, output_path)

    logger.info(f"Successfully expanded {m_datasets} latent datasets into 25 continuous indicators each")
    logger.info(f"Final dimensions per dataset: ({n_target}, 25)")
    logger.info("Hierarchical Likert simulation (Level 2) completed successfully.")


if __name__ == "__main__":
    run_hierarchical_sim_likert()
