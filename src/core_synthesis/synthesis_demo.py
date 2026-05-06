# src/core_synthesis/07_synthesis_demo.py
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("synthesis_demo")


def load_demo_seed(path: Path) -> pd.DataFrame:
    """Load original demographic seed dataframe.

    Args:
        path: Path to df_demo.csv.

    Returns:
        pd.DataFrame: Demographic data (shape: 20, 5).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Demo seed not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Demo seed loaded: {df.shape}")
    return df


def extract_empirical_probabilities(df: pd.DataFrame) -> Dict[str, Dict[str, List[Any]]]:
    """Extract weighted probability distributions per demographic column.

    Args:
        df: Demographic seed dataframe.

    Returns:
        Dict[str, Dict[str, List[Any]]]: Mapping column -> {'kategori': [...], 'probabilitas': [...]}.
    """
    demo_probs: Dict[str, Dict[str, List[Any]]] = {}

    for col in df.columns:
        probs = df[col].value_counts(normalize=True)
        demo_probs[col] = {
            "kategori": probs.index.tolist(),
            "probabilitas": probs.values.tolist(),
        }

    sample_col = list(demo_probs.keys())[0]
    logger.info(
        f"Distribusi Probabilitas Kolom '{sample_col}' terekstrak: "
        f"{dict(zip(demo_probs[sample_col]['kategori'], [round(p, 3) for p in demo_probs[sample_col]['probabilitas']]))}"
    )

    return demo_probs


def generate_synthetic_demo(
    demo_probs: Dict[str, Dict[str, List[Any]]],
    n_target: int,
    random_state: int,
) -> pd.DataFrame:
    """Generate synthetic demographic data via weighted random sampling.

    Args:
        demo_probs: Empirical probability distributions per column.
        n_target: Number of synthetic respondents (N).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic demographic data (shape: N, k).
    """
    rng = np.random.default_rng(random_state)
    df_syn: pd.DataFrame = pd.DataFrame()

    for col, dist in demo_probs.items():
        synthetic_col: np.ndarray = rng.choice(
            a=dist["kategori"],
            size=n_target,
            p=dist["probabilitas"],
        )
        df_syn[col] = synthetic_col

    return df_syn


def validate_demo_shape(df: pd.DataFrame, expected_rows: int, expected_cols: int = 5) -> None:
    """Validate synthetic demographic dataframe dimensions.

    Args:
        df: Synthetic demographic dataframe.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If shape mismatch.
    """
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Demo synthetic shape harus ({expected_rows}, {expected_cols}). "
            f"Ditemukan: {df.shape}"
        )


def save_synthetic_demo(df: pd.DataFrame, output_path: Path) -> None:
    """Export synthetic demographic dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_synthesis_demo(
    processed_dir: Path = Path("data/02_processed"),
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute full demographic synthesis pipeline.

    Args:
        processed_dir: Directory containing df_demo.csv.
        synthetic_dir: Directory for df_demo_syn_*.csv output.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)
    base_seed: int = pipeline_cfg.get("random_seed", 42)

    demo_path = processed_dir / "df_demo.csv"
    df_demo_seed = load_demo_seed(demo_path)

    demo_probs = extract_empirical_probabilities(df_demo_seed)

    for m in range(1, m_datasets + 1):
        random_state = base_seed + m
        df_demo_syn = generate_synthetic_demo(demo_probs, n_target, random_state)

        validate_demo_shape(df_demo_syn, expected_rows=n_target)

        output_path = synthetic_dir / f"df_demo_syn_{m}.csv"
        save_synthetic_demo(df_demo_syn, output_path)

    logger.info(
        f"Successfully exported {m_datasets} demographic datasets "
        f"({n_target}x{df_demo_seed.shape[1]}) to {synthetic_dir}"
    )
    logger.info("Demographic synthesis completed successfully.")


if __name__ == "__main__":
    run_synthesis_demo()
