# src/core_analysis/profiling_demografi.py
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chisquare

from src.utils import setup_logger

logger = setup_logger("profiling_demografi")


def load_dataframes(
    master_path: Path,
    seed_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load master report and seed demographic dataframes.

    Args:
        master_path: Path to df_report_master.csv.
        seed_path: Path to df_demo.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_master, df_seed).

    Raises:
        FileNotFoundError: If either file is missing.
    """
    if not master_path.exists():
        raise FileNotFoundError(f"Master report not found: {master_path}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed demo not found: {seed_path}")

    df_master: pd.DataFrame = pd.read_csv(master_path)
    df_seed: pd.DataFrame = pd.read_csv(seed_path)

    logger.info(f"Master loaded: {df_master.shape} | Seed loaded: {df_seed.shape}")
    return df_master, df_seed


def validate_master_shape(df: pd.DataFrame) -> None:
    """Validate master dataframe has exactly 100 rows.

    Args:
        df: Master dataframe.

    Raises:
        ValueError: If row count != 100.
    """
    if df.shape[0] != 100:
        raise ValueError(
            f"Master dataset harus memiliki 100 baris. Ditemukan: {df.shape[0]}"
        )


def run_chi_square_goodness_of_fit(
    df_master: pd.DataFrame,
    df_seed: pd.DataFrame,
) -> pd.DataFrame:
    """Compute demographic frequencies and Chi-Square goodness-of-fit per variable.

    Iterates only over demo columns present in both dataframes.

    Args:
        df_master: Synthetic master dataframe (shape: 100, 35).
        df_seed: Original seed dataframe (shape: 20, 5).

    Returns:
        pd.DataFrame: Profiling table with frequencies, percentages, and p-values.
    """
    results: list[dict[str, Any]] = []

    demo_cols: list[str] = [c for c in df_seed.columns if c in df_master.columns]

    for col in demo_cols:
        f_obs: pd.Series = df_master[col].value_counts().sort_index()

        seed_props: pd.Series = df_seed[col].value_counts(normalize=True)
        f_exp: pd.Series = (seed_props * len(df_master)).reindex(
            f_obs.index, fill_value=0
        )

        common_idx: pd.Index = f_obs.index.intersection(f_exp.index)
        f_obs_aligned: pd.Series = f_obs.reindex(common_idx, fill_value=0)
        f_exp_aligned: pd.Series = f_exp.reindex(common_idx, fill_value=0)

        chi2_stat: float
        p_value: float
        chi2_stat, p_value = chisquare(f_obs_aligned, f_exp_aligned)

        for idx, cat in enumerate(f_obs.index):
            results.append(
                {
                    "Variabel": col if idx == 0 else "",
                    "Kategori": cat,
                    "Frekuensi (N)": int(f_obs[cat]),
                    "Persentase (%)": round(f_obs[cat] / len(df_master) * 100, 2),
                    "p-value Chi-Square": (
                        round(p_value, 4) if idx == 0 else None
                    ),
                    "Status": (
                        "Valid" if p_value > 0.05 else "Tidak Valid"
                        if idx == 0
                        else None
                    ),
                }
            )

        status = "Valid" if p_value > 0.05 else "Tidak Valid"
        logger.info(f"{col} - p-value: {p_value:.4f} ({status})")

    df_profile: pd.DataFrame = pd.DataFrame(results)
    return df_profile


def export_profile_table(
    df_profile: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export profiling table to Excel.

    Args:
        df_profile: Profiling dataframe.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_profile.to_excel(output_path, index=False, sheet_name="Demografi")
    logger.info(f"Tabel demografi exported: {output_path}")


def run_profiling_demografi(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    seed_path: Path = Path("data/02_processed/df_demo.csv"),
    output_path: Path = Path("reports/tables/tabel_demografi.xlsx"),
) -> pd.DataFrame:
    """Execute full demographic profiling and Chi-Square validation pipeline.

    Args:
        master_path: Path to locked master dataset.
        seed_path: Path to original demographic seed.
        output_path: Path for Excel output.

    Returns:
        pd.DataFrame: Complete profiling table.
    """
    df_master, df_seed = load_dataframes(master_path, seed_path)

    validate_master_shape(df_master)

    df_profile = run_chi_square_goodness_of_fit(df_master, df_seed)

    export_profile_table(df_profile, output_path)

    logger.info("Tabel Karakteristik Responden sukses di-generate.")
    logger.info("Profiling demografi completed successfully.")

    return df_profile


if __name__ == "__main__":
    run_profiling_demografi()
