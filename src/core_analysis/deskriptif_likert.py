# src/core_analysis/09_2_deskriptif_likert.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("deskriptif_likert")

LIKERT_INTERVALS: list[float] = [1.0, 1.81, 2.61, 3.41, 4.21, 5.01]
INTERVAL_LABELS: list[str] = [
    "Sangat Buruk / Sangat Tidak Setuju",
    "Buruk / Tidak Setuju",
    "Cukup / Netral",
    "Baik / Setuju",
    "Sangat Baik / Sangat Setuju",
]

# Indices RELATIVE to df_likert (0-24), NOT df_master
CONSTRUCT_SLICES: dict[str, slice] = {
    "People": slice(0, 5),
    "Process": slice(5, 10),
    "Physical_Evidence": slice(10, 15),
    "Experience_Value": slice(15, 20),
    "Minat_Kunjung": slice(20, 25),
}


def load_master_report(path: Path) -> pd.DataFrame:
    """Load locked master dataset.

    Args:
        path: Path to df_report_master.csv.

    Returns:
        pd.DataFrame: Master dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Master report not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Master report loaded: {df.shape}")
    return df


def extract_likert_block(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 25 Likert columns from master dataframe.

    Args:
        df: Master dataframe (shape: 100, 35).

    Returns:
        pd.DataFrame: Likert block (shape: 100, 25).
    """
    likert_df: pd.DataFrame = df.iloc[:, 5:30].copy()
    return likert_df


def compute_indicator_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for 25 Likert indicators.

    Args:
        df: Likert dataframe (shape: 100, 25).

    Returns:
        pd.DataFrame: Descriptive statistics table.
    """
    desc: pd.DataFrame = df.describe().T
    desc = desc[["mean", "std", "min", "max"]]
    desc.columns = ["Mean", "Std. Dev", "Min", "Max"]
    desc = desc.round(2)
    return desc


def compute_super_feature_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-wise means per super-feature construct.

    Args:
        df: Likert dataframe (shape: 100, 25).

    Returns:
        pd.DataFrame: Super-feature mean dataframe (shape: 100, 5).
    """
    constructs: dict[str, pd.Series] = {}
    for name, slc in CONSTRUCT_SLICES.items():
        constructs[name] = df.iloc[:, slc].mean(axis=1)

    df_latent: pd.DataFrame = pd.DataFrame(constructs)
    return df_latent


def categorize_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Likert interval categorization to super-feature means.

    Args:
        df: Super-feature mean dataframe (shape: 100, 5).

    Returns:
        pd.DataFrame: Categorized dataframe with ordinal labels.
    """
    df_cat: pd.DataFrame = pd.DataFrame(index=df.index)

    for col in df.columns:
        df_cat[col] = pd.cut(
            df[col],
            bins=LIKERT_INTERVALS,
            labels=INTERVAL_LABELS,
            right=False,
            include_lowest=True,
        )

    return df_cat


def build_descriptive_table(df_latent: pd.DataFrame) -> pd.DataFrame:
    """Build summary statistics table for super-features.

    Args:
        df_latent: Super-feature mean dataframe.

    Returns:
        pd.DataFrame: Summary table with mean, std, min, max, and interval label.
    """
    rows: list[dict[str, Any]] = []

    for col in df_latent.columns:
        mean_val: float = float(df_latent[col].mean())
        std_val: float = float(df_latent[col].std())
        min_val: float = float(df_latent[col].min())
        max_val: float = float(df_latent[col].max())

        cat_label: str = pd.cut(
            [mean_val],
            bins=LIKERT_INTERVALS,
            labels=INTERVAL_LABELS,
            right=False,
            include_lowest=True,
        )[0]

        rows.append(
            {
                "Variabel Laten": col,
                "Mean": round(mean_val, 2),
                "Std. Dev": round(std_val, 2),
                "Min": round(min_val, 2),
                "Max": round(max_val, 2),
                "Keterangan (Interval)": str(cat_label),
            }
        )

    return pd.DataFrame(rows)


def build_distribution_table(df_cat: pd.DataFrame) -> pd.DataFrame:
    """Build frequency distribution table per super-feature category.

    Args:
        df_cat: Categorized super-feature dataframe.

    Returns:
        pd.DataFrame: Long-format distribution table.
    """
    rows: list[dict[str, Any]] = []

    for col in df_cat.columns:
        counts: pd.Series = df_cat[col].value_counts().sort_index()
        total: int = int(counts.sum())

        for cat in INTERVAL_LABELS:
            freq: int = int(counts.get(cat, 0))
            pct: float = round(freq / total * 100, 2) if total > 0 else 0.0
            rows.append(
                {
                    "Variabel Laten": col if cat == INTERVAL_LABELS[0] else "",
                    "Kategori": cat,
                    "Frekuensi (N)": freq,
                    "Persentase (%)": pct,
                }
            )

    return pd.DataFrame(rows)


def export_to_excel(
    df_desc: pd.DataFrame,
    df_dist: pd.DataFrame,
    output_desc: Path,
    output_dist: Path,
) -> None:
    """Export descriptive and distribution tables to Excel.

    Args:
        df_desc: Descriptive statistics table.
        df_dist: Distribution frequency table.
        output_desc: Path for descriptive table.
        output_dist: Path for distribution table.
    """
    output_desc.parent.mkdir(parents=True, exist_ok=True)
    output_dist.parent.mkdir(parents=True, exist_ok=True)

    df_desc.to_excel(output_desc, index=False, sheet_name="Deskriptif")
    df_dist.to_excel(output_dist, index=False, sheet_name="Distribusi")

    logger.info(f"Tabel deskriptif exported: {output_desc}")
    logger.info(f"Tabel distribusi exported: {output_dist}")


def run_deskriptif_likert(
    master_path: Path = Path("data/04_report_master/df_report_master.csv"),
    output_desc: Path = Path("reports/tables/tabel_deskriptif_indikator.xlsx"),
    output_dist: Path = Path("reports/tables/tabel_kategorisasi_laten.xlsx"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute full descriptive Likert analysis pipeline.

    Args:
        master_path: Path to locked master dataset.
        output_desc: Path for descriptive statistics Excel.
        output_dist: Path for distribution frequency Excel.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (descriptive_table, distribution_table).
    """
    df_master = load_master_report(master_path)
    df_likert = extract_likert_block(df_master)

    df_desc_ind = compute_indicator_descriptives(df_likert)
    logger.info(f"Indicator descriptives computed: {df_desc_ind.shape}")

    df_latent = compute_super_feature_means(df_likert)
    df_cat = categorize_intervals(df_latent)

    df_desc_table = build_descriptive_table(df_latent)
    df_dist_table = build_distribution_table(df_cat)

    export_to_excel(df_desc_table, df_dist_table, output_desc, output_dist)

    logger.info("Deskriptif Likert analysis completed successfully.")
    return df_desc_table, df_dist_table


if __name__ == "__main__":
    run_deskriptif_likert()
