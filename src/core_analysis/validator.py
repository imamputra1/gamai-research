# src/core_analysis/validator.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def calculate_cronbach_alpha(df: pd.DataFrame) -> float:
    """Calculate Cronbach's Alpha for internal consistency.

    Args:
        df: Likert item DataFrame (shape: n_respondents, n_items).

    Returns:
        float: Alpha coefficient [0.0, 1.0].
    """
    n_items = df.shape[1]
    item_variances = df.var(axis=0, ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return 0.0

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def export_qq_plot(
    series: pd.Series,
    output_path: str | Path,
    title: str = "Q-Q Plot",
) -> None:
    """Generate and save Q-Q plot for normality assessment.

    Args:
        series: Numeric pandas Series.
        output_path: Destination file path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(series.dropna(), dist="norm", plot=ax)
    ax.set_title(title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
