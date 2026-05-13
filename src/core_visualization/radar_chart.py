# src/core_visualization/radar_chart.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core_visualization.constants import (
    RADAR_DPI,
    RADAR_FIGSIZE,
    RADAR_FILL_ALPHA,
    RADAR_GRID_VISIBLE,
    RADAR_LINE_WIDTH,
    RADAR_MARKER_SIZE,
    RADAR_YLIM_MAX,
    RADAR_YLIM_MIN,
)


def compute_tcr_scores(
    df: pd.DataFrame,
    variable_cols: List[str],
    likert_max: float = 5.0,
) -> Dict[str, float]:
    """Compute Tingkat Capaian Responden (TCR) as percentage for each variable.

    TCR = (Mean / Likert_Max) * 100

    Args:
        df: Dataframe containing latent variable columns.
        variable_cols: List of column names to compute TCR for.
        likert_max: Maximum Likert scale value (default: 5.0).

    Returns:
        dict: Mapping of variable name -> TCR percentage.
    """
    tcr: Dict[str, float] = {}
    for col in variable_cols:
        if col in df.columns:
            mean_val: float = float(df[col].mean())
            tcr[col] = round((mean_val / likert_max) * 100, 2)
    return tcr


def _prepare_polar_data(
    labels: List[str],
    values: List[float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for polar plot by closing the polygon loop.

    Args:
        labels: Variable names.
        values: TCR percentage values.

    Returns:
        tuple: (theta angles, values with closing point, labels with closing point).
    """
    n_vars: int = len(labels)
    angles: np.ndarray = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    values_closed: List[float] = values + [values[0]]
    angles_closed: List[float] = angles + [angles[0]]
    labels_closed: List[str] = labels + [labels[0]]

    return np.array(angles_closed), np.array(values_closed), labels_closed


def construct_radar_chart(
    tcr_scores: Dict[str, float],
    title: str = "Tingkat Capaian Responden (TCR)",
    color: str = "#3498DB",
    alpha: float = RADAR_FILL_ALPHA,
) -> plt.Figure:
    """Construct a polar radar chart for TCR visualization.

    Args:
        tcr_scores: Dictionary of variable -> TCR percentage.
        title: Chart title.
        color: Line and fill color hex code.
        alpha: Fill transparency (0.0 - 1.0).

    Returns:
        matplotlib.figure.Figure: Configured radar chart figure.
    """
    labels: List[str] = list(tcr_scores.keys())
    values: List[float] = list(tcr_scores.values())

    angles, values_closed, _ = _prepare_polar_data(labels, values)

    fig: plt.Figure = plt.figure(figsize=RADAR_FIGSIZE)
    ax: plt.Axes = fig.add_subplot(111, polar=True)

    ax.plot(
        angles,
        values_closed,
        color=color,
        linewidth=RADAR_LINE_WIDTH,
        marker="o",
        markersize=RADAR_MARKER_SIZE,
        label="TCR",
    )
    ax.fill(angles, values_closed, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(RADAR_YLIM_MIN, RADAR_YLIM_MAX)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax.grid(RADAR_GRID_VISIBLE, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add value labels at each point
    for angle, val in zip(angles[:-1], values):
        ax.annotate(
            f"{val:.1f}%",
            xy=(angle, val),
            xytext=(angle, val + 5),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def export_radar_chart(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = RADAR_DPI,
) -> None:
    """Export radar chart figure to PNG.

    Args:
        fig: Matplotlib figure.
        output_path: Target file path.
        dpi: Resolution in dots per inch.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
