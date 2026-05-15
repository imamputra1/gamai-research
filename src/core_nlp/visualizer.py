# src/core_nlp/visualizer.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.core_nlp.constants import (
    DEFAULT_CHART_DPI,
    DEFAULT_CHART_FIGSIZE,
    DEFAULT_CHART_ORIENTATION,
    DEFAULT_WC_BG_COLOR,
    DEFAULT_WC_COLORMAP,
    DEFAULT_WC_DPI,
    DEFAULT_WC_HEIGHT,
    DEFAULT_WC_WIDTH,
)


def generate_wordcloud(
    df_freq: pd.DataFrame,
    output_path: str,
    background_color: str = DEFAULT_WC_BG_COLOR,
    colormap: str = DEFAULT_WC_COLORMAP,
    width: int = DEFAULT_WC_WIDTH,
    height: int = DEFAULT_WC_HEIGHT,
    dpi: int = DEFAULT_WC_DPI,
) -> None:
    """15.2.2: Render frequency table to Word Cloud image."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    freq_dict: Dict[str, int] = dict(zip(df_freq["term"], df_freq["frequency"]))
    wc = WordCloud(
        background_color=background_color,
        colormap=colormap,
        width=width,
        height=height,
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(width / 100, height / 100), dpi=dpi)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def generate_ngram_barchart(
    df_freq: pd.DataFrame,
    output_path: str,
    n: int = 2,
    figsize: tuple[int, int] = DEFAULT_CHART_FIGSIZE,
    dpi: int = DEFAULT_CHART_DPI,
    orientation: str = DEFAULT_CHART_ORIENTATION,
) -> None:
    """15.2.3: Render N-Gram frequency to horizontal bar chart."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)

    if orientation == "horizontal":
        # FIX: Assign hue to y variable, set legend=False (seaborn v0.14 deprecation)
        sns.barplot(data=df_freq, y="term", x="frequency", hue="term", palette="Blues_r", legend=False)
        plt.xlabel("Frequency")
        plt.ylabel("Term")
    else:
        sns.barplot(data=df_freq, x="term", y="frequency", hue="term", palette="Blues_r", legend=False)
        plt.xlabel("Term")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")

    plt.title(f"Top {len(df_freq)} {n}-Grams")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
