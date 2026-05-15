# src/core_nlp/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.core_nlp.constants import (
    COL_TEXT_CLEANED,
    COL_TEXT_FILTERED,
    COL_TEXT_FINAL,
    COL_TEXT_NORMALIZED,
    DEFAULT_SLANG_DICT,
    DIR_REPORTS_FIGURES,
    DIR_REPORTS_TEXT_PREPROCESSING,
)
from src.core_nlp.frequency_analyzer import get_all_ngram_levels, get_top_ngrams
from src.core_nlp.preprocessing import (
    build_text_mapping,
    clean_text,
    export_mapping_csv,
    normalize_slang,
    remove_stopwords,
    save_processed,
    save_text_mapping,
    stem_text,
)
from src.core_nlp.visualizer import generate_ngram_barchart, generate_wordcloud


class NLPPreprocessorOrchestrator:
    """Orchestrator NLP Preprocessing Pipeline (4 tahap)."""

    def __init__(
        self,
        slang_dict: Optional[Dict[str, str]] = None,
        custom_stopwords: Optional[List[str]] = None,
    ) -> None:
        self.slang_dict: Dict[str, str] = slang_dict if slang_dict is not None else DEFAULT_SLANG_DICT.copy()
        self.custom_stopwords: List[str] = custom_stopwords if custom_stopwords is not None else []

    def run(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Eksekusi pipeline 4 tahap pada DataFrame."""
        df_out: pd.DataFrame = df.copy()
        df_out[COL_TEXT_CLEANED] = clean_text(df_out[text_column])
        df_out[COL_TEXT_NORMALIZED] = normalize_slang(df_out[COL_TEXT_CLEANED], self.slang_dict)
        df_out[COL_TEXT_FILTERED] = remove_stopwords(df_out[COL_TEXT_NORMALIZED], self.custom_stopwords)
        df_out[COL_TEXT_FINAL] = stem_text(df_out[COL_TEXT_FILTERED])
        return df_out

    def run_series(self, series: pd.Series) -> pd.Series:
        """Eksekusi pipeline pada single pd.Series."""
        cleaned: pd.Series = clean_text(series)
        normalized: pd.Series = normalize_slang(cleaned, self.slang_dict)
        filtered: pd.Series = remove_stopwords(normalized, self.custom_stopwords)
        final: pd.Series = stem_text(filtered)
        return final


class NLPFrequentialOrchestrator:
    """Orchestrator untuk Frequential Analysis (15.2)."""

    def __init__(
        self,
        top_k: int = 20,
        ngram_ranges: Optional[List[int]] = None,
    ) -> None:
        self.top_k: int = top_k
        self.ngram_ranges: List[int] = ngram_ranges if ngram_ranges is not None else [1, 2, 3]

    def analyze(
        self,
        df: pd.DataFrame,
        text_col: str,
        output_prefix: str,
        reports_dir: str = DIR_REPORTS_FIGURES,
    ) -> Dict[int, pd.DataFrame]:
        """Eksekusi frequential analysis dan generate visualisasi."""
        results: Dict[int, pd.DataFrame] = {}
        for n in self.ngram_ranges:
            df_freq: pd.DataFrame = get_top_ngrams(df, text_col, n=n, top_k=self.top_k)
            results[n] = df_freq

            fig_dir = Path(reports_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)

            if n == 1:
                generate_wordcloud(df_freq, str(fig_dir / f"wordcloud_{output_prefix}.png"))
            else:
                generate_ngram_barchart(df_freq, str(fig_dir / f"ngram_{n}_{output_prefix}.png"), n=n)

        return results


def run_nlp_preprocessing(config_path: str = "config/pipeline_config.yaml") -> Dict[str, pd.DataFrame]:
    """Facade: Run NLP preprocessing pipeline on all configured text columns."""
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    paths: Dict[str, str] = config["paths"]
    nlp_cfg: Dict[str, Any] = config.get("nlp", {})

    text_columns: Dict[str, str] = nlp_cfg.get("text_columns", {})
    custom_stopwords: List[str] = nlp_cfg.get("custom_stopwords", [])
    custom_slang: Dict[str, str] = nlp_cfg.get("custom_slang", {})
    output_prefix: str = nlp_cfg.get("output_prefix", "nlp_preprocessed")

    report_master_dir = Path(paths["report_master"])
    synthetic_dir = Path(paths["synthetic"])
    processed_dir = Path(paths["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    project_root = report_master_dir.parent.parent
    mapping_dir = project_root / DIR_REPORTS_TEXT_PREPROCESSING
    mapping_dir.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(report_master_dir / "df_report_master.csv")
    df_teks = pd.read_csv(synthetic_dir / "df_teks_syn_2.csv")

    orch = NLPPreprocessorOrchestrator(
        slang_dict=custom_slang if custom_slang else None,
        custom_stopwords=custom_stopwords if custom_stopwords else None,
    )

    results: Dict[str, pd.DataFrame] = {}

    for col_key, col_name in text_columns.items():
        if col_name in df_master.columns:
            df_source = df_master
        elif col_name in df_teks.columns:
            df_source = df_teks
        else:
            raise KeyError(f"Column '{col_name}' not found in either dataset")

        df_processed = orch.run(df_source, text_column=col_name)
        results[col_key] = df_processed

        out_path = processed_dir / f"{output_prefix}_{col_key}.csv"
        save_processed(df_processed, str(out_path))

        mapping = build_text_mapping(df_processed, col_name)
        save_text_mapping(mapping, str(mapping_dir / f"mapping_{col_key}.json"))
        export_mapping_csv(df_processed, col_name, str(mapping_dir / f"mapping_{col_key}.csv"))

    return results


def run_nlp_frequential(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Dict[int, pd.DataFrame]]:
    """Facade: Run frequential analysis on preprocessed outputs."""
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    paths: Dict[str, str] = config["paths"]
    nlp_cfg: Dict[str, Any] = config.get("nlp", {})

    text_columns: Dict[str, str] = nlp_cfg.get("text_columns", {})
    output_prefix: str = nlp_cfg.get("output_prefix", "nlp_preprocessed")

    freq_cfg: Dict[str, Any] = nlp_cfg.get("frequential", {})
    top_k: int = freq_cfg.get("top_k", 20)
    ngram_ranges: List[int] = freq_cfg.get("ngram_ranges", [1, 2, 3])

    processed_dir = Path(paths["processed"])
    reports_dir = Path(paths.get("reports_figures", "reports/figures"))

    freq_orch = NLPFrequentialOrchestrator(top_k=top_k, ngram_ranges=ngram_ranges)

    all_results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for col_key in text_columns.keys():
        csv_path = processed_dir / f"{output_prefix}_{col_key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {csv_path}")

        df_processed = pd.read_csv(csv_path)
        results = freq_orch.analyze(
            df_processed,
            text_col=COL_TEXT_FINAL,
            output_prefix=col_key,
            reports_dir=str(reports_dir),
        )
        all_results[col_key] = results

    return all_results
