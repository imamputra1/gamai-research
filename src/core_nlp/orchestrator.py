# src/core_nlp/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from dotenv import load_dotenv
import pandas as pd
import yaml
from collections import Counter

from src.core_nlp.aggregator import (
    build_llm_synthesis_prompt,
    build_summary_matrix,
    compute_overall_sentiment,
    merge_dataframes_by_index,
    persist_master_aggregated,
    persist_summary_matrix,
    prepare_dimension_subset,
)
from src.core_nlp.constants import (
    COL_OVERALL_LABEL,
    COL_OVERALL_SCORE,
    COL_TEXT_CLEANED,
    COL_TEXT_FILTERED,
    COL_TEXT_FINAL,
    COL_TEXT_NORMALIZED,
    DEFAULT_SLANG_DICT,
    DEMOGRAPHIC_COLS,
    DIR_REPORTS_FIGURES,
    DIR_REPORTS_TEXT_PREPROCESSING,
    QUESTION_KEYS,
    QUESTION_LABELS,
    SENTIMENT_SCORE_MAP,
    SUFFIX_CONFIDENCE,
    SUFFIX_FINAL_TEXT,
    SUFFIX_SENTIMENT,
    SUMMARY_MATRIX_COLS,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_XLSX,
    DEFAULT_SHEET_NAME,
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
from src.core_nlp.sentiment_engine import SentimentAnalyzer
from src.core_nlp.visualizer import generate_ngram_barchart, generate_wordcloud
from src.ai_integration.indobert_client import IndoBERTClient


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


class NLPSentimentOrchestrator:
    """Orchestrator untuk Sentiment Analysis (15.3)."""

    def __init__(
        self,
        client: Optional[IndoBERTClient] = None,
        batch_size: int = 16,
    ) -> None:
        self.analyzer: SentimentAnalyzer = SentimentAnalyzer(client=client, batch_size=batch_size)

    def analyze(
        self,
        df: pd.DataFrame,
        text_col: str,
        output_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """Eksekusi sentiment analysis pada DataFrame."""
        return self.analyzer.analyze_dataframe(df, text_col, output_prefix)


class NLPAggregatorOrchestrator:
    """Orchestrator untuk Master Aggregation & Synthesis (15.4)."""

    def __init__(
        self,
        question_keys: Optional[List[str]] = None,
        question_labels: Optional[Dict[str, str]] = None,
        demographic_cols: Optional[List[str]] = None,
        score_map: Optional[Dict[str, int]] = None,
        suffix_sentiment: str = SUFFIX_SENTIMENT,
        suffix_confidence: str = SUFFIX_CONFIDENCE,
        suffix_final: str = SUFFIX_FINAL_TEXT,
        col_overall_score: str = COL_OVERALL_SCORE,
        col_overall_label: str = COL_OVERALL_LABEL,
        summary_matrix_cols: List[str] = SUMMARY_MATRIX_COLS,
    ) -> None:
        self.question_keys: List[str] = question_keys if question_keys is not None else QUESTION_KEYS
        self.question_labels: Dict[str, str] = question_labels if question_labels is not None else QUESTION_LABELS
        self.demographic_cols: List[str] = demographic_cols if demographic_cols is not None else DEMOGRAPHIC_COLS
        self.score_map: Dict[str, int] = score_map if score_map is not None else SENTIMENT_SCORE_MAP
        self.suffix_sentiment: str = suffix_sentiment
        self.suffix_confidence: str = suffix_confidence
        self.suffix_final: str = suffix_final
        self.col_overall_score: str = col_overall_score
        self.col_overall_label: str = col_overall_label
        self.summary_matrix_cols: List[str] = summary_matrix_cols

    def run(
        self,
        dfs: Dict[str, pd.DataFrame],
        ngram_data: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full aggregation pipeline.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (master_aggregated, summary_matrix)
        """
        merged: pd.DataFrame = merge_dataframes_by_index(dfs, self.question_keys)
        scored: pd.DataFrame = compute_overall_sentiment(
            merged,
            self.question_keys,
            self.score_map,
            self.col_overall_score,
            self.col_overall_label,
        )
        summary: pd.DataFrame = build_summary_matrix(
            scored,
            self.question_keys,
            self.question_labels,
            self.score_map,
            self.summary_matrix_cols,
            ngram_data,
        )
        return scored, summary


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


def run_nlp_sentiment(config_path: str = "config/pipeline_config.yaml") -> Dict[str, pd.DataFrame]:
    """Facade: Run sentiment analysis on preprocessed outputs."""
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    paths: Dict[str, str] = config["paths"]
    nlp_cfg: Dict[str, Any] = config.get("nlp", {})

    text_columns: Dict[str, str] = nlp_cfg.get("text_columns", {})
    output_prefix: str = nlp_cfg.get("output_prefix", "nlp_preprocessed")

    sentiment_cfg: Dict[str, Any] = nlp_cfg.get("sentiment", {})
    batch_size: int = sentiment_cfg.get("batch_size", 16)
    source_stage: str = sentiment_cfg.get("source_stage", "normalized")

    processed_dir = Path(paths["processed"])

    client: IndoBERTClient = IndoBERTClient()
    sentiment_orch = SentimentAnalyzer(client=client, batch_size=batch_size)

    all_results: Dict[str, pd.DataFrame] = {}

    for col_key in text_columns.keys():
        csv_path = processed_dir / f"{output_prefix}_{col_key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {csv_path}")

        df_processed = pd.read_csv(csv_path)

        source_col = f"text_{source_stage}" if source_stage in ("cleaned", "normalized") else COL_TEXT_NORMALIZED
        if source_col not in df_processed.columns:
            source_col = COL_TEXT_NORMALIZED

        df_sentiment = sentiment_orch.analyze_dataframe(
            df_processed,
            text_col=source_col,
            output_prefix=col_key,
        )

        df_sentiment.to_csv(csv_path, index=False, encoding="utf-8")
        all_results[col_key] = df_sentiment

    return all_results


def run_nlp_aggregation(
    config_path: str = "config/pipeline_config.yaml",
    ngram_data: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Facade: Run NLP aggregation pipeline (15.4.1 - 15.4.3).

    Reads post-sentiment preprocessed CSVs, merges them into a master
    DataFrame, computes overall sentiment, and exports summary matrix.

    Args:
        config_path: Path ke pipeline_config.yaml.
        ngram_data: Optional frequential data for top-ngram injection.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (master_df, summary_df)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    paths: Dict[str, str] = config["paths"]
    nlp_cfg: Dict[str, Any] = config.get("nlp", {})

    text_columns: Dict[str, str] = nlp_cfg.get("text_columns", {})
    output_prefix: str = nlp_cfg.get("output_prefix", "nlp_preprocessed")

    processed_dir = Path(paths["processed"])

    dfs: Dict[str, pd.DataFrame] = {}
    is_first: bool = True

    for col_key in text_columns.keys():
        csv_path = processed_dir / f"{output_prefix}_{col_key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {csv_path}")

        df_full: pd.DataFrame = pd.read_csv(csv_path)

        if COL_TEXT_FINAL in df_full.columns:
            target_name = str(col_key) + "_" + str(SUFFIX_FINAL_TEXT)
            target_name = target_name.replace("__", "_")
            df_full.rename(
                columns={COL_TEXT_FINAL: target_name},
                inplace=True
            )

        df_subset: pd.DataFrame = prepare_dimension_subset(
            df_full,
            col_key,
            DEMOGRAPHIC_COLS,
            SUFFIX_SENTIMENT,
            SUFFIX_CONFIDENCE,
            SUFFIX_FINAL_TEXT,
            include_demographics=is_first,
        )
        is_first = False
        dfs[col_key] = df_subset

    def build_ngram_df(text_list, n, top_k=3):
        """Mengekstrak N-Gram menggunakan Python murni agar tidak bergantung pada modul lain."""
        words = []
        for text in text_list:
            if pd.isna(text): continue
            tokens = str(text).split()
            if n == 1:
                words.extend(tokens)
            else:
                ngrams = zip(*[tokens[i:] for i in range(n)])
                words.extend([" ".join(gram) for gram in ngrams])
        
        # Hitung kemunculan dan ubah ke format DataFrame yang diterima Aggregator
        counts = Counter(words).most_common(top_k)
        return pd.DataFrame([{"term": t, "count": c} for t, c in counts])

    ngram_data = {}
    for col_key, df_subset in dfs.items():
        # Susun ulang nama kolom target dengan aman
        target_col = str(col_key) + "_" + str(SUFFIX_FINAL_TEXT)
        target_col = target_col.replace("__", "_")
        
        if target_col in df_subset.columns:
            texts = df_subset[target_col].tolist()
            # Kalkulasi Unigram (1), Bigram (2), Trigram (3) secara on-the-fly!
            ngram_data[col_key] = {
                1: build_ngram_df(texts, n=1, top_k=3),
                2: build_ngram_df(texts, n=2, top_k=3),
                3: build_ngram_df(texts, n=3, top_k=3)
            }

    orch = NLPAggregatorOrchestrator()
    master_df, summary_df = orch.run(dfs, ngram_data=ngram_data)

    master_path: str = str(processed_dir / "nlp_master_aggregated.csv")
    summary_path: str = str(Path(paths.get("reports_tables", "reports/tables")) / "nlp_summary_matrix.xlsx")

    persist_master_aggregated(master_df, master_path)
    persist_summary_matrix(summary_df, summary_path, DEFAULT_SHEET_NAME)

    return master_df, summary_df

def run_nlp_llm_synthesis(
    config_path: str = "config/pipeline_config.yaml",
    client: Optional[Any] = None,
) -> Optional[str]:
    """Facade: Generate academic narrative via LLM (15.4.4) dan simpan ke file."""
    
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    paths: Dict[str, str] = config["paths"]
    summary_path = Path(paths.get("reports_tables", "reports/tables")) / "nlp_summary_matrix.xlsx"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary matrix not found: {summary_path}")

    summary_df: pd.DataFrame = pd.read_excel(summary_path, sheet_name=DEFAULT_SHEET_NAME)
    prompt: str = build_llm_synthesis_prompt(summary_df)

    # Inisialisasi client dari folder ai_integration jika tidak di-inject
    if client is None:
        from src.ai_integration.open_router_client import OpenRouterClient
        client = OpenRouterClient()

    # Eksekusi AI Generation
    narrative = client.generate(prompt)

    if not narrative:
        print("[FAILED] LLM mengembalikan teks kosong.")
        return None

    # Simpan hasil ketikan AI ke file Markdown
    out_dir = Path("reports/synthesis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Bab_4_Sintesis_Kualitatif.md"
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(narrative)

    print(f"[SUCCESS] Draft Bab 4 berhasil ditulis dan disimpan di:\n -> {out_file.absolute()}")
    
    return narrative
