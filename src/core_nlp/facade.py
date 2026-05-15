# src/core_nlp/facade.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.core_nlp.mapping_output import (
    build_text_mapping,
    export_mapping_csv,
    save_text_mapping,
)
from src.core_nlp.orchestrator import NLPPreprocessorOrchestrator
from src.core_nlp.serializer import save_processed


def run_nlp_preprocessing(config_path: str = "config/pipeline_config.yaml") -> Dict[str, pd.DataFrame]:
    """Facade: Run NLP preprocessing pipeline on all configured text columns.

    Reads config, loads data from both report_master and synthetic paths,
    runs 4-stage preprocessing per text column, persists results, and
    generates text mapping outputs to reports/text_preprocessing/.

    Args:
        config_path: Path to pipeline_config.yaml.

    Returns:
        Dict mapping column key -> processed DataFrame.
    """
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

    report_master_dir = Path(paths["report_master"])
    synthetic_dir = Path(paths["synthetic"])
    processed_dir = Path(paths["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # PERBAIKAN: Gunakan root dir relatif terhadap config/paths agar Pytest tidak bingung
    project_root = report_master_dir.parent.parent
    mapping_dir = project_root / "reports" / "text_preprocessing"
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

        # Persist processed DataFrame
        out_path = processed_dir / f"{output_prefix}_{col_key}.csv"
        save_processed(df_processed, str(out_path))

        # Persist mapping outputs (new)
        mapping = build_text_mapping(df_processed, col_name)
        save_text_mapping(mapping, str(mapping_dir / f"mapping_{col_key}.json"))
        export_mapping_csv(df_processed, col_name, str(mapping_dir / f"mapping_{col_key}.csv"))

    return results
