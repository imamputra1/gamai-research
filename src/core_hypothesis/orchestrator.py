# src/core_hypothesis/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core_hypothesis.evaluator import (
    compile_summary_table,
    export_ringkasan_excel,
    filter_predictors,
    load_coefficients_substruktur1,
    map_hypothesis_labels,
)
from src.utils.config_loader import load_config


class HipotesisLangsungOrchestrator:
    """Orchestrator for H1-H3 direct hypothesis evaluation."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config = app_config
        block_d: dict[str, Any] = app_config["reproducibility"]["block_d"]

        self.input_dir: Path = Path(block_d["paths"]["output_tables"])
        self.output_dir: Path = self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Path:
        """Execute H1-H3 evaluation pipeline.

        Returns:
            Path: Path to exported ringkasan_hipotesis_langsung.xlsx.
        """
        # 13.1.1: Load coefficients
        input_path = self.input_dir / "hasil_regresi_1.xlsx"
        df_raw = load_coefficients_substruktur1(input_path)

        # 13.1.2: Filter predictors
        df_pred = filter_predictors(df_raw)

        # 13.1.3: Map hypothesis labels
        df_mapped = map_hypothesis_labels(df_pred)

        # 13.1.4: Compile summary
        df_summary = compile_summary_table(df_mapped)

        # 13.1.5: Export
        output_path = self.output_dir / "ringkasan_hipotesis_langsung.xlsx"
        export_ringkasan_excel(df_summary, output_path)

        print(
            f"INFO: Evaluasi Hipotesis Langsung (H1-H3) selesai. "
            f"Output: {output_path}"
        )
        return output_path


def run_fase_13_1(config_path: str = "config/pipeline_config.yaml") -> None:
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator = HipotesisLangsungOrchestrator(app_config)
    orchestrator.execute()
