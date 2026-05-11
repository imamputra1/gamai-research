# src/core_hypothesis/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core_reproducibility.config_builder import build_analisis_config
from src.core_reproducibility.schema import AnalisisConfig
from src.core_hypothesis.constants import (
    ANTECEDENT_EFFECTS_FILENAME,
    EVALUASI_HIPOTESIS_FILENAME,
    OUTPUT_TABLES_KEY,
    PATHS_KEY,
    SIGNIFIKANSI_ALPHA_KEY,
)
from src.core_hypothesis.evaluator import (
    compile_summary_table,
    evaluate_decisions,
    export_evaluasi_excel,
    filter_predictors,
    load_coefficients,
    map_hypothesis_labels,
)
from src.utils.config_loader import load_config


class HipotesisLangsungOrchestrator:
    """Orchestrator for H1-H3 direct hypothesis evaluation."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.analisis_cfg: AnalisisConfig = build_analisis_config(app_config)

        self.alpha: float = self.analisis_cfg[SIGNIFIKANSI_ALPHA_KEY]

        self.input_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir: Path = self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Path:
        """Execute H1-H3 evaluation pipeline.

        Returns:
            Path: Path to exported evaluasi_hipotesis_langsung.xlsx.
        """
        input_path: Path = self.input_dir / ANTECEDENT_EFFECTS_FILENAME
        df_raw: pd.DataFrame = load_coefficients(input_path)

        df_pred: pd.DataFrame = filter_predictors(df_raw)
        df_mapped: pd.DataFrame = map_hypothesis_labels(df_pred)
        df_evaluated: pd.DataFrame = evaluate_decisions(df_mapped, self.alpha)
        df_summary: pd.DataFrame = compile_summary_table(df_evaluated)

        output_path: Path = self.output_dir / EVALUASI_HIPOTESIS_FILENAME
        export_evaluasi_excel(df_summary, output_path)

        print(
            f"[SUCCESS] Evaluasi Hipotesis Langsung (H1-H3) selesai. "
            f"Alpha={self.alpha}. Output: {output_path}"
        )
        return output_path


def validate_hypotheses(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 13.1."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: HipotesisLangsungOrchestrator = HipotesisLangsungOrchestrator(
        app_config
    )
    orchestrator.execute()
