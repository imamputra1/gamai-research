# src/core_mediation/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from src.core_reproducibility.config_builder import build_analisis_config
from src.core_reproducibility.schema import AnalisisConfig
from src.core_regression._core import load_master_data
from src.core_regression.constants import (
    DEFAULT_ANTECEDENT_EFFECTS_FILENAME,
    DEFAULT_MEDIATOR_OUTCOMES_FILENAME,
)
from src.core_mediation.bootstrap_core import (
    compute_ci_statistics,
    export_bootstrap_array,
    export_mediation_excel,
    run_bootstrap,
)
from src.core_mediation.decomposition_core import (
    compute_effect_decomposition,
    export_decomposition_excel,
    load_bootstrap_results,
    load_coefficient_table,
)
from src.core_mediation.constants import (
    ARSITEKTUR_MODEL_KEY,
    BOOTSTRAP_ARRAY_FILENAME,
    DECOMPOSITION_FILENAME,
    DEPENDEN_KEY,
    INDEPENDEN_KEY,
    JUMLAH_BOOTSTRAP_KEY,
    KUNCI_RANDOM_SEED_KEY,
    MEDIASI_BOOTSTRAP_FILENAME,
    MEDIATOR_KEY,
    OUTPUT_TABLES_KEY,
    PATHS_KEY,
    REPORT_MASTER_KEY,
    ROOT_PATHS_KEY,
    SHEET_COEFFICIENTS_ANTECEDENT,
    SHEET_COEFFICIENTS_FULL,
    SIGNIFIKANSI_ALPHA_KEY,
)
from src.utils.config_loader import load_config

class MediasiBootstrapOrchestrator:
    """Orchestrator for mediation bootstrap analysis."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.analisis_cfg: AnalisisConfig = build_analisis_config(app_config)

        arsitektur: dict[str, list[str]] = self.analisis_cfg[ARSITEKTUR_MODEL_KEY]

        self.alpha: float = self.analisis_cfg[SIGNIFIKANSI_ALPHA_KEY]
        self.seed: int = self.analisis_cfg[KUNCI_RANDOM_SEED_KEY]
        self.n_iterations: int = self.analisis_cfg[JUMLAH_BOOTSTRAP_KEY]

        predictors: list[str] = arsitektur[INDEPENDEN_KEY]
        if not predictors:
            raise ValueError(
                f"'{INDEPENDEN_KEY}' list in arsitektur_model cannot be empty."
            )
        self.predictors: list[str] = predictors

        mediator_list: list[str] = arsitektur[MEDIATOR_KEY]
        if not mediator_list:
            raise ValueError(
                f"'{MEDIATOR_KEY}' list in arsitektur_model cannot be empty."
            )
        self.mediator: str = mediator_list[0]

        dependen_list: list[str] = arsitektur[DEPENDEN_KEY]
        if not dependen_list:
            raise ValueError(
                f"'{DEPENDEN_KEY}' list in arsitektur_model cannot be empty."
            )
        self.dependen: str = dependen_list[0]

        self.report_master_dir: Path = Path(
            app_config[ROOT_PATHS_KEY][REPORT_MASTER_KEY]
        )
        self.output_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Tuple[Path, Path]:
        """Execute full H4 mediation bootstrap pipeline.

        Returns:
            tuple: (mediasi_bootstrap_H4.xlsx path, bootstrap_array.npy path).
        """
        df: pd.DataFrame = load_master_data(self.report_master_dir)

        bootstrap_array: np.ndarray = run_bootstrap(
            df,
            self.predictors,
            self.mediator,
            self.dependen,
            self.n_iterations,
            self.seed,
        )

        df_ci: pd.DataFrame = compute_ci_statistics(
            bootstrap_array, self.predictors, self.alpha
        )

        excel_path: Path = self.output_dir / MEDIASI_BOOTSTRAP_FILENAME
        export_mediation_excel(df_ci, excel_path)

        npy_path: Path = self.output_dir / BOOTSTRAP_ARRAY_FILENAME
        export_bootstrap_array(bootstrap_array, npy_path)

        print(
            f"[SUCCESS] Uji Mediasi Bootstrap (H4) selesai. "
            f"N={self.n_iterations}, Seed={self.seed}, Alpha={self.alpha}. "
            f"Output: {excel_path}, {npy_path}"
        )
        return excel_path, npy_path

class DekomposisiEfekOrchestrator:
    """Orchestrator for Fase 14.1: Effect Decomposition & VAF."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.analisis_cfg: AnalisisConfig = build_analisis_config(app_config)

        arsitektur: dict[str, list[str]] = self.analisis_cfg[ARSITEKTUR_MODEL_KEY]

        predictors: list[str] = arsitektur[INDEPENDEN_KEY]
        if not predictors:
            raise ValueError(
                f"'{INDEPENDEN_KEY}' list in arsitektur_model cannot be empty."
            )
        self.predictors: list[str] = predictors

        mediator_list: list[str] = arsitektur[MEDIATOR_KEY]
        if not mediator_list:
            raise ValueError(
                f"'{MEDIATOR_KEY}' list in arsitektur_model cannot be empty."
            )
        self.mediator: str = mediator_list[0]

        self.input_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir: Path = self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Path:
        """Execute effect decomposition pipeline with bootstrap gating.

        Reads regression outputs from Fase 12.1 and 12.2, loads bootstrap
        results from Fase 13.2, computes direct/indirect/total effects, and
        derives VAF percentage ONLY for significant indirect effects.

        Returns:
            Path to exported dekomposisi_efek_VAF.xlsx.
        """
        antecedent_path: Path = self.input_dir / DEFAULT_ANTECEDENT_EFFECTS_FILENAME
        full_model_path: Path = self.input_dir / DEFAULT_MEDIATOR_OUTCOMES_FILENAME

        antecedent_df: pd.DataFrame = load_coefficient_table(
            antecedent_path, SHEET_COEFFICIENTS_ANTECEDENT
        )
        full_model_df: pd.DataFrame = load_coefficient_table(
            full_model_path, SHEET_COEFFICIENTS_FULL
        )

        bootstrap_df: pd.DataFrame | None = None
        bootstrap_path: Path = self.input_dir / MEDIASI_BOOTSTRAP_FILENAME
        if bootstrap_path.exists():
            bootstrap_df = load_bootstrap_results(bootstrap_path)

        decomposition_df: pd.DataFrame = compute_effect_decomposition(
            antecedent_df,
            full_model_df,
            self.predictors,
            self.mediator,
            bootstrap_df,
        )

        output_path: Path = self.output_dir / DECOMPOSITION_FILENAME
        export_decomposition_excel(decomposition_df, output_path)

        print(
            f"[SUCCESS] Dekomposisi Efek & VAF selesai. "
            f"Prediktor={self.predictors}, Mediator={self.mediator}. "
            f"Output: {output_path}"
        )
        return output_path


def compute_mediation_bootstrap(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 13.2."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: MediasiBootstrapOrchestrator = MediasiBootstrapOrchestrator(
        app_config
    )
    orchestrator.execute()

def compute_decomposition(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 14.1."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: DekomposisiEfekOrchestrator = DekomposisiEfekOrchestrator(
        app_config
    )
    orchestrator.execute()

