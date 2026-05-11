# src/core_regression/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.core_reproducibility.config_builder import build_analisis_config
from src.core_reproducibility.schema import AnalisisConfig
from src.core_regression._core import (
    build_design_matrix,
    evaluate_hypotheses,
    extract_model_fit,
    extract_standardized_beta,
    extract_unstandardized_coefficients,
    fit_ols_hc3,
    load_master_data,
    merge_coefficient_tables,
    standardize_variables,
)
from src.core_regression.constants import (
    ARSITEKTUR_MODEL_KEY,
    DEFAULT_ANTECEDENT_EFFECTS_FILENAME,
    DEFAULT_DELTA_R2_FILENAME,
    DEFAULT_MEDIATOR_OUTCOMES_FILENAME,
    DEPENDEN_KEY,
    INDEPENDEN_KEY,
    KUNCI_RANDOM_SEED_KEY,
    MEDIATOR_KEY,
    OUTPUT_TABLES_KEY,
    PATHS_KEY,
    REPORT_MASTER_KEY,
    ROOT_PATHS_KEY,
    SIGNIFIKANSI_ALPHA_KEY,
)
from src.core_regression.estimate_antecedent_effects import export_antecedent_effects_excel
from src.core_regression.estimate_mediator_outcomes import (
    build_full_matrix,
    compute_delta_r2,
    export_delta_r2_excel,
    export_mediator_outcomes_excel,
    extract_full_coefficients,
    fit_base_model,
    fit_full_model,
)
from src.utils.config_loader import load_config


class AntecedentEffectsOrchestrator:
    """Orchestrator for Sub-Structure 1: X -> M regression."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.analisis_cfg: AnalisisConfig = build_analisis_config(app_config)

        arsitektur: dict[str, list[str]] = self.analisis_cfg[ARSITEKTUR_MODEL_KEY]

        self.alpha: float = self.analisis_cfg[SIGNIFIKANSI_ALPHA_KEY]
        self.seed: int = self.analisis_cfg[KUNCI_RANDOM_SEED_KEY]

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

        np.random.seed(self.seed)

        self.report_master_dir: Path = Path(
            app_config[ROOT_PATHS_KEY][REPORT_MASTER_KEY]
        )
        self.output_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Path:
        """Execute full Sub-Structure 1 pipeline.

        Returns:
            Path: Path to exported Excel file.
        """
        df: pd.DataFrame = load_master_data(self.report_master_dir)
        cols_to_std: list[str] = self.predictors + [self.mediator]
        df_std: pd.DataFrame = standardize_variables(df, cols_to_std)

        X: np.ndarray
        col_names: list[str]
        X, col_names = build_design_matrix(df, self.predictors)
        y: np.ndarray = df[self.mediator].values

        results: sm.regression.linear_model.RegressionResultsWrapper = fit_ols_hc3(X, y)

        model_fit: dict[str, Any] = extract_model_fit(results)

        unstd_df: pd.DataFrame = extract_unstandardized_coefficients(
            results, col_names, self.alpha
        )
        std_df: pd.DataFrame = extract_standardized_beta(
            df_std, self.predictors, self.mediator
        )
        merged_df: pd.DataFrame = merge_coefficient_tables(unstd_df, std_df)
        coeff_df: pd.DataFrame = evaluate_hypotheses(merged_df, self.alpha)

        output_path: Path = self.output_dir / DEFAULT_ANTECEDENT_EFFECTS_FILENAME
        export_antecedent_effects_excel(model_fit, coeff_df, output_path)

        print(
            f"[SUCCESS] Sub-Struktur 1 (X->M) selesai. "
            f"R²={model_fit['r_squared']:.4f}, Adj_R²={model_fit['adj_r_squared']:.4f}, "
            f"F={model_fit['f_statistic']:.4f}, p={model_fit['f_pvalue']:.4f}. "
            f"Output: {output_path}"
        )
        return output_path


class MediatorOutcomesOrchestrator:
    """Orchestrator for Sub-Structure 2: X, M -> Y regression."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.analisis_cfg: AnalisisConfig = build_analisis_config(app_config)

        arsitektur: dict[str, list[str]] = self.analisis_cfg[ARSITEKTUR_MODEL_KEY]

        self.alpha: float = self.analisis_cfg[SIGNIFIKANSI_ALPHA_KEY]
        self.seed: int = self.analisis_cfg[KUNCI_RANDOM_SEED_KEY]

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

        np.random.seed(self.seed)

        self.report_master_dir: Path = Path(
            app_config[ROOT_PATHS_KEY][REPORT_MASTER_KEY]
        )
        self.output_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Tuple[Path, Path]:
        """Execute full Sub-Structure 2 pipeline.

        Returns:
            tuple: (estimate_mediator_outcomes.xlsx path, delta_R2_mediator.xlsx path).
        """
        df: pd.DataFrame = load_master_data(self.report_master_dir)
        cols_to_std: list[str] = self.predictors + [self.mediator] + [self.dependen]
        df_std: pd.DataFrame = standardize_variables(df, cols_to_std)

        res_base: sm.regression.linear_model.RegressionResultsWrapper = fit_base_model(
            df, self.predictors, self.dependen
        )
        model_fit_base: dict[str, Any] = extract_model_fit(res_base)

        res_full: sm.regression.linear_model.RegressionResultsWrapper = fit_full_model(
            df, self.predictors, self.mediator, self.dependen
        )
        model_fit_full: dict[str, Any] = extract_model_fit(res_full)

        delta_r2: dict[str, Any] = compute_delta_r2(res_base, res_full)

        _, col_names = build_full_matrix(df, self.predictors, self.mediator)
        coeff_df: pd.DataFrame = extract_full_coefficients(res_full, col_names, self.alpha)

        std_df: pd.DataFrame = extract_standardized_beta(
            df_std, self.predictors + [self.mediator], self.dependen
        )
        coeff_df = merge_coefficient_tables(coeff_df, std_df)

        output_path: Path = self.output_dir / DEFAULT_MEDIATOR_OUTCOMES_FILENAME
        export_mediator_outcomes_excel(
            model_fit_base, model_fit_full, delta_r2, coeff_df, output_path
        )

        delta_path: Path = self.output_dir / DEFAULT_DELTA_R2_FILENAME
        export_delta_r2_excel(delta_r2, delta_path)

        print(
            f"[SUCCESS] Sub-Struktur 2 (X,M->Y) selesai. "
            f"R²_base={model_fit_base['r_squared']:.4f}, "
            f"R²_full={model_fit_full['r_squared']:.4f}, "
            f"ΔR²={delta_r2['delta_r2']:.4f}, "
            f"F={delta_r2['f_statistic']:.4f}, p={delta_r2['f_pvalue']:.4f}. "
            f"Output: {output_path}, {delta_path}"
        )
        return output_path, delta_path


def estimate_antecedent_effects(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 12.1."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: AntecedentEffectsOrchestrator = AntecedentEffectsOrchestrator(
        app_config
    )
    orchestrator.execute()


def estimate_mediator_outcomes(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 12.2."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: MediatorOutcomesOrchestrator = MediatorOutcomesOrchestrator(
        app_config
    )
    orchestrator.execute()
