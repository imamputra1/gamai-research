# src/core_regression/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core_regression.substruktur1_core import (
   build_design_matrix,
   evaluate_hypotheses,
   export_substruktur1_excel,
   extract_model_fit,
   extract_standardized_beta,
   extract_unstandardized_coefficients,
   fit_ols_hc3,
   load_master_data,
   merge_coefficient_tables,
   standardize_variables,
)
from src.core_regression.substruktur2_core import (
   build_full_matrix,
   compute_delta_r2,
   export_delta_r2_excel,
   export_substruktur2_excel,
   extract_full_coefficients,
   fit_base_model,
   fit_full_model,
)
from src.utils.config_loader import load_config


class SubStruktur1Orchestrator:
   """Orchestrator for Sub-Structure 1: X -> M regression."""

   def __init__(self, app_config: dict[str, Any]) -> None:
       self.app_config = app_config
       block_d: dict[str, Any] = app_config["reproducibility"]["block_d"]

       self.alpha: float = block_d["signifikansi_alpha"]
       self.seed: int = block_d["kunci_random_seed"]
       self.predictors: list[str] = block_d["arsitektur_model"]["independen"]
       self.mediator: str = block_d["arsitektur_model"]["mediator"][0]

       np.random.seed(self.seed)

       self.report_master_dir: Path = Path(app_config["paths"]["report_master"])
       self.output_dir: Path = Path(block_d["paths"]["output_tables"])
       self.output_dir.mkdir(parents=True, exist_ok=True)

   def execute(self) -> Path:
       """Execute full Sub-Structure 1 pipeline.

       Returns:
           Path: Path to exported Excel file.
       """
       df = load_master_data(self.report_master_dir)
       cols_to_std = self.predictors + [self.mediator]
       df_std = standardize_variables(df, cols_to_std)

       X, col_names = build_design_matrix(df, self.predictors)
       y = df[self.mediator].values

       results = fit_ols_hc3(X, y)

       model_fit = extract_model_fit(results)

       unstd_df = extract_unstandardized_coefficients(results, col_names, self.alpha)
       std_df = extract_standardized_beta(df_std, self.predictors, self.mediator)
       merged_df = merge_coefficient_tables(unstd_df, std_df)
       coeff_df = evaluate_hypotheses(merged_df, self.alpha)

       output_path = self.output_dir / "hasil_regresi_1.xlsx"
       export_substruktur1_excel(model_fit, coeff_df, output_path)

       print(
           f"INFO: Sub-Struktur 1 (X->M) selesai. "
           f"R²={model_fit['r_squared']:.4f}, Adj_R²={model_fit['adj_r_squared']:.4f}, "
           f"F={model_fit['f_statistic']:.4f}, p={model_fit['f_pvalue']:.4f}. "
           f"Output: {output_path}"
       )
       return output_path


class SubStruktur2Orchestrator:
   """Orchestrator for Sub-Structure 2: X, M -> Y regression."""

   def __init__(self, app_config: dict[str, Any]) -> None:
       self.app_config = app_config
       block_d: dict[str, Any] = app_config["reproducibility"]["block_d"]

       self.alpha: float = block_d["signifikansi_alpha"]
       self.seed: int = block_d["kunci_random_seed"]
       self.predictors: list[str] = block_d["arsitektur_model"]["independen"]
       self.mediator: str = block_d["arsitektur_model"]["mediator"][0]
       self.dependen: str = block_d["arsitektur_model"]["dependen"][0]

       np.random.seed(self.seed)

       self.report_master_dir: Path = Path(app_config["paths"]["report_master"])
       self.output_dir: Path = Path(block_d["paths"]["output_tables"])
       self.output_dir.mkdir(parents=True, exist_ok=True)

   def execute(self) -> tuple[Path, Path]:
       """Execute full Sub-Structure 2 pipeline.

       Returns:
           tuple: (hasil_regresi_2.xlsx path, delta_R2_mediator.xlsx path).
       """
       # 12.2.1: Load data
       df = load_master_data(self.report_master_dir)
       cols_to_std = self.predictors + [self.mediator] + [self.dependen]
       df_std = standardize_variables(df, cols_to_std)

       # 12.2.2: Base model (X -> Y)
       res_base = fit_base_model(df, self.predictors, self.dependen)
       model_fit_base = extract_model_fit(res_base)

       # 12.2.3: Full model (X + M -> Y)
       res_full = fit_full_model(df, self.predictors, self.mediator, self.dependen)
       model_fit_full = extract_model_fit(res_full)

       # 12.2.4: Delta R²
       delta_r2 = compute_delta_r2(res_base, res_full)

       # 12.2.5: Coefficients
       _, col_names = build_full_matrix(df, self.predictors, self.mediator)
       coeff_df = extract_full_coefficients(res_full, col_names, self.alpha)

       # Standardized beta
       std_df = extract_standardized_beta(
           df_std, self.predictors + [self.mediator], self.dependen
       )
       coeff_df = merge_coefficient_tables(coeff_df, std_df)

       # 12.2.6: Export
       output_path = self.output_dir / "hasil_regresi_2.xlsx"
       export_substruktur2_excel(
           model_fit_base, model_fit_full, delta_r2, coeff_df, output_path
       )

       delta_path = self.output_dir / "delta_R2_mediator.xlsx"
       export_delta_r2_excel(delta_r2, delta_path)

       print(
           f"INFO: Sub-Struktur 2 (X,M->Y) selesai. "
           f"R²_base={model_fit_base['r_squared']:.4f}, "
           f"R²_full={model_fit_full['r_squared']:.4f}, "
           f"ΔR²={delta_r2['delta_r2']:.4f}, "
           f"F={delta_r2['f_statistic']:.4f}, p={delta_r2['f_pvalue']:.4f}. "
           f"Output: {output_path}, {delta_path}"
       )
       return output_path, delta_path


def run_fase_12_1(config_path: str = "config/pipeline_config.yaml") -> None:
   app_config: dict[str, Any] = load_config(config_path)
   orchestrator = SubStruktur1Orchestrator(app_config)
   orchestrator.execute()


def run_fase_12_2(config_path: str = "config/pipeline_config.yaml") -> None:
   app_config: dict[str, Any] = load_config(config_path)
   orchestrator = SubStruktur2Orchestrator(app_config)
   orchestrator.execute()
