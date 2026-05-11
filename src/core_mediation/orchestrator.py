# src/core_mediation/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core_mediation.bootstrap_core import (
    compute_ci_statistics,
    export_bootstrap_array,
    export_mediation_excel,
    run_bootstrap_parallel,
)
from src.core_regression._core import load_master_data
from src.utils.config_loader import load_config


class MediasiBootstrapOrchestrator:
    """Orchestrator for H4 mediation bootstrap analysis."""

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config = app_config
        block_d: dict[str, Any] = app_config["reproducibility"]["block_d"]

        self.seed: int = block_d["kunci_random_seed"]
        self.n_iterations: int = block_d["jumlah_bootstrap"]
        self.predictors: list[str] = block_d["arsitektur_model"]["Independen"]
        self.mediator: str = block_d["arsitektur_model"]["Mediator"][0]
        self.dependen: str = block_d["arsitektur_model"]["Dependen"][0]

        self.report_master_dir: Path = Path(app_config["paths"]["report_master"])
        self.output_dir: Path = Path(block_d["paths"]["output_tables"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> tuple[Path, Path]:
        """Execute full H4 mediation bootstrap pipeline.

        Returns:
            tuple: (uji_mediasi_bootstrap.xlsx path, bootstrap_array.npy path).
        """
        # 13.2.1: Load data
        df = load_master_data(self.report_master_dir)

        # 13.2.2-13.2.3: Bootstrap parallel execution
        bootstrap_array = run_bootstrap_parallel(
            df,
            self.predictors,
            self.mediator,
            self.dependen,
            self.n_iterations,
            self.seed,
        )

        # 13.2.4: CI statistics
        df_ci = compute_ci_statistics(bootstrap_array, self.predictors)

        # 13.2.5: H4 evaluation (embedded in compute_ci_statistics)

        # 13.2.6: Export
        excel_path = self.output_dir / "uji_mediasi_bootstrap.xlsx"
        export_mediation_excel(df_ci, str(excel_path))

        npy_path = self.output_dir / "bootstrap_array.npy"
        export_bootstrap_array(bootstrap_array, str(npy_path))

        print(
            f"INFO: Uji Mediasi Bootstrap (H4) selesai. "
            f"N={self.n_iterations}, Seed={self.seed}. "
            f"Output: {excel_path}, {npy_path}"
        )
        return excel_path, npy_path


def run_fase_13_2(config_path: str = "config/pipeline_config.yaml") -> None:
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator = MediasiBootstrapOrchestrator(app_config)
    orchestrator.execute()
