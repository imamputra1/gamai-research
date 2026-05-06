# main_pipeline.py
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.excel_exporter import export_dataframe_to_excel
from src.core_synthesis.monte_carlo import (
    synthesize_likert_monte_carlo,
    compute_correlation_matrix,
    validate_correlation_preservation,
)
from src.ai_integration.kimi_client import KimiClient
from src.ai_integration.openclaw_client import OpenClawClient


class ThesisPipeline:
    """End-to-end orchestrator composing synthesis and AI clients."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        kimi_client: KimiClient | None = None,
        openclaw_client: OpenClawClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.logger = setup_logger("thesis_pipeline")
        self.kimi = kimi_client
        self.openclaw = openclaw_client

    def run(self, seed_path: str | Path) -> None:
        """Execute pipeline: load seed, synthesize M datasets, validate, export.

        Args:
            seed_path: Path to raw seed file (.xlsx or .csv).
        """
        self.logger.info("Pipeline execution started")

        seed_path = Path(seed_path)
        if str(seed_path).endswith(".xlsx"):
            seed_df = pd.read_excel(seed_path)
        else:
            seed_df = pd.read_csv(seed_path)

        self.logger.info(f"Seed loaded: {seed_df.shape}")

        pipeline_cfg = self.config["pipeline"]
        likert_cfg = self.config["likert"]
        tol_cfg = self.config["tolerance"]

        numeric_seed = seed_df.select_dtypes(include="number")
        seed_corr = compute_correlation_matrix(numeric_seed)

        n_target = pipeline_cfg["n_target"]
        m_datasets = pipeline_cfg["m_datasets"]
        random_seed = pipeline_cfg["random_seed"]

        synthetic_dir = Path(self.config["paths"]["synthetic"])
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        for m in range(1, m_datasets + 1):
            self.logger.info(f"Synthesizing dataset {m}/{m_datasets}")

            syn_df = synthesize_likert_monte_carlo(
                seed_df=numeric_seed,
                n_target=n_target,
                random_seed=random_seed + m,
                likert_min=likert_cfg["min"],
                likert_max=likert_cfg["max"],
            )

            syn_corr = compute_correlation_matrix(syn_df)
            validation = validate_correlation_preservation(
                seed_corr=seed_corr,
                syn_corr=syn_corr,
                tolerance=tol_cfg["corr"],
            )

            self.logger.info(
                f"Dataset {m} | max_dev: {validation['max_deviation']:.4f} | "
                f"valid: {validation['is_valid']}"
            )

            export_dataframe_to_excel(
                syn_df,
                synthetic_dir / f"df_final_syn_{m}.xlsx",
                sheet_name="Synthetic",
            )

        self.logger.info("Pipeline execution completed")


def main() -> None:
    pipeline = ThesisPipeline()
    seed_file = Path("data/01_raw_seed") / "seed_responses.xlsx"
    if seed_file.exists():
        pipeline.run(seed_file)
    else:
        pipeline.logger.warning(f"Seed file not found: {seed_file}")


if __name__ == "__main__":
    main()
