# src/core_visualization/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.core_reproducibility.config_builder import build_analisis_config
from src.core_reproducibility.schema import AnalisisConfig
from src.core_regression.constants import (
    DEFAULT_ANTECEDENT_EFFECTS_FILENAME,
    DEFAULT_MEDIATOR_OUTCOMES_FILENAME,
)
from src.core_visualization.constants import (
    ARSITEKTUR_MODEL_KEY,
    DEPENDEN_KEY,
    INDEPENDEN_KEY,
    MEDIATOR_KEY,
    OUTPUT_FIGURES_KEY,
    OUTPUT_TABLES_KEY,
    PATH_DIAGRAM_DOT_FILENAME,
    PATH_DIAGRAM_FILENAME,
    PATHS_KEY,
    R_SQUARED_KEY,
    ROOT_PATHS_KEY,
    RADAR_CHART_FILENAME,
)
from src.core_visualization.path_diagram import (
    construct_path_diagram,
    export_dot_source,
    render_diagram,
)
from src.core_visualization.radar_chart import (
    compute_tcr_scores,
    construct_radar_chart,
    export_radar_chart,
)
from src.utils.config_loader import load_config
from src.core_regression._core import load_master_data

class PathDiagramOrchestrator:
    """Orchestrator for Fase 14.2: Path Diagram Visualization."""

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

        dependen_list: list[str] = arsitektur[DEPENDEN_KEY]
        if not dependen_list:
            raise ValueError(
                f"'{DEPENDEN_KEY}' list in arsitektur_model cannot be empty."
            )
        self.dependen: str = dependen_list[0]

        self.input_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_TABLES_KEY]
        )
        self.output_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_FIGURES_KEY]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_coeffs_and_significance(
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """Extract coefficient values and significance flags from a
        coefficient table.

        Args:
            df: Coefficient dataframe with Variabel, B, p_value columns.

        Returns:
            tuple: (coeff_dict, significance_dict).
        """
        coeff_dict: Dict[str, float] = {}
        sig_dict: Dict[str, bool] = {}

        for _, row in df.iterrows():
            var: str = str(row["Variabel"])
            if var == "const":
                continue
            coeff_dict[var] = float(row["B"])
            sig_dict[var] = bool(row["p_value"] < 0.05 and row["B"] > 0)

        return coeff_dict, sig_dict

    @staticmethod
    def _extract_r_squared(
        df_summary: pd.DataFrame,
        r2_key: str = R_SQUARED_KEY,
    ) -> float | None:
        """Extract R-squared from a Model Summary dataframe.

        Args:
            df_summary: Model Summary dataframe.
            r2_key: Column key for R-squared.

        Returns:
            float: R-squared value, or None if unavailable.
        """
        if r2_key in df_summary.columns and not df_summary.empty:
            return float(df_summary[r2_key].iloc[0])
        return None

    def execute(self) -> Path:
        """Execute path diagram construction and rendering with metric injection.

        Injects R-squared into endogenous nodes (Mediator, Dependen) and
        applies conditional edge styling (solid/thick for significant,
        dashed/thin for non-significant).

        Returns:
            Path to rendered PNG file.
        """
        antecedent_path: Path = self.input_dir / DEFAULT_ANTECEDENT_EFFECTS_FILENAME
        full_model_path: Path = self.input_dir / DEFAULT_MEDIATOR_OUTCOMES_FILENAME

        # Load coefficient tables
        antecedent_df: pd.DataFrame = pd.read_excel(
            antecedent_path, sheet_name="Coefficients"
        )
        full_model_df: pd.DataFrame = pd.read_excel(
            full_model_path, sheet_name="Coefficients Full"
        )

        # Load model summaries for R-squared extraction
        antecedent_summary: pd.DataFrame = pd.read_excel(
            antecedent_path, sheet_name="Model Summary"
        )
        full_model_summary: pd.DataFrame = pd.read_excel(
            full_model_path, sheet_name="Model Summary Full"
        )

        antecedent_coeffs, antecedent_sig = self._extract_coeffs_and_significance(
            antecedent_df
        )
        full_model_coeffs, full_model_sig = self._extract_coeffs_and_significance(
            full_model_df
        )

        # Extract R-squared for endogenous variables
        mediator_r2: float | None = self._extract_r_squared(antecedent_summary)
        dependen_r2: float | None = self._extract_r_squared(full_model_summary)

        # Extract M -> Y coefficient and significance
        mediator_row = full_model_df[full_model_df["Variabel"] == self.mediator]
        med_to_dep_b: float = float(mediator_row["B"].iloc[0]) if not mediator_row.empty else 0.0
        med_to_dep_sig: bool = (
            float(mediator_row["p_value"].iloc[0]) < 0.05 and med_to_dep_b > 0
            if not mediator_row.empty
            else False
        )

        dot = construct_path_diagram(
            predictors=self.predictors,
            mediator=self.mediator,
            dependen=self.dependen,
            antecedent_coeffs=antecedent_coeffs,
            full_model_coeffs=full_model_coeffs,
            antecedent_significance=antecedent_sig,
            full_model_significance=full_model_sig,
            mediator_to_dependen_b=med_to_dep_b,
            mediator_to_dependen_significant=med_to_dep_sig,
            mediator_r_squared=mediator_r2,
            dependen_r_squared=dependen_r2,
        )

        png_path: Path = self.output_dir / PATH_DIAGRAM_FILENAME
        render_diagram(dot, png_path)

        dot_path: Path = self.output_dir / PATH_DIAGRAM_DOT_FILENAME
        export_dot_source(dot, dot_path)

        print(
            f"[SUCCESS] Path Diagram Final selesai. "
            f"R²_M={mediator_r2}, R²_Y={dependen_r2}. "
            f"Output: {png_path}, {dot_path}"
        )
        return png_path

class RadarChartOrchestrator:
    """Orchestrator for Visualisasi Ekstraksi Kuantitatif: Radar Chart TCR."""

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

        self.report_master_dir: Path = Path(
            app_config[ROOT_PATHS_KEY]["report_master"]
        )
        self.output_dir: Path = Path(
            self.analisis_cfg[PATHS_KEY][OUTPUT_FIGURES_KEY]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> Path:
        """Execute radar chart construction for TCR visualization.

        Loads master data, computes TCR for each independent variable,
        and renders a polar radar chart.

        Returns:
            Path to exported PNG file.
        """
        df: pd.DataFrame = load_master_data(self.report_master_dir)

        tcr_scores: Dict[str, float] = compute_tcr_scores(df, self.predictors)

        fig = construct_radar_chart(
            tcr_scores=tcr_scores,
            title="Tingkat Capaian Responden (TCR)",
            color="#3498DB",
        )

        output_path: Path = self.output_dir / RADAR_CHART_FILENAME
        export_radar_chart(fig, output_path)

        print(
            f"[SUCCESS] Radar Chart TCR selesai. "
            f"Prediktor={self.predictors}, TCR={tcr_scores}. "
            f"Output: {output_path}"
        )
        return output_path

def run_path_diagram(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Fase 14.2."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: PathDiagramOrchestrator = PathDiagramOrchestrator(app_config)
    orchestrator.execute()


def run_radar_chart(config_path: str = "config/pipeline_config.yaml") -> None:
    """Convenience entry-point for Radar Chart TCR."""
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator: RadarChartOrchestrator = RadarChartOrchestrator(app_config)
    orchestrator.execute()
