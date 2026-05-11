# src/core_analysis/orchestrator.py
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import setup_logger

from src.core_analysis.master_selection import run_master_selection
from src.core_analysis.profiling_demografi import run_profiling_demografi
from src.core_analysis.deskriptif_likert import run_deskriptif_likert
from src.core_analysis.qc_final_check import run_qc_final_check
from src.core_analysis.uji_validitas import run_uji_validitas
from src.core_analysis.uji_reliabilitas import run_uji_reliabilitas
from src.core_analysis.uji_normalitas import run_uji_normalitas
from src.core_analysis.uji_multikol import run_uji_multikol
from src.core_analysis.uji_hetero import run_uji_hetero

logger = setup_logger("orchestrator")


class PipelineOrchestrator:
    """Orchestrator untuk menjalankan seluruh pipeline analisis statistik
    Chapter 4 secara berurutan dan terkontrol.

    Pipeline stages:
        1. Master Selection       -> isolasi dataset final
        2. Profiling Demografi    -> karakteristik responden + Chi-Square
        3. QC Final Check         -> audit kualitas data
        4. Deskriptif Likert      -> statistik deskriptif + kategorisasi
        5. Uji Validitas          -> Pearson item-total correlation
        6. Uji Reliabilitas       -> Cronbach's Alpha
        7. Uji Normalitas         -> Shapiro-Wilk + QQ-Plot
        8. Uji Multikolinearitas  -> VIF + Tolerance
        9. Uji Heteroskedastisitas-> Breusch-Pagan + Glejser

    Attributes:
        synthetic_dir: Direktori dataset sintetik.
        report_dir: Direktori master dataset terkunci.
        seed_demo_path: Path ke seed demografi asli.
        target_index: Index dataset sintetik yang dipilih (1-based).
        results: Dictionary hasil eksekusi tiap stage.
    """

    def __init__(
        self,
        synthetic_dir: Path = Path("data/03_synthetic"),
        report_dir: Path = Path("data/04_report_master"),
        seed_demo_path: Path = Path("data/02_processed/df_demo.csv"),
        target_index: int | None = None,
    ) -> None:
        """Inisialisasi orchestrator dengan path konfigurasi.

        Args:
            synthetic_dir: Direktori berisi df_final_syn_*.csv.
            report_dir: Direktori output df_report_master.csv.
            seed_demo_path: Path ke seed demografi.
            target_index: Index dataset target. None = baca dari config.
        """
        self.synthetic_dir: Path = synthetic_dir
        self.report_dir: Path = report_dir
        self.seed_demo_path: Path = seed_demo_path
        self.target_index: int | None = target_index
        self.results: dict[str, Any] = {}

        self.master_path: Path = self.report_dir / "df_report_master.csv"
        self.output_base: Path = Path("reports")

    def _stage_master_selection(self) -> pd.DataFrame:
        """Stage 1: Isolasi dan kunci dataset master.

        Returns:
            pd.DataFrame: Locked master dataframe.
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: MASTER SELECTION")
        logger.info("=" * 60)

        df_master = run_master_selection(
            synthetic_dir=self.synthetic_dir,
            report_dir=self.report_dir,
            target_index=self.target_index,
        )
        self.results["master_selection"] = {"shape": df_master.shape}
        logger.info("Stage 1 completed.\n")
        return df_master

    def _stage_profiling_demografi(self) -> pd.DataFrame:
        """Stage 2: Profil demografi + Chi-Square goodness-of-fit.

        Returns:
            pd.DataFrame: Tabel profil demografi.
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: PROFILING DEMOGRAFI")
        logger.info("=" * 60)

        df_profile = run_profiling_demografi(
            master_path=self.master_path,
            seed_path=self.seed_demo_path,
            output_path=self.output_base / "tables" / "tabel_demografi.xlsx",
        )
        self.results["profiling_demografi"] = {"shape": df_profile.shape}
        logger.info("Stage 2 completed.\n")
        return df_profile

    def _stage_qc_final_check(self) -> None:
        """Stage 3: Audit kualitas data final."""
        logger.info("=" * 60)
        logger.info("STAGE 3: QC FINAL CHECK")
        logger.info("=" * 60)

        run_qc_final_check(
            master_path=self.master_path,
            output_path=self.output_base / "data_quality_audit.txt",
        )
        self.results["qc_final_check"] = {"status": "PASSED"}
        logger.info("Stage 3 completed.\n")

    def _stage_deskriptif_likert(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 4: Statistik deskriptif dan kategorisasi Likert.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (tabel deskriptif, tabel distribusi).
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: DESKRIPTIF LIKERT")
        logger.info("=" * 60)

        df_desc, df_dist = run_deskriptif_likert(
            master_path=self.master_path,
            output_desc=self.output_base / "tables" / "tabel_deskriptif_indikator.xlsx",
            output_dist=self.output_base / "tables" / "tabel_kategorisasi_laten.xlsx",
        )
        self.results["deskriptif_likert"] = {
            "descriptive_shape": df_desc.shape,
            "distribution_shape": df_dist.shape,
        }
        logger.info("Stage 4 completed.\n")
        return df_desc, df_dist

    def _stage_uji_validitas(self) -> pd.DataFrame:
        """Stage 5: Uji validitas indikator (Pearson).

        Returns:
            pd.DataFrame: Matriks validitas.
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: UJI VALIDITAS")
        logger.info("=" * 60)

        df_validity = run_uji_validitas(
            master_path=self.master_path,
            output_path=self.output_base / "tables" / "tabel_validitas_pearson.xlsx",
        )
        self.results["uji_validitas"] = {"shape": df_validity.shape}
        logger.info("Stage 5 completed.\n")
        return df_validity

    def _stage_uji_reliabilitas(self) -> pd.DataFrame:
        """Stage 6: Uji reliabilitas konstruk (Cronbach's Alpha).

        Returns:
            pd.DataFrame: Matriks reliabilitas.
        """
        logger.info("=" * 60)
        logger.info("STAGE 6: UJI RELIABILITAS")
        logger.info("=" * 60)

        df_reliability = run_uji_reliabilitas(
            master_path=self.master_path,
            output_path=self.output_base / "tables" / "tabel_reliabilitas_cronbach.xlsx",
        )
        self.results["uji_reliabilitas"] = {"shape": df_reliability.shape}
        logger.info("Stage 6 completed.\n")
        return df_reliability

    def _stage_uji_normalitas(self) -> dict[str, Any]:
        """Stage 7: Uji normalitas residual OLS (Shapiro-Wilk + QQ-Plot).

        Returns:
            dict: Hasil uji normalitas.
        """
        logger.info("=" * 60)
        logger.info("STAGE 7: UJI NORMALITAS")
        logger.info("=" * 60)

        result = run_uji_normalitas(
            master_path=self.master_path,
            output_path=self.output_base / "figures" / "qq_plot_residuals.png",
        )
        self.results["uji_normalitas"] = result
        logger.info("Stage 7 completed.\n")
        return result

    def _stage_uji_multikol(self) -> pd.DataFrame:
        """Stage 8: Uji multikolinearitas (VIF + Tolerance).

        Returns:
            pd.DataFrame: Matriks VIF.
        """
        logger.info("=" * 60)
        logger.info("STAGE 8: UJI MULTIKOLINEARITAS")
        logger.info("=" * 60)

        df_vif = run_uji_multikol(
            master_path=self.master_path,
            output_path=self.output_base / "tables" / "tabel_multikolinearitas.xlsx",
        )
        self.results["uji_multikol"] = {"shape": df_vif.shape}
        logger.info("Stage 8 completed.\n")
        return df_vif

    def _stage_uji_hetero(self) -> dict[str, Any]:
        """Stage 9: Uji heteroskedastisitas (Breusch-Pagan + Glejser).

        Returns:
            dict: Hasil uji heteroskedastisitas.
        """
        logger.info("=" * 60)
        logger.info("STAGE 9: UJI HETEROSKEDASTISITAS")
        logger.info("=" * 60)

        result = run_uji_hetero(
            master_path=self.master_path,
            scatter_path=self.output_base / "figures" / "scatterplot_hetero.png",
            table_path=self.output_base / "tables" / "tabel_heteroskedastisitas.xlsx",
        )
        self.results["uji_hetero"] = result
        logger.info("Stage 9 completed.\n")
        return result

    def run_full_pipeline(self) -> dict[str, Any]:
        """Eksekusi seluruh pipeline secara berurutan.

        Returns:
            dict: Summary hasil eksekusi semua stage.
        """
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE ORCHESTRATOR: CHAPTER 4 ANALYSIS")
        logger.info("=" * 60 + "\n")

        # Stage 1
        self._stage_master_selection()

        # Stage 2
        self._stage_profiling_demografi()

        # Stage 3
        self._stage_qc_final_check()

        # Stage 4
        self._stage_deskriptif_likert()

        # Stage 5
        self._stage_uji_validitas()

        # Stage 6
        self._stage_uji_reliabilitas()

        # Stage 7
        self._stage_uji_normalitas()

        # Stage 8
        self._stage_uji_multikol()

        # Stage 9
        self._stage_uji_hetero()

        logger.info("=" * 60)
        logger.info("PIPELINE ORCHESTRATOR: ALL STAGES COMPLETED")
        logger.info("=" * 60)

        return self.results

    def run_stages(self, stages: list[str]) -> dict[str, Any]:
        """Eksekusi stage tertentu secara selektif.

        Args:
            stages: List nama stage yang akan dijalankan.
                    Pilihan: master, demografi, qc, deskriptif,
                             validitas, reliabilitas, normalitas,
                             multikol, hetero.

        Returns:
            dict: Summary hasil eksekusi stage yang dipilih.
        """
        stage_map: dict[str, Any] = {
            "master": self._stage_master_selection,
            "demografi": self._stage_profiling_demografi,
            "qc": self._stage_qc_final_check,
            "deskriptif": self._stage_deskriptif_likert,
            "validitas": self._stage_uji_validitas,
            "reliabilitas": self._stage_uji_reliabilitas,
            "normalitas": self._stage_uji_normalitas,
            "multikol": self._stage_uji_multikol,
            "hetero": self._stage_uji_hetero,
        }

        logger.info("\n" + "=" * 60)
        logger.info(f"PIPELINE ORCHESTRATOR: SELECTIVE RUN [{', '.join(stages)}]")
        logger.info("=" * 60 + "\n")

        for stage_name in stages:
            if stage_name not in stage_map:
                logger.error(f"Stage '{stage_name}' tidak dikenali. Skip.")
                continue
            stage_map[stage_name]()

        logger.info("=" * 60)
        logger.info("PIPELINE ORCHESTRATOR: SELECTIVE RUN COMPLETED")
        logger.info("=" * 60)

        return self.results


def run_orchestrator(
    synthetic_dir: Path = Path("data/03_synthetic"),
    report_dir: Path = Path("data/04_report_master"),
    seed_demo_path: Path = Path("data/02_processed/df_demo.csv"),
    target_index: int | None = None,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Entry point untuk menjalankan orchestrator.

    Args:
        synthetic_dir: Direktori dataset sintetik.
        report_dir: Direktori master dataset.
        seed_demo_path: Path ke seed demografi.
        target_index: Index dataset target. None = baca dari config.
        stages: List stage spesifik. None = jalankan semua.

    Returns:
        dict: Summary hasil pipeline.
    """
    orchestrator = PipelineOrchestrator(
        synthetic_dir=synthetic_dir,
        report_dir=report_dir,
        seed_demo_path=seed_demo_path,
        target_index=target_index,
    )

    if stages is None:
        return orchestrator.run_full_pipeline()
    return orchestrator.run_stages(stages)


if __name__ == "__main__":
    run_orchestrator()
