# src/core_synthesis/orchestrator.py
from pathlib import Path
from typing import Any

from src.utils import setup_logger

from src.core_synthesis.ingestion import run_ingestion
from src.core_synthesis.preprocessing_laten import run_preprocessing_laten
from src.core_synthesis.generator_setup import run_generator_setup
from src.core_synthesis.hierarchical_sim_laten import run_hierarchical_sim_laten
from src.core_synthesis.hierarchical_sim_likert import run_hierarchical_sim_likert
from src.core_synthesis.post_processing_likert import run_post_processing_likert
from src.core_synthesis.synthesis_demo import run_synthesis_demo
from src.core_synthesis.synthesis_text import run_synthesis_text
from src.core_synthesis.assembly_final import run_assembly_final

logger = setup_logger("synth_orchestrator")


class SynthesisOrchestrator:
    """Orchestrator untuk menjalankan seluruh pipeline sintesis data
    Block C secara berurutan dan terkontrol.

    Pipeline stages:
        1. Ingestion              -> load, schema alignment, slicing
        2. Preprocessing Laten    -> reduce 25 Likert ke 5 latent means
        3. Generator Setup        -> extract mean & covariance, regularize PSD
        4. Hierarchical Sim Laten -> generate M multivariate normal datasets
        5. Hierarchical Sim Likert-> expand 5 latent ke 25 continuous indicators
        6. Post Processing Likert -> discretize continuous ke integer [1, 5]
        7. Synthesis Demo         -> weighted random sampling demografi
        8. Synthesis Text         -> AI batch synthesis jawaban kualitatif
        9. Assembly Final         -> concat demo + likert + teks

    Attributes:
        raw_dir: Direktori seed mentah (data/01_raw_seed).
        processed_dir: Direktori hasil preprocessing (data/02_processed).
        synthetic_dir: Direktori output sintetik (data/03_synthetic).
        results: Dictionary hasil eksekusi tiap stage.
    """

    def __init__(
        self,
        raw_dir: Path = Path("data/01_raw_seed"),
        processed_dir: Path = Path("data/02_processed"),
        synthetic_dir: Path = Path("data/03_synthetic"),
    ) -> None:
        """Inisialisasi orchestrator dengan path konfigurasi.

        Args:
            raw_dir: Direktori berisi file seed (.csv / .xlsx).
            processed_dir: Direktori output preprocessing & parameter distribusi.
            synthetic_dir: Direktori output dataset sintetik final.
        """
        self.raw_dir: Path = raw_dir
        self.processed_dir: Path = processed_dir
        self.synthetic_dir: Path = synthetic_dir
        self.results: dict[str, Any] = {}

    def _stage_ingestion(self) -> tuple[Any, Any, Any]:
        """Stage 1: Ingestion & Schema Alignment.

        Returns:
            tuple: (df_demo, df_likert, df_teks).
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: INGESTION & SCHEMA ALIGNMENT")
        logger.info("=" * 60)

        result = run_ingestion(
            seed_dir=self.raw_dir,
            output_dir=self.processed_dir,
        )
        self.results["ingestion"] = {"status": "PASSED"}
        logger.info("Stage 1 completed.\n")
        return result

    def _stage_preprocessing_laten(self) -> Any:
        """Stage 2: Preprocessing Latent Means.

        Returns:
            pd.DataFrame: df_latent_seed (20, 5).
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: PREPROCESSING LATENT MEANS")
        logger.info("=" * 60)

        df_latent = run_preprocessing_laten(
            input_dir=self.processed_dir,
            output_dir=self.processed_dir,
        )
        self.results["preprocessing_laten"] = {"shape": df_latent.shape}
        logger.info("Stage 2 completed.\n")
        return df_latent

    def _stage_generator_setup(self) -> tuple[Any, Any, Any]:
        """Stage 3: Generator Setup (Mean, Covariance, PSD Regularization).

        Returns:
            tuple: (mean_vector, cov_matrix, m_datasets).
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: GENERATOR SETUP")
        logger.info("=" * 60)

        mean_vector, cov_matrix, m_datasets = run_generator_setup(
            input_dir=self.processed_dir,
            output_dir=self.processed_dir,
        )
        self.results["generator_setup"] = {
            "mean_shape": mean_vector.shape,
            "cov_shape": cov_matrix.shape,
            "m_datasets": m_datasets,
        }
        logger.info("Stage 3 completed.\n")
        return mean_vector, cov_matrix, m_datasets

    def _stage_hierarchical_sim_laten(self) -> Any:
        """Stage 4: Hierarchical Simulation Latent (Level 1).

        Returns:
            list[pd.DataFrame]: List of M generated latent datasets.
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: HIERARCHICAL SIMULATION LATENT")
        logger.info("=" * 60)

        generated = run_hierarchical_sim_laten(
            input_dir=self.processed_dir,
            output_dir=self.synthetic_dir,
        )
        self.results["hierarchical_sim_laten"] = {"count": len(generated)}
        logger.info("Stage 4 completed.\n")
        return generated

    def _stage_hierarchical_sim_likert(self) -> None:
        """Stage 5: Hierarchical Simulation Likert (Level 2)."""
        logger.info("=" * 60)
        logger.info("STAGE 5: HIERARCHICAL SIMULATION LIKERT")
        logger.info("=" * 60)

        run_hierarchical_sim_likert(
            processed_dir=self.processed_dir,
            synthetic_dir=self.synthetic_dir,
        )
        self.results["hierarchical_sim_likert"] = {"status": "PASSED"}
        logger.info("Stage 5 completed.\n")

    def _stage_post_processing_likert(self) -> None:
        """Stage 6: Post-Processing Likert Discretization."""
        logger.info("=" * 60)
        logger.info("STAGE 6: POST-PROCESSING LIKERT")
        logger.info("=" * 60)

        run_post_processing_likert(
            synthetic_dir=self.synthetic_dir,
        )
        self.results["post_processing_likert"] = {"status": "PASSED"}
        logger.info("Stage 6 completed.\n")

    def _stage_synthesis_demo(self) -> None:
        """Stage 7: Synthesis Demographic Data."""
        logger.info("=" * 60)
        logger.info("STAGE 7: SYNTHESIS DEMOGRAPHIC")
        logger.info("=" * 60)

        run_synthesis_demo(
            processed_dir=self.processed_dir,
            synthetic_dir=self.synthetic_dir,
        )
        self.results["synthesis_demo"] = {"status": "PASSED"}
        logger.info("Stage 7 completed.\n")

    def _stage_synthesis_text(self) -> None:
        """Stage 8: Synthesis Qualitative Text Responses."""
        logger.info("=" * 60)
        logger.info("STAGE 8: SYNTHESIS TEXT")
        logger.info("=" * 60)

        run_synthesis_text(
            processed_dir=self.processed_dir,
            synthetic_dir=self.synthetic_dir,
        )
        self.results["synthesis_text"] = {"status": "PASSED"}
        logger.info("Stage 8 completed.\n")

    def _stage_assembly_final(self) -> None:
        """Stage 9: Final Assembly (Demo + Likert + Teks)."""
        logger.info("=" * 60)
        logger.info("STAGE 9: ASSEMBLY FINAL")
        logger.info("=" * 60)

        run_assembly_final(
            synthetic_dir=self.synthetic_dir,
        )
        self.results["assembly_final"] = {"status": "PASSED"}
        logger.info("Stage 9 completed.\n")

    def run_full_pipeline(self) -> dict[str, Any]:
        """Eksekusi seluruh pipeline sintesis Block C secara berurutan.

        Returns:
            dict: Summary hasil eksekusi semua stage.
        """
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE ORCHESTRATOR: BLOCK C SYNTHESIS")
        logger.info("=" * 60 + "\n")

        self._stage_ingestion()
        self._stage_preprocessing_laten()
        self._stage_generator_setup()
        self._stage_hierarchical_sim_laten()
        self._stage_hierarchical_sim_likert()
        self._stage_post_processing_likert()
        self._stage_synthesis_demo()
        self._stage_synthesis_text()
        self._stage_assembly_final()

        logger.info("=" * 60)
        logger.info("PIPELINE ORCHESTRATOR: BLOCK C COMPLETED")
        logger.info("=" * 60)

        return self.results

    def run_stages(self, stages: list[str]) -> dict[str, Any]:
        """Eksekusi stage tertentu secara selektif.

        Args:
            stages: List nama stage yang akan dijalankan.
                    Pilihan: ingestion, preprocessing, generator, laten,
                             likert, postproc, demo, text, assembly.

        Returns:
            dict: Summary hasil eksekusi stage yang dipilih.
        """
        stage_map: dict[str, Any] = {
            "ingestion": self._stage_ingestion,
            "preprocessing": self._stage_preprocessing_laten,
            "generator": self._stage_generator_setup,
            "laten": self._stage_hierarchical_sim_laten,
            "likert": self._stage_hierarchical_sim_likert,
            "postproc": self._stage_post_processing_likert,
            "demo": self._stage_synthesis_demo,
            "text": self._stage_synthesis_text,
            "assembly": self._stage_assembly_final,
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


def run_synthesis_pipeline(
    raw_dir: Path = Path("data/01_raw_seed"),
    processed_dir: Path = Path("data/02_processed"),
    synthetic_dir: Path = Path("data/03_synthetic"),
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Entry point untuk menjalankan orchestrator sintesis.

    Args:
        raw_dir: Direktori seed mentah.
        processed_dir: Direktori hasil preprocessing.
        synthetic_dir: Direktori output sintetik.
        stages: List stage spesifik. None = jalankan semua.

    Returns:
        dict: Summary hasil pipeline.
    """
    orchestrator = SynthesisOrchestrator(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        synthetic_dir=synthetic_dir,
    )

    if stages is None:
        return orchestrator.run_full_pipeline()
    return orchestrator.run_stages(stages)


if __name__ == "__main__":
    run_synthesis_pipeline()
