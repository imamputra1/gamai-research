# src/core_reproducibility/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core_reproducibility.config_builder import build_analisis_config, ConfigBuildError
from src.core_reproducibility.config_persistor import ConfigPersistor
from src.core_reproducibility.constants import (
    DEFAULT_ANALISIS_CONFIG_FILENAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PERSIST_BASE_PATH,
)
from src.core_reproducibility.directory_bootstrap import bootstrap_analisis_directories
from src.core_reproducibility.schema import AnalisisConfig
from src.utils.config_loader import load_config


class ReproducibilityOrchestrator:
    """Orchestrates the reproducibility bootstrap phase for analisis.

    Responsibilities:
        1. Build the canonical analisis configuration.
        2. Bootstrap the required output directories.
        3. Persist the resolved configuration to JSON.

    This class follows composition-over-inheritance: it *has-a*
    :class:`ConfigPersistor` injected rather than inheriting behaviour.
    """

    def __init__(
        self,
        app_config: Dict[str, Any],
        persistor: Optional[ConfigPersistor] = None,
        output_filename: str = DEFAULT_ANALISIS_CONFIG_FILENAME,
    ) -> None:
        self.app_config: Dict[str, Any] = app_config
        self.persistor: ConfigPersistor = persistor or ConfigPersistor(
            Path(DEFAULT_PERSIST_BASE_PATH)
        )
        self.output_filename: str = output_filename
        self._analisis_config: Optional[AnalisisConfig] = None

    def execute(self) -> Tuple[Path, AnalisisConfig]:
        """Run the reproducibility bootstrap pipeline.

        Returns:
            A tuple of ``(persisted_file_path, analisis_config)``.
        """
        self._analisis_config = build_analisis_config(self.app_config)
        bootstrap_analisis_directories(self._analisis_config)
        target_path: Path = self.persistor.persist_json(
            self._analisis_config, self.output_filename
        )
        return target_path, self._analisis_config


def initialize_workspace(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> None:
    """Convenience entry-point for Fase 12.0.

    Args:
        config_path: Filesystem path to the pipeline YAML configuration.
    """
    app_config: Dict[str, Any] = load_config(config_path)
    orchestrator: ReproducibilityOrchestrator = ReproducibilityOrchestrator(
        app_config
    )
    target_path: Path
    analisis_config: AnalisisConfig
    target_path, analisis_config = orchestrator.execute()

    seed: int = analisis_config["kunci_random_seed"]
    alpha: float = analisis_config["signifikansi_alpha"]
    bootstrap: int = analisis_config["jumlah_bootstrap"]

    print(
        f"[SUCCESS] Konfigurasi analisis dimuat: Seed={seed}, Alpha={alpha}, "
        f"Bootstrap={bootstrap}. Parameter analisis berhasil disimpan di {target_path}."
    )
