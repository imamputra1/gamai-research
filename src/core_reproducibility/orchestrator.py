# src/core_reproducibility/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core_reproducibility.config_builder import build_block_d_config
from src.core_reproducibility.config_persistor import ConfigPersistor
from src.core_reproducibility.directory_bootstrap import bootstrap_block_d_directories
from src.utils.config_loader import load_config


class ReproducibilityOrchestrator:
    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config: dict[str, Any] = app_config
        self.persistor: ConfigPersistor = ConfigPersistor(Path("config"))

    def execute(self) -> Path:
        config_statistik: dict[str, Any] = build_block_d_config(self.app_config)
        bootstrap_block_d_directories(self.app_config)
        target_path: Path = self.persistor.persist_json(
            config_statistik, "config_blockD.json"
        )
        return target_path


def run_fase_12_0(config_path: str = "config/pipeline_config.yaml") -> None:
    app_config: dict[str, Any] = load_config(config_path)
    orchestrator = ReproducibilityOrchestrator(app_config)
    target_path = orchestrator.execute()

    seed: int = app_config["reproducibility"]["block_d"]["kunci_random_seed"]
    alpha: float = app_config["reproducibility"]["block_d"]["signifikansi_alpha"]
    bootstrap: int = app_config["reproducibility"]["block_d"]["jumlah_bootstrap"]

    print(
        f"INFO: Konfigurasi Blok D dimuat: Seed={seed}, Alpha={alpha}, Bootstrap={bootstrap}. "
        f"Parameter Blok D berhasil disimpan di {target_path}."
    )
