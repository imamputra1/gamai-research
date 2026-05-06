# src/core_reproducibility/config_builder.py
from __future__ import annotations

from typing import Any


def build_block_d_config(app_config: dict[str, Any]) -> dict[str, Any]:
    block_d: dict[str, Any] = app_config["reproducibility"]["block_d"]

    config_statistik: dict[str, Any] = {
        "signifikansi_alpha": block_d["signifikansi_alpha"],
        "jumlah_bootstrap": block_d["jumlah_bootstrap"],
        "kunci_random_seed": block_d["kunci_random_seed"],
        "arsitektur_model": {
            "Independen": block_d["arsitektur_model"]["independen"],
            "Mediator": block_d["arsitektur_model"]["mediator"],
            "Dependen": block_d["arsitektur_model"]["dependen"],
        },
    }
    return config_statistik
