# tests/test_config_builder.py
from __future__ import annotations

from typing import Any, Dict

import pytest

from src.core_reproducibility.config_builder import (
    build_analisis_config,
    ConfigBuildError,
)
from src.core_reproducibility.schema import AnalisisConfig


def test_build_analisis_config_with_nested_paths() -> None:
    """SSOT nested paths must be extracted verbatim."""
    app_config: Dict[str, Any] = {
        "reproducibility": {
            "analisis": {
                "signifikansi_alpha": 0.05,
                "jumlah_bootstrap": 5000,
                "kunci_random_seed": 42,
                "arsitektur_model": {
                    "Independen": ["People", "Process", "Physical_Evidence"],
                    "Mediator": ["Experience_Value"],
                    "Dependen": ["Minat_Kunjungan_Ulang"],
                },
                "paths": {
                    "output_tables": "reports/analisis/tables",
                    "output_figures": "reports/analisis/figures",
                },
            }
        }
    }

    result: AnalisisConfig = build_analisis_config(app_config)

    assert result["signifikansi_alpha"] == 0.05
    assert result["jumlah_bootstrap"] == 5000
    assert result["kunci_random_seed"] == 42
    assert result["paths"]["output_tables"] == "reports/analisis/tables"
    assert result["paths"]["output_figures"] == "reports/analisis/figures"
    assert result["arsitektur_model"]["Dependen"] == ["Minat_Kunjungan_Ulang"]


def test_build_analisis_config_fallback_to_root_paths() -> None:
    """When nested paths are absent, root ``paths`` must serve as fallback."""
    app_config: Dict[str, Any] = {
        "paths": {
            "reports_tables": "reports/tables",
            "reports_figures": "reports/figures",
        },
        "reproducibility": {
            "analisis": {
                "signifikansi_alpha": 0.05,
                "jumlah_bootstrap": 5000,
                "kunci_random_seed": 42,
                "arsitektur_model": {
                    "Independen": ["People"],
                    "Mediator": ["Experience_Value"],
                    "Dependen": ["Minat_Kunjungan_Ulang"],
                },
            }
        },
    }

    result: AnalisisConfig = build_analisis_config(app_config)

    assert result["paths"]["output_tables"] == "reports/tables"
    assert result["paths"]["output_figures"] == "reports/figures"


def test_build_analisis_config_missing_reproducibility_raises() -> None:
    """A completely missing ``reproducibility`` section raises ConfigBuildError."""
    app_config: Dict[str, Any] = {}

    with pytest.raises(ConfigBuildError):
        build_analisis_config(app_config)


def test_build_analisis_config_missing_analisis_raises() -> None:
    """A missing ``analisis`` subsection raises ConfigBuildError."""
    app_config: Dict[str, Any] = {"reproducibility": {}}

    with pytest.raises(ConfigBuildError):
        build_analisis_config(app_config)


def test_build_analisis_config_missing_paths_and_no_fallback_raises() -> None:
    """Missing nested paths *and* missing root paths must raise ConfigBuildError."""
    app_config: Dict[str, Any] = {
        "reproducibility": {
            "analisis": {
                "signifikansi_alpha": 0.05,
                "jumlah_bootstrap": 5000,
                "kunci_random_seed": 42,
                "arsitektur_model": {},
            }
        }
    }

    with pytest.raises(ConfigBuildError):
        build_analisis_config(app_config)
