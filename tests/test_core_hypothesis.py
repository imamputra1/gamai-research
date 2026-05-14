# tests/test_core_hypothesis.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_hypothesis.evaluator import (
    compile_summary_table,
    evaluate_decisions,
    filter_predictors,
    map_hypothesis_labels,
)
from src.core_hypothesis.orchestrator import HipotesisLangsungOrchestrator


def test_filter_predictors_excludes_const() -> None:
    """Intercept row must be removed from coefficient table."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "Variabel": ["const", "People", "Process"],
            "B": [0.5, 1.2, 0.8],
            "p_value": [0.1, 0.001, 0.002],
        }
    )
    result: pd.DataFrame = filter_predictors(df)

    assert result.shape == (2, 3)
    assert "const" not in result["Variabel"].values


def test_map_hypothesis_labels_maps_known_variables() -> None:
    """Known predictors must receive correct H-codes and descriptions."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "Variabel": ["People", "Process", "Physical_Evidence"],
            "B": [1.0, 2.0, 3.0],
            "p_value": [0.01, 0.02, 0.03],
        }
    )
    result: pd.DataFrame = map_hypothesis_labels(df)

    assert result.loc[0, "Kode"] == "H1"
    assert result.loc[1, "Kode"] == "H2"
    assert result.loc[2, "Kode"] == "H3"
    assert "People → Experiential Value" in result["Hubungan Variabel"].values


def test_evaluate_decisions_uses_strict_less_than() -> None:
    """Decision must use p < alpha (not <=) and B > 0."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "Variabel": ["People", "Process", "Physical_Evidence", "const"],
            "B": [1.0, 1.0, -1.0, 0.5],
            "p_value": [0.05, 0.049, 0.001, 0.001],
        }
    )
    result: pd.DataFrame = evaluate_decisions(df, alpha=0.05)

    assert result.loc[0, "Keputusan"] == "DITOLAK"  # p == alpha
    assert result.loc[1, "Keputusan"] == "DITERIMA"  # p < alpha, B > 0
    assert result.loc[2, "Keputusan"] == "DITOLAK"  # B < 0
    assert result.loc[3, "Keputusan"] == "N/A"  # const


def test_compile_summary_table_rounds_and_selects() -> None:
    """Summary must contain exactly the expected columns."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "Variabel": ["People"],
            "Kode": ["H1"],
            "Hubungan Variabel": ["People → Experiential Value"],
            "B": [1.23456],
            "t_stat": [2.34567],
            "p_value": [0.01234],
            "Keputusan": ["DITERIMA"],
        }
    )
    result: pd.DataFrame = compile_summary_table(df)

    expected_cols: list[str] = [
        "Kode",
        "Hubungan Variabel",
        "Koefisien (B)",
        "t-Statistic",
        "p-Value",
        "Keputusan",
    ]
    assert list(result.columns) == expected_cols
    assert result.loc[0, "Koefisien (B)"] == 1.2346


def test_orchestrator_uses_canonical_analisis_config() -> None:
    """Orchestrator must resolve alpha and paths from analisis config."""
    app_config: Dict[str, Any] = {
        "paths": {"report_master": "data/04_report_master"},
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
                "paths": {
                    "output_tables": "reports/analisis/tables",
                    "output_figures": "reports/analisis/figures",
                },
            }
        },
    }

    orch: HipotesisLangsungOrchestrator = HipotesisLangsungOrchestrator(app_config)

    assert orch.alpha == 0.05
    assert str(orch.input_dir) == "reports/analisis/tables"


def test_orchestrator_rejects_deprecated_block_d_config() -> None:
    """Orchestrator must raise ConfigBuildError when deprecated block_d is used."""
    app_config: Dict[str, Any] = {
        "paths": {"report_master": "data/04_report_master"},
        "reproducibility": {
            "block_d": {
                "signifikansi_alpha": 0.05,
                "jumlah_bootstrap": 5000,
                "kunci_random_seed": 42,
                "arsitektur_model": {
                    "Independen": ["People"],
                    "Mediator": ["Experience_Value"],
                    "Dependen": ["Minat_Kunjungan_Ulang"],
                },
                "paths": {"output_tables": "reports/tables"},
            }
        },
    }

    with pytest.raises(ConfigBuildError):
        HipotesisLangsungOrchestrator(app_config)
