# tests/test_core_mediation.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_mediation.bootstrap_core import (
    bootstrap_iteration,
    compute_ci_statistics,
    run_bootstrap,
)
from src.core_mediation.orchestrator import MediasiBootstrapOrchestrator


def test_bootstrap_iteration_returns_correct_shape() -> None:
    """Single iteration must return indirect effects for all predictors."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "People": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Process": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Physical_Evidence": [3.0, 4.0, 5.0, 6.0, 7.0],
            "Experience_Value": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Minat_Kunjungan_Ulang": [2.5, 3.5, 4.5, 5.5, 6.5],
        }
    )
    result: np.ndarray = bootstrap_iteration(
        df,
        predictor_cols=["People", "Process", "Physical_Evidence"],
        mediator_col="Experience_Value",
        dependen_col="Minat_Kunjungan_Ulang",
        seed=42,
    )

    assert result.shape == (3,)
    assert result.dtype == np.float64


def test_bootstrap_iteration_reproducible_with_same_seed() -> None:
    """Same seed must produce identical indirect effects."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "People": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Process": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Physical_Evidence": [3.0, 4.0, 5.0, 6.0, 7.0],
            "Experience_Value": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Minat_Kunjungan_Ulang": [2.5, 3.5, 4.5, 5.5, 6.5],
        }
    )
    r1: np.ndarray = bootstrap_iteration(
        df,
        ["People", "Process"],
        "Experience_Value",
        "Minat_Kunjungan_Ulang",
        seed=123,
    )
    r2: np.ndarray = bootstrap_iteration(
        df,
        ["People", "Process"],
        "Experience_Value",
        "Minat_Kunjungan_Ulang",
        seed=123,
    )

    assert np.allclose(r1, r2)


def test_run_bootstrap_shape_and_reproducibility() -> None:
    """Bootstrap array must have shape (n_iter, n_predictors) and be
    reproducible."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "People": np.random.randn(50),
            "Process": np.random.randn(50),
            "Experience_Value": np.random.randn(50),
            "Minat_Kunjungan_Ulang": np.random.randn(50),
        }
    )
    arr1: np.ndarray = run_bootstrap(
        df,
        ["People", "Process"],
        "Experience_Value",
        "Minat_Kunjungan_Ulang",
        n_iterations=10,
        seed=99,
        n_jobs=1,
    )
    arr2: np.ndarray = run_bootstrap(
        df,
        ["People", "Process"],
        "Experience_Value",
        "Minat_Kunjungan_Ulang",
        n_iterations=10,
        seed=99,
        n_jobs=1,
    )

    assert arr1.shape == (10, 2)
    assert np.allclose(arr1, arr2)


def test_compute_ci_statistics_detects_significance() -> None:
    """CI not crossing zero must be flagged Signifikan."""
    bootstrap_array: np.ndarray = np.array(
        [
            [0.5, 0.3],
            [0.6, 0.4],
            [0.55, 0.35],
            [0.52, 0.32],
            [0.58, 0.38],
        ]
    )
    df: pd.DataFrame = compute_ci_statistics(
        bootstrap_array, ["People", "Process"], alpha=0.05
    )

    assert df.loc[0, "Status"] == "Signifikan"
    assert df.loc[0, "Keputusan"] == "Mediasi Terbukti"
    assert "LLCI_95" in df.columns
    assert "ULCI_95" in df.columns


def test_compute_ci_statistics_detects_non_significance() -> None:
    """CI crossing zero must be flagged Tidak Signifikan."""
    bootstrap_array: np.ndarray = np.array(
        [
            [-0.5, 0.3],
            [0.6, -0.4],
            [0.0, 0.0],
            [-0.2, 0.2],
            [0.1, -0.1],
        ]
    )
    df: pd.DataFrame = compute_ci_statistics(
        bootstrap_array, ["People", "Process"], alpha=0.05
    )

    assert df.loc[0, "Status"] == "Tidak Signifikan"
    assert df.loc[0, "Keputusan"] == "Mediasi Tidak Terbukti"


def test_orchestrator_uses_canonical_analisis_config() -> None:
    """Orchestrator must read parameters from canonical analisis config."""
    app_config: Dict[str, Any] = {
        "paths": {"report_master": "data/04_report_master"},
        "reproducibility": {
            "analisis": {
                "signifikansi_alpha": 0.05,
                "jumlah_bootstrap": 5000,
                "kunci_random_seed": 42,
                "arsitektur_model": {
                    "Independen": ["People", "Process"],
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

    orch: MediasiBootstrapOrchestrator = MediasiBootstrapOrchestrator(app_config)

    assert orch.alpha == 0.05
    assert orch.n_iterations == 5000
    assert orch.seed == 42
    assert orch.predictors == ["People", "Process"]
    assert orch.mediator == "Experience_Value"
    assert orch.dependen == "Minat_Kunjungan_Ulang"


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
        MediasiBootstrapOrchestrator(app_config)
