# tests/test_radar_chart.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_visualization.orchestrator import RadarChartOrchestrator
from src.core_visualization.radar_chart import (
    compute_tcr_scores,
    construct_radar_chart,
    _prepare_polar_data,
)


def test_compute_tcr_scores_returns_percentage() -> None:
    """TCR must be (mean / likert_max) * 100."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "People": [3.0, 4.0, 5.0],
            "Process": [2.0, 3.0, 4.0],
        }
    )
    result: Dict[str, float] = compute_tcr_scores(df, ["People", "Process"], likert_max=5.0)

    assert result["People"] == 80.0  # (4.0 / 5.0) * 100
    assert result["Process"] == 60.0  # (3.0 / 5.0) * 100


def test_compute_tcr_scores_skips_missing_columns() -> None:
    """Must silently skip columns not present in dataframe."""
    df: pd.DataFrame = pd.DataFrame({"People": [3.0, 4.0, 5.0]})
    result: Dict[str, float] = compute_tcr_scores(df, ["People", "Missing"], likert_max=5.0)

    assert "People" in result
    assert "Missing" not in result


def test_prepare_polar_data_closes_loop() -> None:
    """Must append first element to close the polygon."""
    labels: list[str] = ["A", "B", "C"]
    values: list[float] = [50.0, 60.0, 70.0]

    angles, values_closed, labels_closed = _prepare_polar_data(labels, values)

    assert len(angles) == 4
    assert len(values_closed) == 4
    assert len(labels_closed) == 4
    assert values_closed[-1] == values[0]
    assert labels_closed[-1] == labels[0]
    assert np.isclose(values_closed[-1], 50.0)


def test_construct_radar_chart_returns_figure() -> None:
    """Must return a matplotlib Figure with polar axes."""
    tcr: Dict[str, float] = {"People": 75.0, "Process": 60.0, "Physical_Evidence": 80.0}

    fig = construct_radar_chart(tcr)

    assert fig is not None
    ax = fig.axes[0]
    assert ax.name == "polar"


def test_construct_radar_chart_ylim_0_to_100() -> None:
    """Radial limits must be 0-100 for percentage scale."""
    tcr: Dict[str, float] = {"People": 50.0}

    fig = construct_radar_chart(tcr)
    ax = fig.axes[0]

    assert ax.get_ylim() == (0.0, 100.0)


def test_orchestrator_uses_canonical_config() -> None:
    """Orchestrator must resolve predictors from analisis config."""
    app_config: Dict[str, Any] = {
        "paths": {"report_master": "data/04_report_master"},
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
        },
    }

    orch: RadarChartOrchestrator = RadarChartOrchestrator(app_config)

    assert orch.predictors == ["People", "Process", "Physical_Evidence"]


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
        RadarChartOrchestrator(app_config)
