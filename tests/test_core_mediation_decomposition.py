# tests/test_core_mediation_decomposition.py
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_mediation.decomposition_core import (
    compute_effect_decomposition,
    determine_mediation_status,
    extract_coefficient,
    load_bootstrap_results,
)
from src.core_mediation.orchestrator import DekomposisiEfekOrchestrator


def test_extract_coefficient_returns_float() -> None:
    """Must return the correct B value for a known variable."""
    df: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["const", "People", "Process"], "B": [0.5, 1.2, 0.8]}
    )
    result: float = extract_coefficient(df, "People")
    assert result == 1.2


def test_extract_coefficient_raises_on_missing() -> None:
    """Must raise KeyError when variable is absent."""
    df: pd.DataFrame = pd.DataFrame({"Variabel": ["X"], "B": [1.0]})
    with pytest.raises(KeyError):
        extract_coefficient(df, "Missing")


def test_determine_mediation_status_no_mediation() -> None:
    """VAF < 20% must return Tidak Ada Mediasi."""
    assert determine_mediation_status(15.0) == "Tidak Ada Mediasi"
    assert determine_mediation_status(0.0) == "Tidak Ada Mediasi"
    assert determine_mediation_status(19.99) == "Tidak Ada Mediasi"


def test_determine_mediation_status_partial() -> None:
    """VAF 20% - 80% must return Mediasi Parsial."""
    assert determine_mediation_status(20.0) == "Mediasi Parsial"
    assert determine_mediation_status(50.0) == "Mediasi Parsial"
    assert determine_mediation_status(80.0) == "Mediasi Parsial"


def test_determine_mediation_status_full() -> None:
    """VAF > 80% must return Mediasi Penuh."""
    assert determine_mediation_status(80.01) == "Mediasi Penuh"
    assert determine_mediation_status(100.0) == "Mediasi Penuh"


def test_compute_effect_decomposition_with_significant_bootstrap() -> None:
    """VAF and status must be calculated when bootstrap is Signifikan."""
    antecedent: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["const", "People", "Process"], "B": [0.0, 2.0, 3.0]}
    )
    full_model: pd.DataFrame = pd.DataFrame(
        {
            "Variabel": ["const", "People", "Process", "Experience_Value"],
            "B": [0.0, 0.5, 0.7, 1.5],
        }
    )
    bootstrap: pd.DataFrame = pd.DataFrame(
        {
            "Prediktor": ["People", "Process"],
            "Point_Estimate": [3.0, 4.5],
            "Status": ["Signifikan", "Signifikan"],
        }
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People", "Process"], "Experience_Value", bootstrap
    )

    assert result.shape == (2, 9)
    assert "Signifikansi_Bootstrap" in result.columns
    assert "VAF_Persen" in result.columns
    assert "Status_Mediasi" in result.columns

    # People: indirect=3.0, total=3.5, VAF = 85.71% -> Full Mediation
    assert result.loc[0, "VAF_Persen"] == 85.71
    assert result.loc[0, "Status_Mediasi"] == "Mediasi Penuh"

    # Process: indirect=4.5, total=5.2, VAF = 86.54% -> Full Mediation
    assert result.loc[1, "VAF_Persen"] == 86.54
    assert result.loc[1, "Status_Mediasi"] == "Mediasi Penuh"


def test_compute_effect_decomposition_partial_mediation() -> None:
    """Must classify partial mediation correctly for VAF in 20-80 range."""
    antecedent: pd.DataFrame = pd.DataFrame({"Variabel": ["People"], "B": [1.0]})
    full_model: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People", "M"], "B": [0.8, 0.5]}
    )
    bootstrap: pd.DataFrame = pd.DataFrame(
        {"Prediktor": ["People"], "Status": ["Signifikan"]}
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People"], "M", bootstrap
    )

    # indirect=0.5, total=1.3, VAF = 38.46% -> Partial Mediation
    assert result.loc[0, "VAF_Persen"] == 38.46
    assert result.loc[0, "Status_Mediasi"] == "Mediasi Parsial"


def test_compute_effect_decomposition_no_mediation() -> None:
    """Must classify no mediation correctly for VAF < 20%."""
    antecedent: pd.DataFrame = pd.DataFrame({"Variabel": ["People"], "B": [0.2]})
    full_model: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People", "M"], "B": [0.9, 0.5]}
    )
    bootstrap: pd.DataFrame = pd.DataFrame(
        {"Prediktor": ["People"], "Status": ["Signifikan"]}
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People"], "M", bootstrap
    )

    # indirect=0.1, total=1.0, VAF = 10.0% -> No Mediation
    assert result.loc[0, "VAF_Persen"] == 10.0
    assert result.loc[0, "Status_Mediasi"] == "Tidak Ada Mediasi"


def test_compute_effect_decomposition_with_non_significant_bootstrap() -> None:
    """VAF and status must be 'Tidak Dihitung' when bootstrap is not significant."""
    antecedent: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People"], "B": [2.0]}
    )
    full_model: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People", "M"], "B": [0.5, 1.5]}
    )
    bootstrap: pd.DataFrame = pd.DataFrame(
        {"Prediktor": ["People"], "Point_Estimate": [3.0], "Status": ["Tidak Signifikan"]}
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People"], "M", bootstrap
    )

    assert result.loc[0, "VAF_Persen"] == "Tidak Dihitung"
    assert result.loc[0, "Status_Mediasi"] == "Tidak Dihitung"


def test_compute_effect_decomposition_without_bootstrap() -> None:
    """VAF and status must be 'Tidak Dihitung' when bootstrap results are absent."""
    antecedent: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People"], "B": [2.0]}
    )
    full_model: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People", "M"], "B": [0.5, 1.5]}
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People"], "M", bootstrap_df=None
    )

    assert result.loc[0, "VAF_Persen"] == "Tidak Dihitung"
    assert result.loc[0, "Status_Mediasi"] == "Tidak Dihitung"


def test_compute_effect_decomposition_zero_total() -> None:
    """Must handle zero total effect gracefully even when significant."""
    antecedent: pd.DataFrame = pd.DataFrame({"Variabel": ["People"], "B": [1.0]})
    full_model: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["People", "M"], "B": [-1.0, 1.0]}
    )
    bootstrap: pd.DataFrame = pd.DataFrame(
        {"Prediktor": ["People"], "Status": ["Signifikan"]}
    )

    result: pd.DataFrame = compute_effect_decomposition(
        antecedent, full_model, ["People"], "M", bootstrap
    )

    assert result.loc[0, "Total_Effect"] == 0.0
    assert result.loc[0, "VAF_Persen"] == "Tidak Dihitung"
    assert result.loc[0, "Status_Mediasi"] == "Tidak Dihitung"


def test_orchestrator_uses_canonical_config() -> None:
    """Orchestrator must resolve predictors and mediator from analisis config."""
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

    orch: DekomposisiEfekOrchestrator = DekomposisiEfekOrchestrator(app_config)

    assert orch.predictors == ["People", "Process"]
    assert orch.mediator == "Experience_Value"


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
        DekomposisiEfekOrchestrator(app_config)
