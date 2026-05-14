# tests/test_core_regression.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_regression._core import (
    build_design_matrix,
    evaluate_hypotheses,
    extract_model_fit,
    extract_standardized_beta,
    extract_unstandardized_coefficients,
    fit_ols_hc3,
    merge_coefficient_tables,
    standardize_variables,
)
from src.core_regression.estimate_mediator_outcomes import (
    build_base_matrix,
    build_full_matrix,
    compute_delta_r2,
)
from src.core_regression.orchestrator import (
    AntecedentEffectsOrchestrator,
    MediatorOutcomesOrchestrator,
)


def test_build_design_matrix_shape() -> None:
    """Design matrix must have K+1 columns including intercept."""
    df: pd.DataFrame = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    X: np.ndarray
    col_names: list[str]
    X, col_names = build_design_matrix(df, ["A", "B"])

    assert X.shape == (3, 3)
    assert col_names == ["const", "A", "B"]
    assert np.allclose(X[:, 0], 1.0)


def test_standardize_variables_zero_mean_unit_std() -> None:
    """Standardized variables must have ~0 mean and ~1 std."""
    df: pd.DataFrame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_std: pd.DataFrame = standardize_variables(df, ["A"])

    assert abs(df_std["A"].mean()) < 1e-10
    assert abs(df_std["A"].std(ddof=0) - 1.0) < 1e-10


def test_fit_ols_hc3_returns_results_wrapper() -> None:
    """OLS fit must return a RegressionResultsWrapper with HC3 cov_type."""
    X: np.ndarray = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y: np.ndarray = np.array([2.0, 4.0, 6.0])

    results: Any = fit_ols_hc3(X, y)

    assert hasattr(results, "rsquared")
    assert hasattr(results, "params")
    assert results.cov_type == "HC3"


def test_extract_model_fit_keys() -> None:
    """Extracted fit metrics must contain expected keys."""
    mock_results: MagicMock = MagicMock()
    mock_results.rsquared = 0.85
    mock_results.rsquared_adj = 0.80
    mock_results.fvalue = 25.0
    mock_results.f_pvalue = 0.001
    mock_results.nobs = 100
    mock_results.df_model = 3.0
    mock_results.df_resid = 96.0

    fit: Dict[str, Any] = extract_model_fit(mock_results)

    assert fit["r_squared"] == 0.85
    assert fit["adj_r_squared"] == 0.80
    assert fit["n_observations"] == 100


def test_extract_unstandardized_coefficients_shape() -> None:
    """Coefficient dataframe must have correct shape and columns."""
    mock_results: MagicMock = MagicMock()
    mock_results.params = np.array([1.0, 2.0])
    mock_results.bse = np.array([0.1, 0.2])
    mock_results.tvalues = np.array([10.0, 10.0])
    mock_results.pvalues = np.array([0.0, 0.0])
    mock_results.conf_int.return_value = np.array([[0.8, 1.2], [1.6, 2.4]])

    df: pd.DataFrame = extract_unstandardized_coefficients(
        mock_results, ["const", "X"], alpha=0.05
    )

    assert df.shape == (2, 7)
    assert list(df.columns) == [
        "Variabel",
        "B",
        "Robust_SE",
        "t_stat",
        "p_value",
        "CI_lower_95",
        "CI_upper_95",
    ]


def test_evaluate_hypotheses_rejects_negative_beta() -> None:
    """Hypothesis must be rejected when B <= 0 even if p < alpha."""
    coeff_df: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["const", "X"], "B": [0.5, -1.0], "p_value": [0.001, 0.001]}
    )
    result: pd.DataFrame = evaluate_hypotheses(coeff_df, alpha=0.05)

    assert result.loc[result["Variabel"] == "const", "Hipotesis"].iloc[0] == "N/A"
    assert result.loc[result["Variabel"] == "X", "Hipotesis"].iloc[0] == "Ditolak"


def test_evaluate_hypotheses_accepts_positive_beta() -> None:
    """Hypothesis must be accepted when B > 0 and p < alpha."""
    coeff_df: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["X"], "B": [1.0], "p_value": [0.001]}
    )
    result: pd.DataFrame = evaluate_hypotheses(coeff_df, alpha=0.05)

    assert result.loc[0, "Hipotesis"] == "Diterima"


def test_merge_coefficient_tables() -> None:
    """Merged table must contain columns from both inputs."""
    unstd: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["const", "X"], "B": [1.0, 2.0]}
    )
    std: pd.DataFrame = pd.DataFrame(
        {"Variabel": ["const", "X"], "Beta_Standardized": [0.0, 0.5]}
    )
    merged: pd.DataFrame = merge_coefficient_tables(unstd, std)

    assert "Beta_Standardized" in merged.columns
    assert merged.shape == (2, 3)


def test_build_full_matrix_includes_mediator() -> None:
    """Full matrix must include mediator column."""
    df: pd.DataFrame = pd.DataFrame(
        {"X1": [1.0, 2.0], "X2": [3.0, 4.0], "M": [5.0, 6.0]}
    )
    X: np.ndarray
    names: list[str]
    X, names = build_full_matrix(df, ["X1", "X2"], "M")

    assert names == ["const", "X1", "X2", "M"]
    assert X.shape == (2, 4)


def test_compute_delta_r2_computes_difference() -> None:
    """Delta R² must equal r2_full - r2_base."""
    mock_base: MagicMock = MagicMock()
    mock_base.rsquared = 0.30

    mock_full: MagicMock = MagicMock()
    mock_full.rsquared = 0.50

    with patch(
        "src.core_regression.estimate_mediator_outcomes.anova_lm"
    ) as mock_anova:
        mock_anova.return_value = pd.DataFrame(
            {"F": [np.nan, 10.0], "Pr(>F)": [np.nan, 0.01]}
        )
        result: Dict[str, Any] = compute_delta_r2(mock_base, mock_full)

    assert result["r2_base"] == 0.30
    assert result["r2_full"] == 0.50
    assert result["delta_r2"] == 0.20


def test_antecedent_orchestrator_uses_canonical_config() -> None:
    """Orchestrator must read predictors from canonical analisis config."""
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

    orch: AntecedentEffectsOrchestrator = AntecedentEffectsOrchestrator(app_config)

    assert orch.predictors == ["People", "Process"]
    assert orch.mediator == "Experience_Value"
    assert orch.alpha == 0.05


def test_mediator_orchestrator_uses_canonical_config() -> None:
    """Orchestrator must read dependen from canonical analisis config."""
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

    orch: MediatorOutcomesOrchestrator = MediatorOutcomesOrchestrator(app_config)

    assert orch.dependen == "Minat_Kunjungan_Ulang"
    assert orch.mediator == "Experience_Value"
    assert orch.predictors == ["People", "Process"]


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
        AntecedentEffectsOrchestrator(app_config)
