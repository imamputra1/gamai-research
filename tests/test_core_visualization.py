# tests/test_core_visualization.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core_reproducibility.config_builder import ConfigBuildError
from src.core_visualization.constants import (
    EDGE_COLOR_SIGNIFIKAN,
    EDGE_COLOR_NON_SIGNIFIKAN,
    EDGE_STYLE_SIGNIFIKAN,
    EDGE_STYLE_NON_SIGNIFIKAN,
    EDGE_PENWIDTH_SIGNIFIKAN,
    EDGE_PENWIDTH_NON_SIGNIFIKAN,
    NODE_FILLCOLOR_DEFAULT,
    NODE_FILLCOLOR_MEDIATOR,
    NODE_FILLCOLOR_DEPENDEN,
)
from src.core_visualization.orchestrator import PathDiagramOrchestrator
from src.core_visualization.path_diagram import (
    GraphvizNotFoundError,
    _build_edge_attributes,
    _build_endogenous_label,
    _build_node_attributes,
    _ensure_graphviz_executable,
    _sanitize_node_id,
    construct_path_diagram,
    render_diagram,
)


def test_sanitize_node_id_replaces_spaces_and_dashes() -> None:
    """Must replace spaces and dashes with underscores."""
    assert _sanitize_node_id("Physical Evidence") == "Physical_Evidence"
    assert _sanitize_node_id("X-1") == "X_1"


def test_build_node_attributes_returns_expected_keys() -> None:
    """Must contain all standard Graphviz node keys."""
    attrs: Dict[str, str] = _build_node_attributes()
    assert attrs["shape"] == "box"
    assert attrs["style"] == "rounded,filled"
    assert "fillcolor" in attrs


def test_build_node_attributes_with_label_override() -> None:
    """Must inject label when provided."""
    attrs: Dict[str, str] = _build_node_attributes(label="M\\nR² = 0.850")
    assert attrs["label"] == "M\\nR² = 0.850"


def test_build_endogenous_label_with_r_squared() -> None:
    """Must format label with variable name and R-squared."""
    label: str = _build_endogenous_label("Experience_Value", 0.752)
    assert "Experience_Value" in label
    assert "R² = 0.752" in label


def test_build_endogenous_label_without_r_squared() -> None:
    """Must return plain variable name when R-squared is None."""
    label: str = _build_endogenous_label("People", None)
    assert label == "People"


def test_build_edge_attributes_significant() -> None:
    """Significant edges: solid, thick, green, asterisk."""
    attrs: Dict[str, str] = _build_edge_attributes(0.5, is_significant=True)
    assert attrs["color"] == EDGE_COLOR_SIGNIFIKAN
    assert attrs["style"] == EDGE_STYLE_SIGNIFIKAN
    assert attrs["penwidth"] == EDGE_PENWIDTH_SIGNIFIKAN
    assert "*" in attrs["label"]


def test_build_edge_attributes_non_significant() -> None:
    """Non-significant edges: dashed, thin, red, 'ns'."""
    attrs: Dict[str, str] = _build_edge_attributes(0.5, is_significant=False)
    assert attrs["color"] == EDGE_COLOR_NON_SIGNIFIKAN
    assert attrs["style"] == EDGE_STYLE_NON_SIGNIFIKAN
    assert attrs["penwidth"] == EDGE_PENWIDTH_NON_SIGNIFIKAN
    assert "ns" in attrs["label"]


def test_construct_path_diagram_contains_r_squared_in_nodes() -> None:
    """Endogenous nodes must contain R-squared in their labels."""
    dot = construct_path_diagram(
        predictors=["People"],
        mediator="M",
        dependen="Y",
        antecedent_coeffs={"People": 0.5},
        full_model_coeffs={"People": 0.3},
        antecedent_significance={"People": True},
        full_model_significance={"People": False},
        mediator_to_dependen_b=0.8,
        mediator_to_dependen_significant=True,
        mediator_r_squared=0.65,
        dependen_r_squared=0.78,
    )

    source: str = dot.source
    assert "R² = 0.650" in source
    assert "R² = 0.780" in source


def test_construct_path_diagram_edge_styling() -> None:
    """Edges must reflect significance via style and penwidth."""
    dot = construct_path_diagram(
        predictors=["X1"],
        mediator="M",
        dependen="Y",
        antecedent_coeffs={"X1": 0.5},
        full_model_coeffs={"X1": 0.3},
        antecedent_significance={"X1": True},
        full_model_significance={"X1": False},
        mediator_to_dependen_b=0.8,
        mediator_to_dependen_significant=True,
    )

    source: str = dot.source
    # Significant edges: solid, thick (X1->M and M->Y)
    assert f"style={EDGE_STYLE_SIGNIFIKAN}" in source
    assert f"penwidth={EDGE_PENWIDTH_SIGNIFIKAN}" in source
    # Non-significant edge: dashed, thin (X1->Y)
    assert f"style={EDGE_STYLE_NON_SIGNIFIKAN}" in source
    assert f"penwidth={EDGE_PENWIDTH_NON_SIGNIFIKAN}" in source


def test_construct_path_diagram_has_correct_edge_count() -> None:
    """Must have edges: X->M (1) + M->Y (1) + X->Y (1) = 3 edges."""
    dot = construct_path_diagram(
        predictors=["X1"],
        mediator="M",
        dependen="Y",
        antecedent_coeffs={"X1": 0.5},
        full_model_coeffs={"X1": 0.3},
        antecedent_significance={"X1": True},
        full_model_significance={"X1": False},
        mediator_to_dependen_b=0.8,
        mediator_to_dependen_significant=True,
    )

    edge_count: int = dot.source.count("->")
    assert edge_count == 3


def test_construct_path_diagram_rankdir_lr() -> None:
    """Graph orientation must be Left-to-Right."""
    dot = construct_path_diagram(
        predictors=["X1"],
        mediator="M",
        dependen="Y",
        antecedent_coeffs={"X1": 0.5},
        full_model_coeffs={"X1": 0.3},
        antecedent_significance={"X1": True},
        full_model_significance={"X1": False},
        mediator_to_dependen_b=0.8,
        mediator_to_dependen_significant=True,
    )

    assert 'rankdir=LR' in dot.source


def test_ensure_graphviz_executable_raises_when_missing() -> None:
    """Must raise GraphvizNotFoundError when `dot` is absent."""
    with patch("src.core_visualization.path_diagram.shutil.which", return_value=None):
        with pytest.raises(GraphvizNotFoundError):
            _ensure_graphviz_executable()


def test_ensure_graphviz_executable_passes_when_present() -> None:
    """Must not raise when `dot` is present."""
    with patch("src.core_visualization.path_diagram.shutil.which", return_value="/usr/bin/dot"):
        _ensure_graphviz_executable()  # no exception


def test_orchestrator_uses_canonical_config() -> None:
    """Orchestrator must resolve architecture from analisis config."""
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

    orch: PathDiagramOrchestrator = PathDiagramOrchestrator(app_config)

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
        PathDiagramOrchestrator(app_config)
