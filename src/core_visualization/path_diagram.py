# src/core_visualization/path_diagram.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import graphviz

from src.core_visualization.constants import (
    EDGE_COLOR_DEFAULT,
    EDGE_COLOR_NON_SIGNIFIKAN,
    EDGE_COLOR_SIGNIFIKAN,
    EDGE_PENWIDTH_NON_SIGNIFIKAN,
    EDGE_PENWIDTH_SIGNIFIKAN,
    EDGE_STYLE_NON_SIGNIFIKAN,
    EDGE_STYLE_SIGNIFIKAN,
    LABEL_FORMAT_EDGE,
    LABEL_FORMAT_R_SQUARED,
    LABEL_NON_SIGNIFIKAN,
    LABEL_SIGNIFIKAN,
    NODE_FILLCOLOR_DEFAULT,
    NODE_FILLCOLOR_DEPENDEN,
    NODE_FILLCOLOR_MEDIATOR,
    NODE_FONTNAME,
    NODE_FONTSIZE,
    NODE_SHAPE,
    NODE_STYLE,
    R_SQUARED_KEY,
    RANK_DIRECTION,
)


class GraphvizNotFoundError(RuntimeError):
    """Raised when the Graphviz system executable (`dot`) is not found."""
    pass


def _ensure_graphviz_executable() -> None:
    """Verify that the Graphviz `dot` executable is available on PATH.

    Raises:
        GraphvizNotFoundError: If `dot` is not found.
    """
    if shutil.which("dot") is None:
        raise GraphvizNotFoundError(
            "Graphviz executable 'dot' tidak ditemukan di PATH sistem. "
            "Install Graphviz system-level terlebih dahulu: "
            "sudo apt-get install -y graphviz  (Debian/Ubuntu)  atau  "
            "brew install graphviz  (macOS)."
        )


def _sanitize_node_id(name: str) -> str:
    """Sanitize variable name for Graphviz node ID compatibility.

    Args:
        name: Raw variable name.

    Returns:
        str: Sanitized identifier.
    """
    return name.replace(" ", "_").replace("-", "_")


def _build_node_attributes(
    fillcolor: str = NODE_FILLCOLOR_DEFAULT,
    label: str | None = None,
) -> Dict[str, str]:
    """Construct standard node attribute dictionary.

    Args:
        fillcolor: Background color hex code.
        label: Optional override label (e.g. with R-squared).

    Returns:
        dict: Graphviz node attributes.
    """
    attrs: Dict[str, str] = {
        "shape": NODE_SHAPE,
        "style": NODE_STYLE,
        "fillcolor": fillcolor,
        "fontname": NODE_FONTNAME,
        "fontsize": NODE_FONTSIZE,
    }
    if label is not None:
        attrs["label"] = label
    return attrs


def _build_edge_attributes(
    b_value: float,
    is_significant: bool,
    label: str | None = None,
) -> Dict[str, str]:
    """Construct edge attribute dictionary with conditional styling.

    Significant paths: solid thick line + asterisk.
    Non-significant paths: dashed thin line + 'ns'.

    Args:
        b_value: Regression coefficient.
        is_significant: Whether the effect is statistically significant.
        label: Optional override label.

    Returns:
        dict: Graphviz edge attributes.
    """
    sign: str = LABEL_SIGNIFIKAN if is_significant else LABEL_NON_SIGNIFIKAN
    display_label: str = label or LABEL_FORMAT_EDGE.format(b=b_value, sign=sign)
    color: str = EDGE_COLOR_SIGNIFIKAN if is_significant else EDGE_COLOR_NON_SIGNIFIKAN
    penwidth: str = EDGE_PENWIDTH_SIGNIFIKAN if is_significant else EDGE_PENWIDTH_NON_SIGNIFIKAN
    style: str = EDGE_STYLE_SIGNIFIKAN if is_significant else EDGE_STYLE_NON_SIGNIFIKAN

    return {
        "label": display_label,
        "color": color,
        "fontcolor": color,
        "penwidth": penwidth,
        "style": style,
    }


def _build_endogenous_label(
    variable_name: str,
    r_squared: float | None,
) -> str:
    """Build node label with variable name and R-squared for endogenous vars.

    Args:
        variable_name: Display name of the variable.
        r_squared: R-squared value (None for exogenous variables).

    Returns:
        str: Formatted label string.
    """
    if r_squared is None:
        return variable_name
    return f"{variable_name}\\n{LABEL_FORMAT_R_SQUARED.format(r2=r_squared)}"


def construct_path_diagram(
    predictors: List[str],
    mediator: str,
    dependen: str,
    antecedent_coeffs: Dict[str, float],
    full_model_coeffs: Dict[str, float],
    antecedent_significance: Dict[str, bool],
    full_model_significance: Dict[str, bool],
    mediator_to_dependen_b: float,
    mediator_to_dependen_significant: bool,
    mediator_r_squared: float | None = None,
    dependen_r_squared: float | None = None,
) -> graphviz.Digraph:
    """Construct a left-to-right path diagram using Graphviz.

    Layout: Independen (left) -> Mediator (center) -> Dependen (right).

    Endogenous nodes (Mediator, Dependen) display R-squared below the name.
    Significant edges are solid and thick; non-significant are dashed and thin.

    Args:
        predictors: List of independent variable names.
        mediator: Mediator variable name.
        dependen: Dependent variable name.
        antecedent_coeffs: Coefficients from X -> M regression.
        full_model_coeffs: Coefficients from X,M -> Y full model.
        antecedent_significance: Significance flags for X -> M paths.
        full_model_significance: Significance flags for X -> Y direct paths.
        mediator_to_dependen_b: Coefficient for M -> Y path.
        mediator_to_dependen_significant: Significance flag for M -> Y.
        mediator_r_squared: Optional R-squared for mediator model.
        dependen_r_squared: Optional R-squared for full model.

    Returns:
        graphviz.Digraph: Configured directed graph.
    """
    dot: graphviz.Digraph = graphviz.Digraph(
        name="PathDiagram",
        format="png",
        graph_attr={
            "rankdir": RANK_DIRECTION,
            "bgcolor": "white",
            "splines": "true",
            "nodesep": "0.6",
            "ranksep": "1.2",
        },
    )

    # --- Exogenous Nodes (X) ---
    for predictor in predictors:
        node_id: str = _sanitize_node_id(predictor)
        dot.node(
            node_id,
            predictor,
            **_build_node_attributes(NODE_FILLCOLOR_DEFAULT),
        )

    # --- Endogenous Node: Mediator (M) ---
    med_id: str = _sanitize_node_id(mediator)
    med_label: str = _build_endogenous_label(mediator, mediator_r_squared)
    dot.node(
        med_id,
        med_label,
        **_build_node_attributes(NODE_FILLCOLOR_MEDIATOR),
    )

    # --- Endogenous Node: Dependen (Y) ---
    dep_id: str = _sanitize_node_id(dependen)
    dep_label: str = _build_endogenous_label(dependen, dependen_r_squared)
    dot.node(
        dep_id,
        dep_label,
        **_build_node_attributes(NODE_FILLCOLOR_DEPENDEN),
    )

    # --- Edges: X -> M ---
    for predictor in predictors:
        src: str = _sanitize_node_id(predictor)
        dst: str = med_id
        b_val: float = antecedent_coeffs.get(predictor, 0.0)
        is_sig: bool = antecedent_significance.get(predictor, False)
        dot.edge(src, dst, **_build_edge_attributes(b_val, is_sig))

    # --- Edges: M -> Y ---
    dot.edge(
        med_id,
        dep_id,
        **_build_edge_attributes(
            mediator_to_dependen_b,
            mediator_to_dependen_significant,
        ),
    )

    # --- Edges: X -> Y (direct) ---
    for predictor in predictors:
        src: str = _sanitize_node_id(predictor)
        dst: str = dep_id
        b_val: float = full_model_coeffs.get(predictor, 0.0)
        is_sig: bool = full_model_significance.get(predictor, False)
        dot.edge(
            src,
            dst,
            **_build_edge_attributes(b_val, is_sig),
        )

    return dot


def render_diagram(
    dot: graphviz.Digraph,
    output_path: Path,
    cleanup: bool = True,
) -> Path:
    """Render Graphviz diagram to file.

    Args:
        dot: Configured Digraph.
        output_path: Target file path (including extension).
        cleanup: Remove intermediate DOT file after rendering.

    Returns:
        Path to rendered output file.

    Raises:
        GraphvizNotFoundError: If Graphviz system executable is missing.
    """
    _ensure_graphviz_executable()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered: str = dot.render(
        filename=str(output_path.with_suffix("")),
        format=output_path.suffix.lstrip("."),
        cleanup=cleanup,
    )
    return Path(rendered)


def export_dot_source(dot: graphviz.Digraph, output_path: Path) -> None:
    """Export raw DOT source for manual editing or version control.

    Args:
        dot: Configured Digraph.
        output_path: Target .dot file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dot.source, encoding="utf-8")
