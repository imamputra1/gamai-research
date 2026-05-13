# src/core_visualization/path_diagram.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import graphviz

from src.core_visualization.constants import (
    EDGE_COLOR_DEFAULT,
    EDGE_COLOR_NON_SIGNIFIKAN,
    EDGE_COLOR_SIGNIFIKAN,
    EDGE_PENWIDTH,
    LABEL_FORMAT_EDGE,
    LABEL_NON_SIGNIFIKAN,
    LABEL_SIGNIFIKAN,
    NODE_FILLCOLOR_DEFAULT,
    NODE_FILLCOLOR_DEPENDEN,
    NODE_FILLCOLOR_MEDIATOR,
    NODE_FONTNAME,
    NODE_FONTSIZE,
    NODE_SHAPE,
    NODE_STYLE,
    RANK_DIRECTION,
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
) -> Dict[str, str]:
    """Construct standard node attribute dictionary.

    Args:
        fillcolor: Background color hex code.

    Returns:
        dict: Graphviz node attributes.
    """
    return {
        "shape": NODE_SHAPE,
        "style": NODE_STYLE,
        "fillcolor": fillcolor,
        "fontname": NODE_FONTNAME,
        "fontsize": NODE_FONTSIZE,
    }


def _build_edge_attributes(
    b_value: float,
    is_significant: bool,
    label: str | None = None,
) -> Dict[str, str]:
    """Construct edge attribute dictionary with conditional styling.

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

    return {
        "label": display_label,
        "color": color,
        "fontcolor": color,
        "penwidth": EDGE_PENWIDTH,
    }


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
) -> graphviz.Digraph:
    """Construct a left-to-right path diagram using Graphviz.

    Layout: Independen (left) -> Mediator (center) -> Dependen (right).

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

    # --- Nodes ---
    for predictor in predictors:
        node_id: str = _sanitize_node_id(predictor)
        dot.node(
            node_id,
            predictor,
            **_build_node_attributes(NODE_FILLCOLOR_DEFAULT),
        )

    med_id: str = _sanitize_node_id(mediator)
    dot.node(
        med_id,
        mediator,
        **_build_node_attributes(NODE_FILLCOLOR_MEDIATOR),
    )

    dep_id: str = _sanitize_node_id(dependen)
    dot.node(
        dep_id,
        dependen,
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
    """
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
