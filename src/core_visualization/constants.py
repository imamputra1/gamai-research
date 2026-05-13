# src/core_visualization/constants.py
from __future__ import annotations

# =============================================================================
# CONFIG KEYS
# =============================================================================
REPRODUCIBILITY_KEY: str = "reproducibility"
ANALISIS_KEY: str = "analisis"
ARSITEKTUR_MODEL_KEY: str = "arsitektur_model"
INDEPENDEN_KEY: str = "Independen"
MEDIATOR_KEY: str = "Mediator"
DEPENDEN_KEY: str = "Dependen"
PATHS_KEY: str = "paths"
OUTPUT_TABLES_KEY: str = "output_tables"
OUTPUT_FIGURES_KEY: str = "output_figures"
ROOT_PATHS_KEY: str = "paths"

# =============================================================================
# DEFAULT FILENAMES
# =============================================================================
PATH_DIAGRAM_FILENAME: str = "path_diagram_final.png"
PATH_DIAGRAM_DOT_FILENAME: str = "path_diagram_final.dot"

# =============================================================================
# LAYOUT CONSTANTS (Graphviz rankdir LR)
# =============================================================================
RANK_DIRECTION: str = "LR"
NODE_SHAPE: str = "box"
NODE_STYLE: str = "rounded,filled"
NODE_FILLCOLOR_DEFAULT: str = "#E8F4FD"
NODE_FILLCOLOR_MEDIATOR: str = "#FFF3CD"
NODE_FILLCOLOR_DEPENDEN: str = "#D4EDDA"
NODE_FONTNAME: str = "Helvetica"
NODE_FONTSIZE: str = "12"
EDGE_COLOR_DEFAULT: str = "#333333"
EDGE_COLOR_SIGNIFIKAN: str = "#28A745"
EDGE_COLOR_NON_SIGNIFIKAN: str = "#DC3545"
EDGE_PENWIDTH: str = "2"

# =============================================================================
# LABEL FORMATS
# =============================================================================
LABEL_FORMAT_EDGE: str = "{b:.3f}{sign}"
LABEL_SIGNIFIKAN: str = "*"
LABEL_NON_SIGNIFIKAN: str = " ns"
