# tests/test_data_validation.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core_regression._core import load_master_data
from src.utils.config_loader import load_config


def test_schema_alignment() -> None:
    """Assert YAML variables are a valid subset of aggregated DataFrame columns."""
    config = load_config("config/pipeline_config.yaml")
    block_d = config["reproducibility"]["analisis"]

    required_cols: list[str] = (
        block_d["arsitektur_model"]["Independen"]
        + block_d["arsitektur_model"]["Mediator"]
        + block_d["arsitektur_model"]["Dependen"]
    )

    df = load_master_data(Path(config["paths"]["report_master"]))
    missing = [c for c in required_cols if c not in df.columns]

    assert not missing, f"Kolom hilang di df_report_master: {missing}"


def test_numeric_types() -> None:
    """Assert all target columns are numeric (float or int)."""
    config = load_config("config/pipeline_config.yaml")
    block_d = config["reproducibility"]["analisis"]

    target_cols: list[str] = (
        block_d["arsitektur_model"]["Independen"]
        + block_d["arsitektur_model"]["Mediator"]
        + block_d["arsitektur_model"]["Dependen"]
    )

    df = load_master_data(Path(config["paths"]["report_master"]))
    non_numeric = [
        c for c in target_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]

    assert not non_numeric, f"Kolom non-numerik: {non_numeric}"
