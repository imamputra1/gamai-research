# tests/test_core_nlp_mapping_output.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.core_nlp.constants import (
    COL_TEXT_CLEANED,
    COL_TEXT_FINAL,
    COL_TEXT_FILTERED,
    COL_TEXT_NORMALIZED,
)
from src.core_nlp.mapping_output import (
    build_text_mapping,
    export_mapping_csv,
    get_unique_tokens,
    load_text_mapping,
    save_text_mapping,
)


class TestBuildTextMapping:
    def test_maps_original_to_all_stages(self) -> None:
        df = pd.DataFrame({
            "jawaban": ["Bagus. Sekali!"],
            COL_TEXT_CLEANED: ["bagus sekali"],
            COL_TEXT_NORMALIZED: ["bagus sekali"],
            COL_TEXT_FILTERED: ["bagus sekali"],
            COL_TEXT_FINAL: ["bagus sekali"],
        })
        mapping = build_text_mapping(df, "jawaban")
        assert "Bagus. Sekali!" in mapping
        assert mapping["Bagus. Sekali!"]["cleaned"] == "bagus sekali"
        assert mapping["Bagus. Sekali!"]["final"] == "bagus sekali"

    def test_multiple_rows(self) -> None:
        df = pd.DataFrame({
            "jawaban": ["A", "B"],
            COL_TEXT_CLEANED: ["a", "b"],
            COL_TEXT_NORMALIZED: ["a", "b"],
            COL_TEXT_FILTERED: ["a", "b"],
            COL_TEXT_FINAL: ["a", "b"],
        })
        mapping = build_text_mapping(df, "jawaban")
        assert len(mapping) == 2
        assert "A" in mapping
        assert "B" in mapping


class TestSaveAndLoadTextMapping:
    def test_roundtrip_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mapping.json")
            data = {"original": {"cleaned": "a", "normalized": "a", "filtered": "a", "final": "a"}}
            save_text_mapping(data, path)
            loaded = load_text_mapping(path)
            assert loaded == data

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "mapping.json")
            save_text_mapping({"x": {"cleaned": "a", "final": "a"}}, path)
            assert Path(path).exists()


class TestExportMappingCsv:
    def test_outputs_correct_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mapping.csv")
            df = pd.DataFrame({
                "jawaban": ["Bagus"],
                COL_TEXT_CLEANED: ["bagus"],
                COL_TEXT_NORMALIZED: ["bagus"],
                COL_TEXT_FILTERED: ["bagus"],
                COL_TEXT_FINAL: ["bagus"],
            })
            export_mapping_csv(df, "jawaban", path)
            loaded = pd.read_csv(path)
            assert list(loaded.columns) == ["original", "cleaned", "normalized", "filtered", "final"]

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "mapping.csv")
            df = pd.DataFrame({
                "jawaban": ["Bagus"],
                COL_TEXT_CLEANED: ["bagus"],
                COL_TEXT_NORMALIZED: ["bagus"],
                COL_TEXT_FILTERED: ["bagus"],
                COL_TEXT_FINAL: ["bagus"],
            })
            export_mapping_csv(df, "jawaban", path)
            assert Path(path).exists()


class TestGetUniqueTokens:
    def test_extracts_unique_tokens(self) -> None:
        df = pd.DataFrame({
            COL_TEXT_FINAL: ["bagus sekali bagus", "kurang banget"],
        })
        tokens = get_unique_tokens(df, stage="final")
        assert sorted(tokens) == ["bagus", "banget", "kurang", "sekali"]

    def test_empty_series_returns_empty(self) -> None:
        df = pd.DataFrame({COL_TEXT_FINAL: [""]})
        tokens = get_unique_tokens(df, stage="final")
        assert tokens == []
