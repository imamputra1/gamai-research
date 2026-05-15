# tests/test_core_nlp_serializer.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.core_nlp.serializer import (
    load_processed,
    load_slang_dict,
    load_stopwords,
    save_processed,
    save_slang_dict,
    save_stopwords,
)


class TestSerializer:
    def test_save_and_load_slang_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "slang.json")
            data = {"yg": "yang", "bgt": "banget"}
            save_slang_dict(path, data)
            loaded = load_slang_dict(path)
            assert loaded == data

    def test_save_and_load_stopwords(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "stopwords.json")
            data = {"yang", "dan", "di"}
            save_stopwords(path, data)
            loaded = load_stopwords(path)
            assert loaded == data

    def test_save_and_load_processed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "processed.csv")
            df = pd.DataFrame({"text": ["bagus", "kurang"]})
            save_processed(df, path)
            loaded = load_processed(path)
            pd.testing.assert_frame_equal(loaded, df)

    def test_save_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "slang.json")
            save_slang_dict(path, {"a": "b"})
            assert Path(path).exists()
