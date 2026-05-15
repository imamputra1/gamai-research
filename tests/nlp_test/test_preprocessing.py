# tests/nlp_test/test_preprocessing.py
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
from src.core_nlp.preprocessing import (
    build_text_mapping,
    clean_text,
    export_mapping_csv,
    get_unique_tokens,
    load_processed,
    load_slang_dict,
    load_stopwords,
    load_text_mapping,
    normalize_slang,
    remove_stopwords,
    save_processed,
    save_slang_dict,
    save_stopwords,
    save_text_mapping,
    stem_text,
)


class TestCleaningCaseFolding:
    def test_lowercase_output(self) -> None:
        series = pd.Series(["BAIK", "Bagus", "SANGAT MEMUASKAN"])
        result = clean_text(series)
        assert result.iloc[0] == "baik"
        assert result.iloc[1] == "bagus"
        assert result.iloc[2] == "sangat memuaskan"

    def test_punctuation_replaced_with_space_not_removed(self) -> None:
        series = pd.Series(["bagus.pelayanan", "sangat;baik", "kurang,ramah"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus pelayanan"
        assert result.iloc[1] == "sangat baik"
        assert result.iloc[2] == "kurang ramah"

    def test_numbers_replaced_with_space(self) -> None:
        series = pd.Series(["pelayanan123", "1sangat2baik3"])
        result = clean_text(series)
        assert result.iloc[0] == "pelayanan"
        assert result.iloc[1] == "sangat baik"

    def test_emojis_and_symbols_removed(self) -> None:
        series = pd.Series(["bagus 😊", "sangat👍baik", "kurang@#$%ramah"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus"
        assert result.iloc[1] == "sangat baik"
        assert result.iloc[2] == "kurang ramah"

    def test_only_a_z_and_spaces_remain(self) -> None:
        series = pd.Series(["Pelayanan!!! yang@ #1 Bagus..."])
        result = clean_text(series)
        assert all(c.islower() or c.isspace() for c in result.iloc[0])
        assert result.iloc[0] == "pelayanan yang bagus"

    def test_multiple_spaces_collapsed(self) -> None:
        series = pd.Series(["bagus    sekali   !!!"])
        result = clean_text(series)
        assert result.iloc[0] == "bagus sekali"

    def test_empty_string(self) -> None:
        series = pd.Series([""])
        result = clean_text(series)
        assert result.iloc[0] == ""

    def test_output_column_name(self) -> None:
        series = pd.Series(["BAIK"])
        result = clean_text(series, output_column="text_cleaned")
        assert result.name == "text_cleaned"


class TestSlangNormalization:
    def test_default_slang_krg_bgt(self) -> None:
        series = pd.Series(["pelayanannya krg bgt"])
        result = normalize_slang(series)
        assert result.iloc[0] == "pelayanannya kurang banget"

    def test_word_boundary_prevents_partial_replacement(self) -> None:
        series = pd.Series(["bagusnya", "kerja", "juga"])
        result = normalize_slang(series, {"jg": "juga"})
        assert result.iloc[0] == "bagusnya"
        assert result.iloc[1] == "kerja"
        assert result.iloc[2] == "juga"

    def test_custom_slang_dict(self) -> None:
        series = pd.Series(["rs bagus", "bpjs lancar"])
        custom = {"rs": "rumah sakit", "bpjs": "jaminan kesehatan"}
        result = normalize_slang(series, custom)
        assert result.iloc[0] == "rumah sakit bagus"
        assert result.iloc[1] == "jaminan kesehatan lancar"

    def test_multiple_slangs_in_one_text(self) -> None:
        series = pd.Series(["yg pelayanannya krg bgt jg"])
        result = normalize_slang(series)
        assert result.iloc[0] == "yang pelayanannya kurang banget juga"

    def test_output_column_name(self) -> None:
        series = pd.Series(["krg"])
        result = normalize_slang(series, output_column="text_normalized")
        assert result.name == "text_normalized"


class TestStopwordsRemoval:
    def test_default_stopwords_removed(self) -> None:
        series = pd.Series(["pelayanannya yang kurang banget"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "pelayanannya kurang banget"

    def test_custom_stopwords_removed(self) -> None:
        series = pd.Series(["rumah sakit bagus"])
        result = remove_stopwords(series, custom_stopwords=["rumah", "sakit"])
        assert result.iloc[0] == "bagus"

    def test_all_stopwords_returns_empty(self) -> None:
        series = pd.Series(["yang dan di"])
        result = remove_stopwords(series)
        assert result.iloc[0] == ""

    def test_no_stopwords_unchanged(self) -> None:
        series = pd.Series(["pelayanan kurang"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "pelayanan kurang"

    def test_multiple_sentences(self) -> None:
        series = pd.Series(["yang bagus dan ramah", "pelayanan yang sangat baik"])
        result = remove_stopwords(series)
        assert result.iloc[0] == "bagus ramah"
        assert result.iloc[1] == "pelayanan baik"

    def test_output_column_name(self) -> None:
        series = pd.Series(["yang bagus"])
        result = remove_stopwords(series, output_column="text_filtered")
        assert result.name == "text_filtered"


class TestStemming:
    def test_stemming_layan_variants(self) -> None:
        series = pd.Series(["pelayanan", "melayani", "dilayani"])
        result = stem_text(series)
        assert result.iloc[0] == "layan"
        assert result.iloc[1] == "layan"
        assert result.iloc[2] == "layan"

    def test_stemming_deterministic(self) -> None:
        series = pd.Series(["pelayanan kurang banget"])
        result1 = stem_text(series)
        result2 = stem_text(series)
        pd.testing.assert_series_equal(result1, result2)

    def test_stemming_common_words(self) -> None:
        series = pd.Series(["membantu", "pembantu", "dibantu"])
        result = stem_text(series)
        assert result.iloc[0] == "bantu"
        assert result.iloc[1] == "bantu"
        assert result.iloc[2] == "bantu"

    def test_empty_string_stemming(self) -> None:
        series = pd.Series([""])
        result = stem_text(series)
        assert result.iloc[0] == ""

    def test_output_column_name(self) -> None:
        series = pd.Series(["pelayanan"])
        result = stem_text(series, output_column="text_final")
        assert result.name == "text_final"


class TestTextMapping:
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
