# tests/nlp_test/test_orchestrator.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from src.core_nlp.orchestrator import (
    NLPFrequentialOrchestrator,
    NLPPreprocessorOrchestrator,
    run_nlp_frequential,
    run_nlp_preprocessing,
)


class TestNLPPreprocessorOrchestrator:
    def test_run_creates_all_columns(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2],
            "jawaban": [
                "Pelayanannya yg krg bgt.",
                "Rumah sakit ini sangat bagus dan ramah!",
            ],
        })
        orch = NLPPreprocessorOrchestrator(custom_stopwords=["rumah", "sakit"])
        result = orch.run(df, text_column="jawaban")
        assert "text_cleaned" in result.columns
        assert "text_normalized" in result.columns
        assert "text_filtered" in result.columns
        assert "text_final_preprocessed" in result.columns

    def test_run_preserves_original_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "jawaban": ["bagus"]})
        orch = NLPPreprocessorOrchestrator()
        result = orch.run(df, text_column="jawaban")
        assert "id" in result.columns
        assert "jawaban" in result.columns

    def test_run_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"jawaban": ["bagus"]})
        orch = NLPPreprocessorOrchestrator()
        original_columns = list(df.columns)
        _ = orch.run(df, text_column="jawaban")
        assert list(df.columns) == original_columns

    def test_run_series_returns_series(self) -> None:
        series = pd.Series(["Pelayanannya yg krg bgt."])
        orch = NLPPreprocessorOrchestrator()
        result = orch.run_series(series)
        assert isinstance(result, pd.Series)
        assert "layan" in result.iloc[0]

    def test_custom_slang_dict_injected(self) -> None:
        df = pd.DataFrame({"jawaban": ["ok banget"]})
        orch = NLPPreprocessorOrchestrator(slang_dict={"ok": "baik"})
        result = orch.run(df, text_column="jawaban")
        assert "baik" in result["text_normalized"].iloc[0]

    def test_custom_stopwords_injected(self) -> None:
        df = pd.DataFrame({"jawaban": ["rumah sakit bagus"]})
        orch = NLPPreprocessorOrchestrator(custom_stopwords=["rumah", "sakit"])
        result = orch.run(df, text_column="jawaban")
        assert result["text_filtered"].iloc[0] == "bagus"


class TestNLPFrequentialOrchestrator:
    def test_analyze_generates_wordcloud_for_unigram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "text_final_preprocessed": [
                    "bagus sekali pelayanan",
                    "bagus ramah dokter",
                    "kurang bagus",
                ]
            })
            orch = NLPFrequentialOrchestrator(top_k=5, ngram_ranges=[1])
            results = orch.analyze(df, "text_final_preprocessed", "test_q1", reports_dir=tmpdir)
            assert 1 in results
            assert (Path(tmpdir) / "wordcloud_test_q1.png").exists()

    def test_analyze_generates_barchart_for_bigram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "text_final_preprocessed": [
                    "dokter ramah sekali",
                    "dokter ramah baik",
                    "perawat baik",
                ]
            })
            orch = NLPFrequentialOrchestrator(top_k=5, ngram_ranges=[2])
            results = orch.analyze(df, "text_final_preprocessed", "test_q2", reports_dir=tmpdir)
            assert 2 in results
            assert (Path(tmpdir) / "ngram_2_test_q2.png").exists()

    def test_analyze_generates_all_visualizations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "text_final_preprocessed": [
                    "bagus sekali pelayanan dokter ramah",
                    "bagus ramah baik perawat",
                    "kurang bagus fasilitas",
                ]
            })
            orch = NLPFrequentialOrchestrator(top_k=5, ngram_ranges=[1, 2, 3])
            results = orch.analyze(df, "text_final_preprocessed", "test_q3", reports_dir=tmpdir)
            assert (Path(tmpdir) / "wordcloud_test_q3.png").exists()
            assert (Path(tmpdir) / "ngram_2_test_q3.png").exists()
            assert (Path(tmpdir) / "ngram_3_test_q3.png").exists()

    def test_analyze_returns_dataframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "text_final_preprocessed": ["bagus sekali pelayanan"]
            })
            orch = NLPFrequentialOrchestrator(top_k=3, ngram_ranges=[1])
            results = orch.analyze(df, "text_final_preprocessed", "test_q4", reports_dir=tmpdir)
            assert isinstance(results[1], pd.DataFrame)
            assert list(results[1].columns) == ["term", "frequency"]


class TestRunNlpPreprocessingIntegration:
    def test_creates_mapping_files_in_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            data_master = root / "data" / "04_report_master"
            data_synthetic = root / "data" / "03_synthetic"
            data_processed = root / "data" / "02_processed"

            for d in [config_dir, data_master, data_synthetic, data_processed]:
                d.mkdir(parents=True, exist_ok=True)

            config = {
                "paths": {
                    "report_master": str(data_master),
                    "synthetic": str(data_synthetic),
                    "processed": str(data_processed),
                },
                "nlp": {
                    "text_columns": {"q1_test": "jawaban"},
                    "custom_stopwords": [],
                    "custom_slang": {},
                    "output_prefix": "test_preprocessed",
                },
            }
            config_path = config_dir / "pipeline_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            df = pd.DataFrame({"id": [1, 2], "jawaban": ["bagus sekali", "kurang"]})
            df.to_csv(data_master / "df_report_master.csv", index=False)
            df.to_csv(data_synthetic / "df_teks_syn_2.csv", index=False)

            _ = run_nlp_preprocessing(str(config_path))

            mapping_dir = root / "reports" / "text_preprocessing"
            assert mapping_dir.exists()
            assert (mapping_dir / "mapping_q1_test.json").exists()
            assert (mapping_dir / "mapping_q1_test.csv").exists()

    def test_mapping_json_contains_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            data_master = root / "data" / "04_report_master"
            data_synthetic = root / "data" / "03_synthetic"
            data_processed = root / "data" / "02_processed"

            for d in [config_dir, data_master, data_synthetic, data_processed]:
                d.mkdir(parents=True, exist_ok=True)

            config = {
                "paths": {
                    "report_master": str(data_master),
                    "synthetic": str(data_synthetic),
                    "processed": str(data_processed),
                },
                "nlp": {
                    "text_columns": {"q1_test": "jawaban"},
                    "custom_stopwords": [],
                    "custom_slang": {},
                    "output_prefix": "test_preprocessed",
                },
            }
            config_path = config_dir / "pipeline_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            df = pd.DataFrame({"id": [1], "jawaban": ["Bagus. Sekali!"]})
            df.to_csv(data_master / "df_report_master.csv", index=False)
            df.to_csv(data_synthetic / "df_teks_syn_2.csv", index=False)

            _ = run_nlp_preprocessing(str(config_path))

            mapping_path = root / "reports" / "text_preprocessing" / "mapping_q1_test.json"
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)

            assert "Bagus. Sekali!" in mapping
            assert "cleaned" in mapping["Bagus. Sekali!"]
            assert "normalized" in mapping["Bagus. Sekali!"]
            assert "filtered" in mapping["Bagus. Sekali!"]
            assert "final" in mapping["Bagus. Sekali!"]


class TestRunNlpFrequentialIntegration:
    def test_runs_on_preprocessed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            data_processed = root / "data" / "02_processed"
            reports_dir = root / "reports" / "figures"

            for d in [config_dir, data_processed, reports_dir]:
                d.mkdir(parents=True, exist_ok=True)

            config = {
                "paths": {
                    "processed": str(data_processed),
                    "reports_figures": str(reports_dir),
                },
                "nlp": {
                    "text_columns": {"q1_test": "jawaban"},
                    "output_prefix": "test_preprocessed",
                    "frequential": {
                        "top_k": 5,
                        "ngram_ranges": [1, 2],
                    },
                },
            }
            config_path = config_dir / "pipeline_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            df = pd.DataFrame({
                "id": [1, 2],
                "text_final_preprocessed": [
                    "bagus sekali pelayanan",
                    "bagus ramah dokter",
                ],
            })
            df.to_csv(data_processed / "test_preprocessed_q1_test.csv", index=False)

            results = run_nlp_frequential(str(config_path))

            assert "q1_test" in results
            assert 1 in results["q1_test"]
            assert 2 in results["q1_test"]
            assert (reports_dir / "wordcloud_q1_test.png").exists()
            assert (reports_dir / "ngram_2_q1_test.png").exists()
