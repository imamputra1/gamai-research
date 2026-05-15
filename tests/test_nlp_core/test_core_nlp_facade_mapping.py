# tests/test_core_nlp_facade_mapping.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from src.core_nlp.facade import run_nlp_preprocessing


class TestFacadeMappingOutput:
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
        import json
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
