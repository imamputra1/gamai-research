# tests/test_core_nlp_facade.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.core_nlp.facade import run_nlp_preprocessing


class TestFacade:
    def test_facade_reads_config_and_processes(self) -> None:
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
                    "custom_stopwords": ["test"],
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

            results = run_nlp_preprocessing(str(config_path))

            assert "q1_test" in results
            assert "text_final_preprocessed" in results["q1_test"].columns
            assert (data_processed / "test_preprocessed_q1_test.csv").exists()

    def test_facade_raises_on_missing_column(self) -> None:
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
                    "text_columns": {"q1": "nonexistent_column"},
                },
            }
            config_path = config_dir / "pipeline_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            df = pd.DataFrame({"id": [1]})
            df.to_csv(data_master / "df_report_master.csv", index=False)
            df.to_csv(data_synthetic / "df_teks_syn_2.csv", index=False)

            with pytest.raises(KeyError):
                run_nlp_preprocessing(str(config_path))
