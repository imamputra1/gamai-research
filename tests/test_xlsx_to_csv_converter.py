# tests/test_xlsx_to_csv_converter.py
from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils.xlsx_to_csv_converter import (
    XlsxToCsvOrchestrator,
    discover_xlsx_files,
    enrich_with_metadata,
    export_to_csv,
    read_all_sheets,
    stack_dataframes,
)


def test_discover_xlsx_files_finds_only_xlsx(tmp_path: Path) -> None:
    """Must return only .xlsx files, sorted, from existing directories."""
    d1: Path = tmp_path / "dir1"
    d2: Path = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()

    (d1 / "a.xlsx").touch()
    (d1 / "b.csv").touch()
    (d2 / "c.xlsx").touch()
    (d2 / "d.txt").touch()

    result: List[Path] = discover_xlsx_files([d1, d2])
    names: List[str] = [p.name for p in result]

    assert names == ["a.xlsx", "c.xlsx"]


def test_discover_xlsx_files_skips_missing_dirs(tmp_path: Path) -> None:
    """Must silently skip non-existent directories."""
    missing: Path = tmp_path / "ghost"
    result: List[Path] = discover_xlsx_files([missing])
    assert result == []


def test_read_all_sheets_returns_multiple_frames(tmp_path: Path) -> None:
    """Must return one dataframe per sheet in workbook order."""
    path: Path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"A": [1]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"B": [2]}).to_excel(writer, sheet_name="Sheet2", index=False)

    sheets = read_all_sheets(path)
    assert len(sheets) == 2
    assert sheets[0][0] == "Sheet1"
    assert sheets[1][0] == "Sheet2"
    assert sheets[0][1]["A"].iloc[0] == 1
    assert sheets[1][1]["B"].iloc[0] == 2


def test_enrich_with_metadata_inserts_columns() -> None:
    """Must prepend _source_file and _sheet_name columns."""
    df: pd.DataFrame = pd.DataFrame({"X": [1, 2]})
    result: pd.DataFrame = enrich_with_metadata(df, "file.xlsx", "Data")

    assert list(result.columns[:2]) == ["_source_file", "_sheet_name"]
    assert result["_source_file"].unique()[0] == "file.xlsx"
    assert result["_sheet_name"].unique()[0] == "Data"


def test_stack_dataframes_vertical_concat() -> None:
    """Must stack dataframes vertically with union of columns."""
    df1: pd.DataFrame = pd.DataFrame({"A": [1], "B": [2]})
    df2: pd.DataFrame = pd.DataFrame({"A": [3], "C": [4]})
    result: pd.DataFrame = stack_dataframes([df1, df2])

    assert result.shape == (2, 3)
    assert pd.isna(result.loc[1, "B"])
    assert pd.isna(result.loc[0, "C"])


def test_stack_dataframes_empty_list_returns_empty() -> None:
    """Must return empty DataFrame when given empty list."""
    result: pd.DataFrame = stack_dataframes([])
    assert result.empty


def test_export_to_csv_creates_file(tmp_path: Path) -> None:
    """Must create parent directories and write CSV with BOM."""
    df: pd.DataFrame = pd.DataFrame({"X": [1]})
    target: Path = tmp_path / "nested" / "out.csv"
    export_to_csv(df, target)

    assert target.exists()
    content: str = target.read_text(encoding="utf-8-sig")
    assert "X" in content


def test_orchestrator_executes_end_to_end(tmp_path: Path) -> None:
    """Must discover, read, stack, and export in one pass."""
    src: Path = tmp_path / "reports" / "tables"
    src.mkdir(parents=True)
    out: Path = tmp_path / "csv_export"

    path: Path = src / "data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"A": [1, 2]}).to_excel(writer, sheet_name="S1", index=False)
        pd.DataFrame({"A": [3, 4]}).to_excel(writer, sheet_name="S2", index=False)

    orch: XlsxToCsvOrchestrator = XlsxToCsvOrchestrator(
        source_dirs=[str(src)],
        output_dir=str(out),
        output_filename="merged.csv",
    )
    result: Path = orch.execute()

    assert result.exists()
    df: pd.DataFrame = pd.read_csv(result)
    assert df.shape == (4, 3)  # 2 sheets x 2 rows + 2 metadata cols
    assert "_source_file" in df.columns
    assert "_sheet_name" in df.columns


def test_orchestrator_raises_when_no_files(tmp_path: Path) -> None:
    """Must raise FileNotFoundError when source directories are empty."""
    empty: Path = tmp_path / "empty"
    empty.mkdir()

    orch: XlsxToCsvOrchestrator = XlsxToCsvOrchestrator(
        source_dirs=[str(empty)],
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(FileNotFoundError):
        orch.execute()
