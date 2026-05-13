# src/utils/xlsx_to_csv_converter.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.constants import (
    CSV_INDEX,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SOURCE_DIRS,
    ENCODING_UTF8_SIG,
    FILE_EXTENSION_XLSX,
    METADATA_COL_SHEET,
    METADATA_COL_SOURCE,
)


def discover_xlsx_files(directories: List[Path]) -> List[Path]:
    """Discover all .xlsx files across multiple directories.

    Args:
        directories: List of directory paths to scan.

    Returns:
        Sorted list of discovered .xlsx file paths.
    """
    files: List[Path] = []
    for directory in directories:
        if directory.exists() and directory.is_dir():
            files.extend(sorted(directory.glob(f"*{FILE_EXTENSION_XLSX}")))
    return files


def read_all_sheets(filepath: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Read every sheet from an Excel workbook into separate DataFrames.

    Args:
        filepath: Path to the .xlsx file.

    Returns:
        List of (sheet_name, dataframe) tuples in workbook order.
    """
    xl_file: pd.ExcelFile = pd.ExcelFile(filepath)
    sheets: List[Tuple[str, pd.DataFrame]] = []
    for sheet_name in xl_file.sheet_names:
        df: pd.DataFrame = pd.read_excel(xl_file, sheet_name=sheet_name)
        sheets.append((sheet_name, df))
    return sheets


def enrich_with_metadata(
    df: pd.DataFrame,
    source_file: str,
    sheet_name: str,
) -> pd.DataFrame:
    """Prepend metadata columns for downstream traceability.

    Args:
        df: Raw dataframe from a single sheet.
        source_file: Original filename.
        sheet_name: Sheet name within the workbook.

    Returns:
        pd.DataFrame: Copy with metadata columns inserted at index 0 and 1.
    """
    df = df.copy()
    df.insert(0, METADATA_COL_SOURCE, source_file)
    df.insert(1, METADATA_COL_SHEET, sheet_name)
    return df


def stack_dataframes(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Vertically concatenate a list of DataFrames, ignoring index.

    Args:
        frames: List of DataFrames to stack.

    Returns:
        pd.DataFrame: Consolidated dataframe. Returns empty frame if input
            list is empty.
    """
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def export_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Persist DataFrame to CSV with Excel-compatible UTF-8 BOM encoding.

    Args:
        df: DataFrame to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=CSV_INDEX,
        encoding=ENCODING_UTF8_SIG,
    )


class XlsxToCsvOrchestrator:
    """Orchestrates bulk XLSX-to-CSV consolidation.

    Scans source directories, reads all sheets from every workbook,
    optionally injects metadata, stacks everything vertically, and
    writes a single CSV file.
    """

    def __init__(
        self,
        source_dirs: List[str] | None = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        include_metadata: bool = True,
    ) -> None:
        self.source_dirs: List[Path] = [
            Path(d) for d in (source_dirs or list(DEFAULT_SOURCE_DIRS))
        ]
        self.output_dir: Path = Path(output_dir)
        self.output_filename: str = output_filename
        self.include_metadata: bool = include_metadata

    def execute(self) -> Path:
        """Run the conversion pipeline.

        Returns:
            Path to the generated consolidated CSV file.

        Raises:
            FileNotFoundError: If no .xlsx files are discovered.
        """
        xlsx_files: List[Path] = discover_xlsx_files(self.source_dirs)
        if not xlsx_files:
            raise FileNotFoundError(
                f"No {FILE_EXTENSION_XLSX} files found in {self.source_dirs}"
            )

        all_frames: List[pd.DataFrame] = []

        for filepath in xlsx_files:
            sheets: List[Tuple[str, pd.DataFrame]] = read_all_sheets(filepath)
            for sheet_name, df in sheets:
                if self.include_metadata:
                    df = enrich_with_metadata(
                        df, source_file=filepath.name, sheet_name=sheet_name
                    )
                all_frames.append(df)

        consolidated: pd.DataFrame = stack_dataframes(all_frames)

        output_path: Path = self.output_dir / self.output_filename
        export_to_csv(consolidated, output_path)

        print(
            f"[SUCCESS] Converted {len(xlsx_files)} files "
            f"({len(all_frames)} sheets) → {output_path}"
        )
        return output_path


def convert_xlsx_to_csv(
    source_dirs: List[str] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> None:
    """Convenience entry-point for XLSX-to-CSV bulk conversion.

    Args:
        source_dirs: Directories to scan. Defaults to
            ``reports/analisis/tables`` and ``reports/tables``.
        output_dir: Destination directory for the CSV.
        output_filename: Name of the output CSV file.
    """
    orchestrator: XlsxToCsvOrchestrator = XlsxToCsvOrchestrator(
        source_dirs=source_dirs,
        output_dir=output_dir,
        output_filename=output_filename,
    )
    orchestrator.execute()


def _cli() -> None:
    """Parse CLI arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Bulk XLSX-to-CSV converter with sheet stacking."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=None,
        help="One or more directories to scan for .xlsx files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the consolidated CSV.",
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Name of the output CSV file.",
    )
    args = parser.parse_args()

    convert_xlsx_to_csv(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    _cli()
