# src/utils/excel_exporter.py
from pathlib import Path

import pandas as pd


def export_dataframe_to_excel(
    df: pd.DataFrame,
    filepath: str | Path,
    sheet_name: str = "Sheet1",
) -> None:
    """Export DataFrame to Excel workbook.

    Args:
        df: DataFrame to export.
        filepath: Destination path.
        sheet_name: Target sheet name.
    """
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(dest, sheet_name=sheet_name, index=False)
