"""
Data reader module for the classification pipeline.

Handles reading Excel files and extracting the target column
for classification.
"""

import logging
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)


def read_descriptions(file_path: str | Path, column_name: str) -> list[str]:
    """
    Read descriptions from an Excel file.

    Parameters
    ----------
    file_path : str | Path
        Path to the ``.xlsx`` file.
    column_name : str
        Name of the column to extract (must match header exactly).

    Returns
    -------
    list[str]
        List of description strings (empty cells become ``""``).

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If ``column_name`` is not found in the header row.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active

    # --- Locate the target column index ---
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    try:
        col_idx = headers.index(column_name)
    except ValueError:
        wb.close()
        raise ValueError(
            f"Column '{column_name}' not found. "
            f"Available columns: {headers}"
        )

    # --- Extract descriptions ---
    descriptions: list[str] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        value = row[col_idx] if col_idx < len(row) else None
        descriptions.append(str(value).strip() if value is not None else "")

    wb.close()
    logger.info(
        "Loaded %d descriptions from '%s' (column: '%s')",
        len(descriptions),
        file_path.name,
        column_name,
    )
    return descriptions
