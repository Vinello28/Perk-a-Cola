"""
Data reader module for the classification pipeline.

Handles reading Excel and CSV files and extracting the target column
for classification.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def read_descriptions(file_path: str | Path, column_name: str) -> list[str]:
    """
    Read descriptions from an Excel or CSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.
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

    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        
    descriptions = df[column_name].fillna("").astype(str).tolist()

    logger.info(
        "Loaded %d descriptions from '%s' (column: '%s')",
        len(descriptions),
        file_path.name,
        column_name,
    )
    return descriptions
