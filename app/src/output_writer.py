"""
Output writer module for the classification pipeline.

Writes classification results to an Excel file with
``Descrizione`` and ``Label`` columns.
"""

import logging
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)


def write_results(
    descriptions: list[str],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """
    Write classified results to an Excel file.

    Parameters
    ----------
    descriptions : list[str]
        Original description texts.
    labels : list[str]
        Predicted labels (must be same length as ``descriptions``).
    output_path : str | Path
        Destination ``.xlsx`` file path.

    Returns
    -------
    Path
        Absolute path to the created file.

    Raises
    ------
    ValueError
        If ``descriptions`` and ``labels`` have different lengths.
    """
    if len(descriptions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(descriptions)} descriptions "
            f"vs {len(labels)} labels"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Classificazione"

    # Header row
    ws.append(["Descrizione", "Label"])

    # Data rows
    for desc, label in zip(descriptions, labels):
        ws.append([desc, label])

    wb.save(output_path)
    wb.close()

    logger.info(
        "Results written to '%s' (%d rows)", output_path, len(descriptions)
    )
    return output_path.resolve()
