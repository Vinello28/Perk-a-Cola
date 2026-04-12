"""
Output writer module for the classification pipeline.

Writes classification results to an Excel or CSV file with
``Descrizione`` and ``Label`` columns.
"""

import csv
import logging
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)


def _write_results_xlsx(
    descriptions: list[str], labels: list[str], output_path: Path
) -> None:
    """Write results to an ``.xlsx`` file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Classificazione"

    ws.append(["Descrizione", "Label"])
    for desc, label in zip(descriptions, labels):
        ws.append([desc, label])

    wb.save(output_path)
    wb.close()


def _write_results_csv(
    descriptions: list[str], labels: list[str], output_path: Path
) -> None:
    """Write results to a ``.csv`` file."""
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Descrizione", "Label"])
        for desc, label in zip(descriptions, labels):
            writer.writerow([desc, label])


def write_results(
    descriptions: list[str],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """
    Write classified results to an Excel or CSV file.

    The output format is determined by the file extension of
    ``output_path`` (``.xlsx`` or ``.csv``).

    Parameters
    ----------
    descriptions : list[str]
        Original description texts.
    labels : list[str]
        Predicted labels (must be same length as ``descriptions``).
    output_path : str | Path
        Destination file path (``.xlsx`` or ``.csv``).

    Returns
    -------
    Path
        Absolute path to the created file.

    Raises
    ------
    ValueError
        If ``descriptions`` and ``labels`` have different lengths
        or the file extension is unsupported.
    """
    if len(descriptions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(descriptions)} descriptions "
            f"vs {len(labels)} labels"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext == ".csv":
        _write_results_csv(descriptions, labels, output_path)
    elif ext == ".xlsx":
        _write_results_xlsx(descriptions, labels, output_path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}' (expected .xlsx or .csv)")

    logger.info(
        "Results written to '%s' (%d rows)", output_path, len(descriptions)
    )
    return output_path.resolve()
