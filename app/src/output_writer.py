"""
Output writer module for the classification pipeline.

Writes classification results to an Excel or CSV file with
``Descrizione`` and ``Label`` columns.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def write_results(
    descriptions: list[str],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """
    Write classified results to an Excel or CSV file.

    Parameters
    ----------
    descriptions : list[str]
        Original description texts.
    labels : list[str]
        Predicted labels (must be same length as ``descriptions``).
    output_path : str | Path
        Destination file path.

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

    df = pd.DataFrame({"Descrizione": descriptions, "Label": labels})
    
    if output_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False, sheet_name="Classificazione")

    logger.info(
        "Results written to '%s' (%d rows)", output_path, len(descriptions)
    )
    return output_path.resolve()
