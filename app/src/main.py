"""
Main pipeline orchestrator for the classification system.

Usage::

    python main.py                          # default config.yaml
    python main.py --config path/to/cfg.yaml  # custom config
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from classifier import LMStudioClassifier
from config import load_config
from data_reader import read_descriptions
from output_writer import write_results

# ── Logging setup ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Pipeline ─────────────────────────────────────────────────────


async def run_pipeline(config_path: str | None = None) -> None:
    """Execute the full classification pipeline."""
    # 1. Load configuration
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent

    input_dir = project_root / cfg.paths.input_dir
    output_dir = project_root / cfg.paths.output_dir

    logger.info("Configuration loaded")
    logger.info("  Model        : %s", cfg.llm.model_name)
    logger.info("  Thinking     : %s", cfg.llm.enable_thinking)
    logger.info("  Labels       : %s", cfg.classification.labels)
    logger.info("  Max workers  : %d", cfg.concurrency.max_workers)
    logger.info("  Input dir    : %s", input_dir)
    logger.info("  Output dir   : %s", output_dir)

    # 2. Discover input files
    xlsx_files = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.error("No .xlsx files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d input file(s): %s", len(xlsx_files),
                [f.name for f in xlsx_files])

    # 3. Build classifier
    classifier = LMStudioClassifier(
        llm_cfg=cfg.llm,
        cls_cfg=cfg.classification,
        prompt_cfg=cfg.prompt,
        concurrency_cfg=cfg.concurrency,
    )

    # 4. Process each file
    for xlsx_path in xlsx_files:
        logger.info("━" * 60)
        logger.info("Processing: %s", xlsx_path.name)

        # Read
        descriptions = read_descriptions(
            xlsx_path, cfg.classification.description_column
        )

        if not descriptions:
            logger.warning("No descriptions found, skipping.")
            continue

        # Classify
        t0 = time.perf_counter()
        labels = await classifier.classify_batch(descriptions)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Classified %d descriptions in %.1fs (%.1f desc/s)",
            len(descriptions),
            elapsed,
            len(descriptions) / elapsed if elapsed > 0 else 0,
        )

        # Write output
        out_name = f"{xlsx_path.stem}_classified.xlsx"
        out_path = output_dir / out_name
        write_results(descriptions, labels, out_path)

    logger.info("━" * 60)
    logger.info("Pipeline complete.")


# ── CLI entry point ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based text classification pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.config))


if __name__ == "__main__":
    main()
