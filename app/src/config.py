"""
Configuration loader for the classification pipeline.

Loads settings from config.yaml into a strongly-typed dataclass,
providing validation and sensible defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    """LLM server connection and generation settings."""

    base_url: str = "http://127.0.0.1:1234/v1"
    model_name: str = "qwen3-4b"
    temperature: float = 0.0
    max_tokens: int = 32
    enable_thinking: bool = True


@dataclass(frozen=True)
class ClassificationConfig:
    """Classification task settings (labels, column, prompt)."""

    labels: list[str] = field(default_factory=lambda: ["ai", "non_ai"])
    default_label: str = "non_ai"
    description_column: str = "Descrizione"


@dataclass(frozen=True)
class PromptConfig:
    """Prompt templates for the LLM."""

    system: str = ""
    user: str = ""


@dataclass(frozen=True)
class PathsConfig:
    """I/O directory paths."""

    input_dir: str = "app/data"
    output_dir: str = "app/out"


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Concurrency settings for batch processing."""

    max_workers: int = 10
    batch_log_interval: int = 100


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration aggregating all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    classification: ClassificationConfig = field(
        default_factory=ClassificationConfig
    )
    prompt: PromptConfig = field(default_factory=PromptConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


def _build_sub_config(cls: type, raw: dict[str, Any] | None):
    """Build a dataclass instance from a raw dict, ignoring unknown keys."""
    if raw is None:
        return cls()
    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return cls(**filtered)


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to the YAML config file. If None, looks for
        ``config.yaml`` next to this module.

    Returns
    -------
    PipelineConfig
        Fully-populated configuration object.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return PipelineConfig(
        llm=_build_sub_config(LLMConfig, raw.get("llm")),
        classification=_build_sub_config(
            ClassificationConfig, raw.get("classification")
        ),
        prompt=_build_sub_config(PromptConfig, raw.get("prompt")),
        paths=_build_sub_config(PathsConfig, raw.get("paths")),
        concurrency=_build_sub_config(
            ConcurrencyConfig, raw.get("concurrency")
        ),
    )
