"""
Classifier module for the classification pipeline.

Provides an abstract classifier interface (Strategy pattern) and a
concrete implementation that uses LM Studio's OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from config import (
    ClassificationConfig,
    ConcurrencyConfig,
    LLMConfig,
    PromptConfig,
)

logger = logging.getLogger(__name__)


# ── Abstract Interface ──────────────────────────────────────────────


class BaseClassifier(ABC):
    """Strategy interface for text classifiers."""

    @abstractmethod
    async def classify(self, description: str) -> str:
        """Classify a single description and return a label string."""

    @abstractmethod
    async def classify_batch(
        self,
        descriptions: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        """Classify a list of descriptions and return labels."""


# ── LM Studio Implementation ────────────────────────────────────────


class LMStudioClassifier(BaseClassifier):
    """
    Classifier backed by an LM Studio server (OpenAI-compatible API).

    Uses async HTTP requests with a semaphore to process thousands of
    descriptions concurrently without overwhelming the server.
    """

    def __init__(
        self,
        llm_cfg: LLMConfig,
        cls_cfg: ClassificationConfig,
        prompt_cfg: PromptConfig,
        concurrency_cfg: ConcurrencyConfig,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=llm_cfg.base_url,
            api_key="lm-studio",  # LM Studio ignores the key
        )
        self._model = llm_cfg.model_name
        self._temperature = llm_cfg.temperature
        self._max_tokens = llm_cfg.max_tokens
        self._enable_thinking = llm_cfg.enable_thinking

        self._labels = cls_cfg.labels
        self._default_label = cls_cfg.default_label

        # Pre-compile regex: matches any label as whole word, case-insensitive
        escaped = [re.escape(lbl) for lbl in self._labels]
        self._label_pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
        )

        # Build prompts with label list injected
        labels_str = ", ".join(self._labels)
        self._system_prompt = prompt_cfg.system.format(labels=labels_str)
        self._user_template = prompt_cfg.user

        self._semaphore = asyncio.Semaphore(concurrency_cfg.max_workers)
        self._log_interval = concurrency_cfg.batch_log_interval

    # ── Core classification ──────────────────────────────────────

    async def classify(self, description: str) -> str:
        """Send one description to the LLM and return the parsed label."""
        user_content = self._user_template.format(description=description)

        # Prepend /no_think directive if thinking is disabled
        if not self._enable_thinking:
            user_content = "/no_think\n" + user_content

        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                raw_output = response.choices[0].message.content or ""
            except Exception:
                logger.exception(
                    "LLM request failed for description: %.80s…", description
                )
                return self._default_label

        return self._parse_label(raw_output)

    async def classify_batch(
        self,
        descriptions: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        """
        Classify many descriptions concurrently.

        Uses an asyncio semaphore to cap the number of in-flight
        requests (configured via ``concurrency.max_workers``).

        Parameters
        ----------
        descriptions : list[str]
            Texts to classify.
        progress_callback : callable, optional
            Called as ``progress_callback(completed, total)`` after each
            classification finishes.  Used by the Streamlit GUI to
            update the progress bar in real time.
        """
        total = len(descriptions)

        if progress_callback is None:
            # CLI mode – keep the original tqdm behaviour
            tasks = [self.classify(desc) for desc in descriptions]
            results: list[str] = await tqdm_asyncio.gather(
                *tasks,
                desc="Classifying",
                unit="desc",
            )
            return results

        # GUI mode – use an atomic counter + callback
        counter_lock = asyncio.Lock()
        done_count = 0

        async def _classify_and_report(desc: str) -> str:
            nonlocal done_count
            label = await self.classify(desc)
            async with counter_lock:
                done_count += 1
                current = done_count
            progress_callback(current, total)
            return label

        tasks = [_classify_and_report(desc) for desc in descriptions]
        results: list[str] = list(await asyncio.gather(*tasks))
        return results

    # ── Output parsing ───────────────────────────────────────────

    def _parse_label(self, raw_output: str) -> str:
        """
        Extract a valid label from the LLM response.

        Strategy:
          1. Strip thinking tags (``<think>…</think>``) if present.
          2. Search for an exact label match via regex.
          3. Fall back to the default label with a warning.
        """
        # Remove <think>...</think> blocks (Qwen3 thinking output)
        cleaned = re.sub(
            r"<think>.*?</think>", "", raw_output, flags=re.DOTALL
        ).strip()

        match = self._label_pattern.search(cleaned)
        if match:
            # Normalise to the canonical casing from config
            found = match.group(1).lower()
            for label in self._labels:
                if label.lower() == found:
                    return label
            return found  # pragma: no cover – safety fallback

        logger.warning(
            "Could not parse label from LLM output: '%.120s' → "
            "defaulting to '%s'",
            raw_output,
            self._default_label,
        )
        return self._default_label
