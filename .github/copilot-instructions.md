# Perk-a-Cola Project Guidelines

These instructions define the architectural conventions, testing commands, and potential pitfalls for developing the Perk-a-Cola local LLM classification pipeline.

## Code Style & Best Practices
- **Async-first Architecture:** All classification operations run via `asyncio`. Ensure any new IO-bound operations (like file reading or API calls) use `async`/`await`.
- **Typing & Immutability:** Use `from __future__ import annotations`, exhaustive type hints, and `@dataclass(frozen=True)` for all configuration objects to make them immutable after loading.
- **Structured Logging:** Use the standard Python `logging` module with ISO time formatting instead of `print` statements. Support dual progress tracking (`tqdm_asyncio` for CLI, callbacks for Streamlit GUI).

## Architecture
- **Strategy Pattern (`app/src/classifier.py`):** Implementing new models requires extending the `BaseClassifier` abstract base class.
- **Regex Parsing:** Classification parsing is regex-based to safely handle reasoning models that output `<think>...</think>` tags alongside the JSON/text classification.
- **Config-driven (`app/src/config.py`):** Uses strict YAML configs. Unknown keys in YAML are silently ignored by design. Relative paths (like `input_dir`, `output_dir`) are resolved from the project root.

## Build and Run
There are no automated testing frameworks (like pytest) or formatters configured yet.
- **CLI Pipeline:** `python app/src/main.py`
- **Streamlit GUI:** `streamlit run app/src/gui.py`
- **Dependencies:** Install via `pip install -r requirements.txt`.

## Conventions & Pitfalls
- **LM Studio VRAM Throttling:** Concurrency is managed via `asyncio.Semaphore(max_workers)`. Do not remove the semaphore, as setting it too high will crash the local LM Studio instance due to VRAM limits.
- **Hardcoded Endpoint:** LM Studio's default `base_url` is assumed to be `http://127.0.0.1:1234/v1`.
- **Parsing Fallback:** If the regex map fails to match a classification output (often due to context limits `max_tokens=32`), the pipeline defaults to the config's `default_label`.
- **Model specific fixes:** When thinking mode is disabled for Qwen models, `/no_think\n` is prepended to user content. Be careful not to break this workaround when modifying prompts.

## References
- See [README.md](../README.md) for general setup and pipeline execution.
- See `/docs/USER_GUIDE.md` for end-user configurations.