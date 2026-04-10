# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI pipeline
python app/src/main.py

# Run with a custom config
python app/src/main.py --config path/to/custom_config.yaml

# Run Streamlit GUI
streamlit run app/src/gui.py

# Run containerized (requires Docker Desktop 4.41+ with Model Runner enabled)
docker compose up
```

There are no automated tests or formatters configured.

## Architecture

The pipeline is fully **async** (`asyncio`) and follows a **Strategy pattern** for classification:

- `config.py` — loads `config.yaml` into frozen dataclasses (`PipelineConfig`). Unknown YAML keys are silently ignored. Relative paths (`input_dir`, `output_dir`) are resolved from the project root.
- `data_reader.py` — reads `.xlsx` files from `app/data/`, extracts the target column defined by `classification.description_column`.
- `classifier.py` — `BaseClassifier` ABC defines the Strategy interface. `LMStudioClassifier` is the concrete implementation; it communicates with any OpenAI-compatible endpoint and parses labels via regex.
- `output_writer.py` — writes results back to `app/out/` as `*_classified.xlsx`.
- `main.py` — orchestrates the full pipeline; parses `--config` CLI arg.
- `gui.py` — Streamlit front-end; must call the async pipeline via `asyncio.run(...)`.

**Docker:** `docker-compose.yml` spins up the app container and uses Docker Model Runner (a Docker Desktop built-in feature) to serve the LLM via llama.cpp. The `config.yaml` default `base_url` is `http://model-runner.docker.internal/engines/v1`. For local LM Studio, override to `http://127.0.0.1:1234/v1`. Docker Desktop 4.41+ with Model Runner enabled is required.

## Key Conventions & Pitfalls

- **Concurrency semaphore:** `asyncio.Semaphore(max_workers)` in `LMStudioClassifier` must not be removed — exceeding VRAM limits will crash the inference server.
- **Thinking mode toggle:** When `enable_thinking: false`, `/no_think\n` is prepended to user content (Qwen-specific workaround). Do not break this when modifying prompt logic.
- **Label parsing fallback:** If the regex fails to extract a known label (e.g., due to `max_tokens=32` truncation), the pipeline silently returns `default_label`. Increase `max_tokens` or check the prompt if you see unexpected defaults.
- **Logging over print:** Use the standard `logging` module. CLI progress via `tqdm_asyncio`; GUI progress via callbacks that update `st.session_state` or Streamlit placeholders.
- **Streamlit async:** Streamlit is synchronous — always call async pipeline functions with `asyncio.run(...)` inside button callbacks.
- **Config immutability:** All config dataclasses use `@dataclass(frozen=True)`. Do not attempt in-place mutation.
- **New classifiers:** Extend `BaseClassifier` in `app/src/classifier.py`.

## Workflow Orchestration

### 1. Plan Node Default
-   Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
-   If something goes sideways, STOP and re-plan immediately - don't keep pushing
-   Use plan mode for verification steps, not just building
-   Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
-   Use subagents liberally to keep main contect window clean
-   Offload research, exploration, and parallel analysis to subagents
-   For complex problens, throw more compute at it via subagents
-   One tack per subagent for focused execution

### 3. Self-Improvement Loop
-   After ANY correction from the user: update 'tasks/lessons.md' with the pattern
-   Write rules for yourself that prevent the same mistake
-   Ruthlessly iterate on these lessons until mistake rate drops
-   Review lessons at session start for relevant project

### 4. Verification Before Done
-   Never mark a task complete without proving it works
-   Diff behavior between main and your changes when relevant
-   Ask yourself: "Would a staff engineer approve this?"
-   Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
-   For non-trivial changes: pause and ask "is there a more elegant way?"
-   If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
-   Skip this for simple, chvious fixes - don't over-engineer
-   Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
-   When given a bug report: just fix it. Don't ask for hand-holding
-   Point at logs, errors, failing tests - then resolve them
-   Zero context switching required from the user
-   Go fix failing CI tests without being told how

## Task Management
1.    **PLan First**: Write plan to 'tasks/todo.md' with checkable items
2.    **Verify Plan**: Check in before starting implementation
3.    **Track Progress**: Mark items complete as you go
4.    **Explain Changes**: High-level summary at each step
5.    **Document Results**: Add review section to 'tasks/todo.md"
6.    **Capture Lessons**: Update 'tasks/lessons. md' after corrections

## Core Principles
-   **Simplicity First**: Make every change as simple as possible. Inpact minimal code.
-   **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
-   **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
