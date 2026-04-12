# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the CLI pipeline:**
```bash
python app/src/main.py
# or with a custom config:
python app/src/main.py --config path/to/custom_config.yaml
```

**Run the Streamlit GUI:**
```bash
streamlit run app/src/gui.py
```

> All source modules (`config`, `data_reader`, `classifier`, `output_writer`) are imported using bare names (no package prefix), so scripts must be run from within `app/src/` or the runner must add it to `sys.path`. Streamlit handles this automatically; for the CLI, run via `python app/src/main.py` from the project root.

## Architecture

The pipeline has two entry points that share the same core modules:

- **`main.py`** — CLI orchestrator. Discovers `.xlsx` files in `app/data/`, runs async batch classification, writes results to `app/out/`.
- **`gui.py`** — Streamlit web UI. Accepts file uploads, lets users edit the system prompt inline, polls a background thread for progress updates, and offers a download button for results.

Core modules:

| File | Role |
|---|---|
| `config.py` | Loads `config.yaml` into frozen dataclasses (`PipelineConfig`, `LLMConfig`, etc.). Unknown YAML keys are silently ignored. |
| `classifier.py` | `BaseClassifier` (Strategy ABC) + `LMStudioClassifier` (async, OpenAI-compatible). Uses `asyncio.Semaphore` to cap concurrent requests. Strips `<think>…</think>` blocks before regex-matching labels. |
| `data_reader.py` | Reads the target column from `.xlsx` files via `openpyxl`. |
| `output_writer.py` | Writes description + label pairs back to `.xlsx`. |

## Configuration (`app/src/config.yaml`)

Key knobs:

- `llm.model_name` — must match the model identifier shown in LM Studio.
- `llm.enable_thinking` — set `false` to prepend `/no_think` to each user message (for non-reasoning models).
- `classification.labels` — list of valid output labels; the classifier regex-matches against these.
- `classification.default_label` — fallback when the model output cannot be parsed.
- `concurrency.max_workers` — semaphore cap; tune based on available VRAM.

The GUI reads this file at startup for display and uses the system prompt from it as the default editable value, but allows overriding the prompt at runtime without touching the file.

## Prerequisites

LM Studio must be running on `http://127.0.0.1:1234` with a model loaded before invoking either entry point. The `api_key` sent is the literal string `"lm-studio"` (ignored by LM Studio).


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
-   Point at logs,errors, failing tests - then resolve them
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
