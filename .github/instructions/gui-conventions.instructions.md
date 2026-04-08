---
description: "Streamlit GUI best practices, state management, and async execution rules."
applyTo:
  - "app/src/gui.py"
---

# Streamlit GUI Conventions

When working on the Perk-a-Cola Streamlit GUI (`app/src/gui.py`), adhere to the following conventions to ensure a responsive and reliable application:

## State Management
- **`st.session_state`:** Always use `st.session_state` to persist data, configuration states, and long-running pipeline statuses across Streamlit's script reruns.
- **Initialization:** Initialize necessary session state variables at the beginning of the script to prevent `KeyError` exceptions on first load.

## Async Integration inside Streamlit
- **Event Loops:** Streamlit is strictly synchronous. To execute the `asyncio`-based classification pipeline, you must use `asyncio.run(pipeline_function(...))` within a button callback or execution block.
- **Callback Tracking:** Because you cannot natively yield to Streamlit during an async loop without custom thread handling, ensure progress updates are passed via callbacks that modify `st.session_state` or directly update Streamlit placeholder elements (e.g., `progress_bar = st.progress(0)` mapped to an async callback).

## Performance & Caching
- **Caching:** Use `@st.cache_data` for loading the initial YAML configuration or reading standard layout elements to prevent unnecessary disk I/O on every rerun. 

## UI Layout
- **Separation of Concerns:** Keep configuration controls (like model selection, endpoints, and concurrency limits) in the `st.sidebar`.
- **Feedback:** Provide immediate, clear feedback using `st.info()`, `st.warning()`, and `st.error()` for missing configurations or LM Studio connection failures.