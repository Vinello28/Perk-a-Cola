# Project Containerization & Classification Taxonomy Update Plan

## Phase 1: Planning and Discovery
- [x] Create this `todo.md` and `lessons.md`.
- [x] Read `app/src/config.yaml`, `app/src/config.py`, and `app/src/classifier.py` to understand the current configuration structure and prompt setup.

## Phase 2: Configuration Update
- [x] Update `app/src/config.yaml` to hit `http://vllm:8000/v1` instead of `http://127.0.0.1:1234/v1`.
- [x] Modify the taxonomy in `app/src/config.yaml` to the 8 new categories (Autonomous driving, Healthcare AI, Robotics AI, Research, Data Science, Virtual assistants, Fintech, Enterprise AI) along with their descriptions.
- [x] Update `app/src/config.py` structural dependencies if they dictate specific taxonomy format.
- [x] Update `app/src/classifier.py` prompts to utilize the new taxonomy cleanly if hardcoded.

## Phase 3: Containerization
- [x] Create `docker/Dockerfile` for the Python application with necessary dependencies.
- [x] Create a `docker-compose.yml` in the root configuring the built app container and the vLLM container (`Qwen/Qwen-3.5-4B-Instruct` or similar) exposing an OpenAI-compatible API on NVIDIA GPU.

## Phase 4: Verification
- [x] Validate Docker setup by running a `docker-compose config` check.
- [x] Validate Python files for syntax correctness.
