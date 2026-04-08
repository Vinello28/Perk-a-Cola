---
description: "Scaffold a new LLM classifier implementing BaseClassifier."
---

# Scaffold New Classifier

Please generate a new text classifier implementation that integrates into the Perk-a-Cola pipeline.

## Requirements

1. **Inheritance:** Ensure the new classifier inherits from `BaseClassifier` (located in `app/src/classifier.py`).
2. **Typing:** Use `from __future__ import annotations` and provide strict type hints for all parameters and return types.
3. **Async Processing:** The core classification method (e.g., `classify` or `process_text`) must be an `async` function. 
4. **Concurrency:** Wrap the network/API request in the injected `asyncio.Semaphore` block to prevent VRAM overflow.
5. **Configuration Integration:** The `__init__` method should accept relevant frozen dataclasses from `app/src/config.py` (like `LLMConfig` and `ClassificationConfig`).
6. **Robust Parsing:** Implement logic to parse the text output securely using standard regex to extract the target label, accounting for potential `<think>...</think>` artifacts from reasoning models.
7. **Logging:** Use the standard Python `logging` module to report errors and raw output mapping failures.

## Expected Boilerplate Example

```python
from __future__ import annotations
import asyncio
import logging
import re

from app.src.classifier import BaseClassifier
from app.src.config import LLMConfig, ClassificationConfig

logger = logging.getLogger(__name__)

class CustomAPIClassifier(BaseClassifier):
    def __init__(self, llm_config: LLMConfig, class_config: ClassificationConfig, max_workers: int):
        self.llm_config = llm_config
        self.class_config = class_config
        self.semaphore = asyncio.Semaphore(max_workers)
        # compile parsing regex ...

    async def classify_item(self, text: str) -> str:
        async with self.semaphore:
            try:
                # Add async network/LLM call here
                pass
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                return self.class_config.default_label
```

Ensure the generated code accurately matches the conventions of the rest of the workspace.