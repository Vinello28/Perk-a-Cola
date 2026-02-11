# Perk-a-Cola Classification 🥤

![Copertina](public/rdm1.png)

> Automated, privacy-first text classification pipeline powered by local LLMs via LM Studio.

![Python Version](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)
![LM Studio](https://img.shields.io/badge/LM_Studio-yellow?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSJjdXJyZW50Q29sb3IiIGZpbGwtcnVsZT0iZXZlbm9kZCIgaGVpZ2h0PSIxZW0iIHN0eWxlPSJmbGV4Om5vbmU7bGluZS1oZWlnaHQ6MSIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMWVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjx0aXRsZT5MTSBTdHVkaW88L3RpdGxlPjxwYXRoIGQ9Ik0yLjg0IDJhMS4yNzMgMS4yNzMgMCAxMDAgMi41NDdoMTQuMTA3YTEuMjczIDEuMjczIDAgMTAwLTIuNTQ3SDIuODR6TTcuOTM1IDUuMzNhMS4yNzMgMS4yNzMgMCAwMDAgMi41NDhIMjIuMDRhMS4yNzQgMS4yNzQgMCAwMDAtMi41NDdINy45MzV6TTMuNjI0IDkuOTM1YzAtLjcwNC41Ny0xLjI3NCAxLjI3NC0xLjI3NGgxNC4xMDZhMS4yNzQgMS4yNzQgMCAwMTAgMi41NDdINC44OThjLS43MDMgMC0xLjI3NC0uNTctMS4yNzQtMS4yNzN6TTEuMjczIDEyLjE4OGExLjI3MyAxLjI3MyAwIDEwMCAyLjU0N0gxNS4zOGExLjI3NCAxLjI3NCAwIDAwMC0yLjU0N0gxLjI3M3pNMy42MjQgMTYuNzkyYzAtLjcwNC41Ny0xLjI3NCAxLjI3NC0xLjI3NGgxNC4xMDZhMS4yNzMgMS4yNzMgMCAxMTAgMi41NDdINC44OThjLS43MDMgMC0xLjI3NC0uNTctMS4yNzQtMS4yNzN6TTEzLjAyOSAxOC44NDlhMS4yNzMgMS4yNzMgMCAxMDAgMi41NDdoOS42OThhMS4yNzMgMS4yNzMgMCAxMDAtMi41NDdoLTkuNjk4eiIgZmlsbC1vcGFjaXR5PSIuMyI+PC9wYXRoPjxwYXRoIGQ9Ik0yLjg0IDJhMS4yNzMgMS4yNzMgMCAxMDAgMi41NDdoMTAuMjg3YTEuMjc0IDEuMjc0IDAgMDAwLTIuNTQ3SDIuODR6TTcuOTM1IDUuMzNhMS4yNzMgMS4yNzMgMCAwMDAgMi41NDhIMTguMjJhMS4yNzQgMS4yNzQgMCAwMDAtMi41NDdINy45MzV6TTMuNjI0IDkuOTM1YzAtLjcwNC41Ny0xLjI3NCAxLjI3NC0xLjI3NGgxMC4yODZhMS4yNzMgMS4yNzMgMCAwMTAgMi41NDdINC44OThjLS43MDMgMC0xLjI3NC0uNTctMS4yNzQtMS4yNzN6TTEuMjczIDEyLjE4OGExLjI3MyAxLjI3MyAwIDEwMCAyLjU0N0gxMS41NmExLjI3NCAxLjI3NCAwIDAwMC0yLjU0N0gxLjI3M3pNMy42MjQgMTYuNzkyYzAtLjcwNC41Ny0xLjI3NCAxLjI3NC0xLjI3NGgxMC4yODZhMS4yNzMgMS4yNzMgMCAxMTAgMi41NDdINC44OThjLS43MDMgMC0xLjI3NC0uNTctMS4yNzQtMS4yNzN6TTEzLjAyOSAxOC44NDlhMS4yNzMgMS4yNzMgMCAxMDAgMi41NDdoNS43OGExLjI3MyAxLjI3MyAwIDEwMC0yLjU0N2gtNS43OHoiPjwvcGF0aD48L3N2Zz4=&logo-color=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

## Overview

Perk-a-Cola is a modular, high-performance classification system designed to process large Excel datasets using local Large Language Models. Built with a focus on **privacy**, **modularity**, and **software engineering best practices**, it allows you to classify text descriptions (e.g., identifying AI-related content) without sending data to external cloud providers.

## Key Features

- 🔒 **Privacy First**: Runs entirely local using LM Studio (tested with Qwen3-4B).
- ⚡ **High Performance**: Asynchronous concurrency handles 10,000+ rows efficiently.
- 🧩 **Modular Architecture**: 
  - **Strategy Pattern** for interchangeable classifiers.
  - **Config-driven** design (YAML) for easy customization of labels, prompts, and models.
- 🧠 **Smart Parsing**: Robust regex-based output parsing with support for "thinking" models.
- 📊 **Excel Integration**: Native read/write support for `.xlsx` files.

## Architecture

The system follows a clean separation of concerns:

```
src/
├── config.py        # Typed configuration loader
├── data_reader.py   # Specialized Excel reader
├── classifier.py    # Async LLM classifier (Strategy implementation)
├── output_writer.py # Result writer
└── main.py          # Pipeline orchestrator
```

## Prerequisites

- **Python 3.10+**
- **LM Studio** (or any OpenAI-compatible local server) running on `localhost:1234`
- A model loaded in LM Studio (recommended: `qwen3-4b-instruct`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vinello28/Perk-a-Cola.git
   cd Perk-a-Cola
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Customize the classification behavior in `app/src/config.yaml`:

```yaml
llm:
  model_name: "qwen3-4b"
  enable_thinking: true  # Utilize model's reasoning capabilities

classification:
  labels: ["ai", "non_ai"]
  description_column: "Descrizione"

concurrency:
  max_workers: 10  # Adjust based on your VRAM/System specs
```

## Usage

1. **Start LM Studio**: Load your model and start the server on port `1234`.
2. **Prepare Data**: Place your `.xlsx` files in `app/data/`. Ensure they have the configured target column (default: "Descrizione").
3. **Run the Pipeline**:

   ```bash
   python app/src/main.py
   ```

   You can also specify a custom config file:
   ```bash
   python app/src/main.py --config path/to/custom_config.yaml
   ```

4. **View Results**: Classified files will be generated in `app/out/` with the suffix `_classified.xlsx`.

## License

Distribued under the MIT License. See [LICENSE](LICENSE) for more information.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Vinello28">Vinello28</a>
</p>
