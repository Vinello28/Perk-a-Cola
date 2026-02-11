# User Guide: Perk-a-Cola Classification System

This guide provides detailed instructions on how to set up, configure, and run the Perk-a-Cola text classification system.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Classification](#running-the-classification)
5. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10** or higher.
- **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/).
- **A PC with sufficient RAM/VRAM** to run a 4B parameter model (approx. 4GB VRAM recommended).

---

## 2. Setting Up LM Studio

The system relies on a local Large Language Model (LLM) running via LM Studio.

1. **Download a Model**: Open LM Studio, search for `qwen3-4b-instruct`, and download a quantized version (e.g., `Q4_K_M` or `Q5_K_M`).
2. **Load the Model**: Go to the "Local Server" tab (diagram icon on the left) and select the downloaded model from the top dropdown.
3. **Configure Server**:
   - Ensure the port is set to `1234` (default).
   - Enable "Cross-Origin-Resource-Sharing (CORS)" if necessary (usually on by default).
4. **Start Server**: Click the green **"Start Server"** button.

> **Verify**: You should see a log message saying "Server listening on http://localhost:1234".

---

## 3. Installation

1. **Clone the Repository** (if you haven't already):
   ```bash
   git clone https://github.com/Vinello28/Perk-a-Cola.git
   cd Perk-a-Cola
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Configuration

The system is controlled via `app/src/config.yaml`. You can modify this file to change labels, prompts, or performance settings.

### Key Settings

- **`llm.model_name`**: The ID of the model in LM Studio.
- **`llm.enable_thinking`**: Set to `true` to let the model "reason" before answering (better accuracy, slower). Set to `false` for faster responses.
- **`classification.labels`**: A list of valid categories (e.g., `["ai", "non_ai"]`).
- **`concurrency.max_workers`**: How many descriptions to classify at once.
  - **Increase** if you have a powerful GPU.
  - **Decrease** (e.g., to 2-4) if LM Studio becomes unstable or slow.

---

## 5. Running the Classification

1. **Prepare Data**:
   - Save your Excel file (e.g., `data.xlsx`) in the `app/data/` folder.
   - Ensure it has a column named **"Descrizione"** (or whatever you configured in `config.yaml`).

2. **Run the Script**:
   ```bash
   python app/src/main.py
   ```

3. **Monitor Progress**:
   - The terminal will show a progress bar.
   - Example output:
     ```text
     Processing: data.xlsx
     Classifying: 100%|██████████| 1500/1500 [02:30<00:00, 10.00 desc/s]
     Results written to '.../app/out/data_classified.xlsx'
     ```

4. **Check Results**:
   - Open `app/out/`.
   - You will find a new file (e.g., `data_classified.xlsx`) containing the original descriptions and a new "Label" column.

---

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| **ConnectionRefusedError** | Ensure LM Studio server is running and the port is correct (1234). |
| **Timeout / Slow Speed** | Reduce `max_workers` in `config.yaml` or disable `enable_thinking`. |
| **"Column not found"** | Check your Excel header. It must match `description_column` in `config.yaml` exactly. |
| **Permission Denied** | Ensure the Excel file is not open in another program (like Excel) while the script is running. |
