# Discharge Summary POC

A proof-of-concept pipeline for processing hospital discharge summary PDFs using Google Gemini LLM.

## Pipeline Flow

```
PDF drop → pre-validate → Vision LLM text extraction → Reasoning LLM JSON extraction
→ structure validation → hallucination check → store → API call
```

## Architecture

| POC Component            | Simulates (Production)              |
|--------------------------|-------------------------------------|
| Local `input/` folder    | Azure Blob Storage input container  |
| Local `processed/` folder| Azure Blob Storage processed container |
| Local `failed/` folder   | Azure Blob Storage failed container |
| SQLite database          | Azure Cosmos DB audit logging       |
| Watchdog file watcher    | Azure Event Grid BlobCreated events |
| FastAPI mock server      | Client database replication API     |

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd discharge_summary_poc
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Configure environment variables

```bash
copy .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running

### Start the Mock API Server

```bash
uvicorn mock_api.server:app --port 8000
```

### Start the File Watcher

```bash
python -m src.watcher
```

The watcher monitors the `input/` folder. Drop a PDF there to trigger the pipeline.

### Run the Demo

```bash
python demo/run_demo.py
```

This runs all 3 test scenarios (happy path, validation failure, hallucination failure).

## Project Structure

```
discharge_summary_poc/
├── input/                      # Drop PDFs here to trigger pipeline
├── processed/                  # Successfully processed PDFs moved here
├── failed/                     # Failed PDFs moved here
├── text_dumps/                 # Raw text output from Vision LLM
├── database/                   # SQLite database file
├── logs/                       # Pipeline run logs
├── ontology/                   # YAML ontology file
│   └── discharge_ontology.yaml
├── src/
│   ├── config.py               # All constants and configuration
│   ├── db.py                   # SQLite database layer
│   ├── validator.py            # Pre-validation and quality checks
│   ├── gemini_client.py        # Gemini API wrapper
│   ├── vision_extractor.py     # Vision LLM text extraction
│   ├── ontology_extractor.py   # Reasoning LLM JSON extraction
│   ├── validation_layer1.py    # Structure validation
│   ├── validation_layer2.py    # Hallucination check
│   ├── pipeline.py             # Main orchestrator
│   └── watcher.py              # Watchdog file watcher
├── mock_api/
│   └── server.py               # FastAPI mock client API
├── demo/
│   └── run_demo.py             # Demo script for test scenarios
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All settings are loaded from `.env` via `src/config.py`:

| Variable              | Default   | Description                        |
|-----------------------|-----------|------------------------------------|
| `GEMINI_API_KEY`      | —         | Google AI Studio API key           |
| `ACCEPTANCE_THRESHOLD`| `0.6`     | Minimum confidence score to accept |
| `CONCURRENCY_LIMIT`   | `3`       | Max concurrent pipeline runs       |
| `LOG_LEVEL`           | `INFO`    | Logging verbosity                  |
| `MOCK_API_PORT`       | `8000`    | Port for the mock API server       |
