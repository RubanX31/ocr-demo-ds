"""
Configuration module for Discharge Summary POC.
Loads environment variables and defines all folder paths and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Folder paths (derived from project root)
# ---------------------------------------------------------------------------
INPUT_DIR = PROJECT_ROOT / "input"
PROCESSED_DIR = PROJECT_ROOT / "processed"
FAILED_DIR = PROJECT_ROOT / "failed"
TEXT_DUMPS_DIR = PROJECT_ROOT / "text_dumps"
DATABASE_DIR = PROJECT_ROOT / "database"
LOGS_DIR = PROJECT_ROOT / "logs"
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"


# ---------------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MODEL_TEXT = "llama-3.3-70b-versatile"

# Prompt versions (static — no dynamic modification)
VISION_PROMPT_VERSION = "V1"
REASONING_PROMPT_VERSION = "V1"
VALIDATION_PROMPT_VERSION = "V1"

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
ACCEPTANCE_THRESHOLD = float(os.getenv("ACCEPTANCE_THRESHOLD", "0.6"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MOCK_API_PORT = int(os.getenv("MOCK_API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Pre-validation rules
# ---------------------------------------------------------------------------
# Expected PDF filename pattern: <alphanum>_<YYYYMMDD>_<alphanum>.pdf
PDF_NAMING_PATTERN = r'^[a-zA-Z0-9]+_\d{8}_[a-zA-Z0-9]+\.pdf$'

# Maximum allowed ratio of gibberish / non-printable characters in extracted text
MAX_GIBBERISH_RATIO = 0.30

# Minimum characters per KB of PDF file size (quality gate)
MIN_CHAR_TO_KB_RATIO = 50
