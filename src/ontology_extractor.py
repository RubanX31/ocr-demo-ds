"""
Step 6 — Reasoning LLM JSON extraction.

Takes the raw text produced by the Vision LLM (Step 5) and extracts
structured fields into a JSON object that matches the discharge summary
ontology defined in ``ontology/discharge_ontology.yaml``.

The prompt instructs Gemini to extract *only* information explicitly present
in the raw text — no guessing, no inference, no hallucination.
"""

import json
import logging
from pathlib import Path

import yaml

from src.config import (
    ONTOLOGY_DIR,
    TEXT_DUMPS_DIR,
    REASONING_PROMPT_VERSION,
)
from src.gemini_client import gemini
from src import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level ontology cache (loaded once on first call)
# ---------------------------------------------------------------------------
_ontology_cache: dict | None = None

# Sections in the ontology that hold extractable fields (excludes metadata)
_ONTOLOGY_SECTIONS = [
    "patient_demographics",
    "admission_details",
    "clinical_summary",
    "diagnoses",
    "procedures",
    "medications",
    "investigations",
    "follow_up",
    "discharge_details",
]


# ============================================================================
# 1. Load ontology
# ============================================================================

def load_ontology() -> dict:
    """
    Load and parse the discharge summary ontology YAML file.

    The result is cached at module level so the file is read only once.

    Returns:
        Parsed ontology as a Python dict.
    """
    global _ontology_cache
    if _ontology_cache is not None:
        return _ontology_cache

    ontology_path = ONTOLOGY_DIR / "discharge_ontology.yaml"
    logger.info("Loading ontology from %s", ontology_path)

    with open(ontology_path, "r", encoding="utf-8") as f:
        _ontology_cache = yaml.safe_load(f)

    logger.info(
        "Ontology loaded — version=%s, sections=%d",
        _ontology_cache.get("metadata", {}).get("ontology_version", "unknown"),
        len([k for k in _ontology_cache if k != "metadata"]),
    )
    return _ontology_cache


# ============================================================================
# 2. Build extraction prompt
# ============================================================================

def build_extraction_prompt(raw_text: str, ontology: dict) -> str:
    """
    Build the Reasoning LLM prompt for structured JSON extraction.

    The prompt provides:
      - The raw text dump as the sole source of truth.
      - The full ontology schema (serialized as YAML) as the extraction template.
      - Strict instructions to prevent hallucination.

    Args:
        raw_text: Verbatim text extracted from the PDF by the Vision LLM.
        ontology: Parsed ontology dict (from :func:`load_ontology`).

    Returns:
        A single prompt string ready to send to Gemini.
    """
    # Serialize the ontology sections (without metadata) as YAML for the prompt
    schema_sections = {
        section: ontology[section]
        for section in _ONTOLOGY_SECTIONS
        if section in ontology
    }
    schema_yaml = yaml.dump(schema_sections, default_flow_style=False, sort_keys=False)

    prompt = f"""\
You are a medical data extraction engine. You are given:
1. RAW TEXT extracted from a hospital discharge summary (below).
2. An ONTOLOGY SCHEMA that defines the exact fields to extract (below).

Your task is to extract structured data from the RAW TEXT and return it as a \
single valid JSON object that follows the ontology schema exactly.

=== STRICT RULES ===

1. EXTRACT ONLY information that is EXPLICITLY STATED in the raw text. \
Do NOT guess, infer, assume, or fabricate any information.

2. If a field's value is NOT found in the raw text, set it to null. \
NEVER omit a field key — every field in the schema must appear in the output, \
even if its value is null.

3. For LIST fields (e.g., medications, diagnoses, procedures, lab_results, \
imaging_findings): return an empty list [] if no matching data is found in \
the raw text. If data IS found, return each item as an object with ALL \
sub-fields defined in the schema (set missing sub-fields to null).

4. Do NOT add any fields that are not defined in the ontology schema.

5. Return ONLY a single valid JSON object. Do NOT include any commentary, \
preamble, explanation, markdown formatting, or code fences. Output the raw \
JSON directly.

6. The JSON must be well-formed and parseable by a standard JSON parser.

7. Preserve exact values from the raw text — do not reformat dates, do not \
correct spelling, do not convert units.

=== ONTOLOGY SCHEMA ===

{schema_yaml}

=== RAW TEXT (source of truth) ===

{raw_text}

=== OUTPUT ===

Extract all fields from the schema above using ONLY the raw text as the \
source of truth. Return the result as a single JSON object now:
"""
    return prompt


# ============================================================================
# 3. Count filled fields
# ============================================================================

def count_filled_fields(extracted_json: dict, ontology: dict) -> dict:
    """
    Count how many ontology fields were filled vs. total expected.

    Counts only the top-level leaf fields within each section (e.g.,
    ``patient_demographics.patient_name``). List item sub-fields
    (e.g., ``medications.medications_on_discharge[].drug_name``) are NOT
    counted individually — the list field itself is counted as filled if
    the list is non-empty.

    Args:
        extracted_json: The JSON object returned by the Reasoning LLM.
        ontology:       The parsed ontology dict.

    Returns:
        dict with keys:
            - total (int): total countable fields in the ontology
            - filled (int): non-null, non-empty fields in the extracted JSON
            - empty_fields (list[str]): dot-path names of missing/empty fields
    """
    total = 0
    filled = 0
    empty_fields = []

    for section_name in _ONTOLOGY_SECTIONS:
        section_schema = ontology.get(section_name)
        if not section_schema or not isinstance(section_schema, dict):
            continue

        section_data = extracted_json.get(section_name, {})
        if not isinstance(section_data, dict):
            section_data = {}

        for field_name, field_def in section_schema.items():
            # Skip non-field entries (e.g., if somehow a stray key appears)
            if not isinstance(field_def, dict):
                continue

            total += 1
            field_path = f"{section_name}.{field_name}"

            value = section_data.get(field_name)

            if _is_filled(value):
                filled += 1
            else:
                empty_fields.append(field_path)

    return {
        "total": total,
        "filled": filled,
        "empty_fields": empty_fields,
    }


def _is_filled(value) -> bool:
    """
    Check if a field value is considered "filled" (non-null, non-empty).

    - None → not filled
    - Empty string "" → not filled
    - Empty list [] → not filled
    - Empty dict {} → not filled
    - Everything else → filled
    """
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


# ============================================================================
# 4. Main extraction function
# ============================================================================

def extract_ontology_fields(raw_text: str, audit_id: str) -> dict:
    """
    Extract structured fields from raw discharge summary text using the
    Reasoning LLM and the discharge ontology.

    Workflow:
        1. Load ontology schema.
        2. Build extraction prompt with raw text + ontology.
        3. Call Gemini via :meth:`gemini.send_prompt_with_text`.
        4. Parse the JSON response.
        5. Count filled vs. total fields.
        6. Save extracted JSON to ``text_dumps/{audit_id}_extracted.json``.
        7. Update audit log with extraction metrics.

    Args:
        raw_text: Verbatim text from the Vision LLM extraction.
        audit_id: The audit record ID to update in the database.

    Returns:
        dict with keys:
            - success (bool)
            - extracted_json (dict | None)
            - fields_total (int)
            - fields_filled (int)
            - empty_fields (list[str])
            - failure_reason (str | None)
    """
    logger.info("Starting ontology extraction — audit_id=%s", audit_id)

    # --- Load ontology ---
    ontology = load_ontology()
    ontology_version = ontology.get("metadata", {}).get("ontology_version", "unknown")

    # --- Build prompt ---
    prompt = build_extraction_prompt(raw_text, ontology)
    logger.info(
        "Extraction prompt built — audit_id=%s, prompt_length=%d chars",
        audit_id, len(prompt),
    )

    # --- Call Gemini ---
    llm_result = gemini.send_prompt_with_text(
        prompt=prompt,
        prompt_version=REASONING_PROMPT_VERSION,
    )

    if not llm_result["success"]:
        logger.error(
            "Reasoning LLM API call failed — audit_id=%s, error=%s",
            audit_id, llm_result["error"],
        )
        db.update_audit_record(
            audit_id,
            ontology_yaml_version=ontology_version,
            reasoning_model_used=llm_result["model_used"],
            reasoning_prompt_version=REASONING_PROMPT_VERSION,
            ontology_extraction_time_ms=llm_result["time_ms"],
            overall_pipeline_status="FAILED",
            failure_stage="OntologyExtraction",
        )
        return {
            "success": False,
            "extracted_json": None,
            "fields_total": 0,
            "fields_filled": 0,
            "empty_fields": [],
            "failure_reason": f"Gemini API error: {llm_result['error']}",
        }

    # --- Parse JSON response ---
    parsed = gemini.parse_json_response(llm_result["response_text"])

    if "error" in parsed and parsed["error"] == "parse_failed":
        logger.error(
            "JSON parse failed — audit_id=%s, raw response (first 500 chars): %s",
            audit_id, (llm_result["response_text"] or "")[:500],
        )
        db.update_audit_record(
            audit_id,
            ontology_yaml_version=ontology_version,
            reasoning_model_used=llm_result["model_used"],
            reasoning_prompt_version=REASONING_PROMPT_VERSION,
            ontology_extraction_time_ms=llm_result["time_ms"],
            overall_pipeline_status="FAILED",
            failure_stage="OntologyExtraction",
        )
        return {
            "success": False,
            "extracted_json": None,
            "fields_total": 0,
            "fields_filled": 0,
            "empty_fields": [],
            "failure_reason": "Failed to parse LLM response as JSON",
        }

    # --- Count filled fields ---
    field_counts = count_filled_fields(parsed, ontology)

    logger.info(
        "Ontology extraction complete — audit_id=%s, filled=%d/%d",
        audit_id, field_counts["filled"], field_counts["total"],
    )

    # --- Save extracted JSON ---
    TEXT_DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TEXT_DUMPS_DIR / f"{audit_id}_extracted.json"
    json_path.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved extracted JSON to %s", json_path)

    # --- Update audit log ---
    db.update_audit_record(
        audit_id,
        ontology_yaml_version=ontology_version,
        reasoning_model_used=llm_result["model_used"],
        structured_fields_total=field_counts["total"],
        structured_fields_filled=field_counts["filled"],
        reasoning_prompt_version=REASONING_PROMPT_VERSION,
        ontology_extraction_time_ms=llm_result["time_ms"],
    )

    return {
        "success": True,
        "extracted_json": parsed,
        "fields_total": field_counts["total"],
        "fields_filled": field_counts["filled"],
        "empty_fields": field_counts["empty_fields"],
        "failure_reason": None,
    }
