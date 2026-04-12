"""
Pre-validation and quality checks for discharge summary PDFs.

This module runs *before* any LLM calls. It validates filenames, file sizes,
and (post-extraction) raw text quality. No external dependencies — uses only
the Python standard library and pathlib.
"""

import re
from pathlib import Path

from src.config import PDF_NAMING_PATTERN, MIN_CHAR_TO_KB_RATIO, MAX_GIBBERISH_RATIO
from src import db


# ---------------------------------------------------------------------------
# Characters considered "standard" for gibberish detection.
# Alphanumeric + common punctuation + whitespace.
# ---------------------------------------------------------------------------
_STANDARD_CHARS_RE = re.compile(
    r"[a-zA-Z0-9"
    r"\s"                       # whitespace (space, tab, newline, etc.)
    r".,;:!?'\"\-\(\)\[\]{}"    # common punctuation & brackets
    r"@#\$%&\*/\\+=<>~`^_|"    # common symbols
    r"\u2013\u2014\u2018\u2019\u201C\u201D"  # en-dash, em-dash, smart quotes
    r"\u00B0\u00B1\u00D7\u00F7"              # degree, plus-minus, multiply, divide
    r"]"
)


# ============================================================================
# 1. Filename Validation
# ============================================================================

def validate_filename(filename: str) -> dict:
    """
    Validate that a PDF filename matches the expected naming convention.

    Expected pattern: ``<patientid>_<YYYYMMDD>_<uniquedocnum>.pdf``
    where each segment is alphanumeric (no spaces, no special characters).

    Args:
        filename: The PDF filename to validate (basename only, not full path).

    Returns:
        dict with keys:
            - valid (bool)
            - reason (str | None): human-readable failure reason, None if valid
            - patient_id (str): parsed patient ID, empty string if invalid
            - service_date (str): parsed date segment, empty string if invalid
            - unique_doc_num (str): parsed document number, empty string if invalid
    """
    result = {
        "valid": False,
        "reason": None,
        "patient_id": "",
        "service_date": "",
        "unique_doc_num": "",
    }

    if not filename:
        result["reason"] = "Filename is empty"
        return result

    if not filename.lower().endswith(".pdf"):
        result["reason"] = f"File is not a PDF: '{filename}'"
        return result

    if not re.match(PDF_NAMING_PATTERN, filename):
        result["reason"] = (
            f"Filename '{filename}' does not match the expected pattern "
            f"'<patientid>_<YYYYMMDD>_<uniquedocnum>.pdf'. "
            f"Each segment must be alphanumeric, separated by underscores."
        )
        return result

    # Pattern matched — safe to split exactly on underscores.
    # Strip '.pdf' suffix, then split into exactly 3 parts.
    stem = filename[:-4]  # remove '.pdf'
    parts = stem.split("_")

    if len(parts) != 3:
        result["reason"] = (
            f"Filename '{filename}' has {len(parts)} underscore-separated "
            f"segments; expected exactly 3 (patientid_date_docnum)."
        )
        return result

    patient_id, service_date, unique_doc_num = parts

    # Validate the date segment is a plausible YYYYMMDD
    if len(service_date) != 8 or not service_date.isdigit():
        result["reason"] = (
            f"Date segment '{service_date}' is not a valid YYYYMMDD date."
        )
        return result

    result["valid"] = True
    result["patient_id"] = patient_id
    result["service_date"] = service_date
    result["unique_doc_num"] = unique_doc_num
    return result


# ============================================================================
# 2. File Size Validation
# ============================================================================

def validate_file_size(filepath: Path) -> dict:
    """
    Validate that the file exists and has a size greater than 0 KB.

    Args:
        filepath: Absolute path to the PDF file.

    Returns:
        dict with keys:
            - valid (bool)
            - size_kb (float): file size in kilobytes (0.0 if file not found)
            - reason (str | None): human-readable failure reason, None if valid
    """
    result = {
        "valid": False,
        "size_kb": 0.0,
        "reason": None,
    }

    if not filepath.exists():
        result["reason"] = f"File not found: '{filepath}'"
        return result

    size_bytes = filepath.stat().st_size
    size_kb = round(size_bytes / 1024, 2)
    result["size_kb"] = size_kb

    if size_bytes == 0:
        result["reason"] = "File is empty (0 bytes)"
        return result

    result["valid"] = True
    return result


# ============================================================================
# 3. Text Quality Check (run after Vision LLM extraction)
# ============================================================================

def check_text_quality(raw_text: str, file_size_kb: float) -> dict:
    """
    Assess the quality of text extracted from a PDF by the Vision LLM.

    Performs two checks:
        1. **Character-to-KB ratio** — ensures the extracted text has enough
           characters relative to the PDF file size. A very low ratio indicates
           the Vision LLM may have failed to extract meaningful content.
        2. **Gibberish detection** — counts non-standard characters (anything
           that isn't alphanumeric, common punctuation, or whitespace). A high
           ratio indicates garbled or corrupted output.

    Args:
        raw_text:    The raw text string extracted by the Vision LLM.
        file_size_kb: The PDF file size in kilobytes.

    Returns:
        dict with keys:
            - passed (bool)
            - quality_score (float): 1.0 - gibberish_ratio, clamped [0.0, 1.0]
            - char_count (int)
            - char_to_kb_ratio (float)
            - gibberish_ratio (float)
            - failure_reason (str | None): None if passed
    """
    char_count = len(raw_text)
    failure_reasons = []

    # --- Check 1: Character-to-KB ratio ---
    if file_size_kb > 0:
        char_to_kb_ratio = round(char_count / file_size_kb, 2)
    else:
        char_to_kb_ratio = 0.0

    if char_to_kb_ratio < MIN_CHAR_TO_KB_RATIO:
        failure_reasons.append(
            f"Character-to-KB ratio ({char_to_kb_ratio}) is below the "
            f"minimum threshold ({MIN_CHAR_TO_KB_RATIO})"
        )

    # --- Check 2: Gibberish detection ---
    if char_count > 0:
        non_standard_count = sum(
            1 for ch in raw_text if not _STANDARD_CHARS_RE.match(ch)
        )
        gibberish_ratio = round(non_standard_count / char_count, 4)
    else:
        gibberish_ratio = 0.0

    if gibberish_ratio > MAX_GIBBERISH_RATIO:
        failure_reasons.append(
            f"Gibberish ratio ({gibberish_ratio:.2%}) exceeds the "
            f"maximum allowed ({MAX_GIBBERISH_RATIO:.0%})"
        )

    # --- Quality score ---
    quality_score = round(max(0.0, min(1.0, 1.0 - gibberish_ratio)), 4)

    passed = len(failure_reasons) == 0

    return {
        "passed": passed,
        "quality_score": quality_score,
        "char_count": char_count,
        "char_to_kb_ratio": char_to_kb_ratio,
        "gibberish_ratio": gibberish_ratio,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
    }


# ============================================================================
# 4. Pre-Validation Orchestrator
# ============================================================================

def run_pre_validation(filepath: Path, audit_id: str) -> dict:
    """
    Run filename and file-size validation in sequence and update the audit log.

    This is the main entry point called by the pipeline before any LLM work.
    It validates the filename pattern, checks the file size, and writes the
    results to the ``audit_log`` table via :func:`db.update_audit_record`.

    Args:
        filepath:  Absolute path to the PDF file.
        audit_id:  The audit record ID to update in the database.

    Returns:
        dict with keys:
            - passed (bool): True only if BOTH filename and size checks pass
            - patient_id (str)
            - service_date (str)
            - unique_doc_num (str)
            - file_size_kb (float)
    """
    filename = filepath.name

    # --- Step 1: Filename validation ---
    fn_result = validate_filename(filename)

    # --- Step 2: File size validation ---
    fs_result = validate_file_size(filepath)

    # --- Determine eligibility ---
    both_passed = fn_result["valid"] and fs_result["valid"]

    # Build a combined failure reason (may be None if both passed)
    reasons = []
    if not fn_result["valid"] and fn_result["reason"]:
        reasons.append(fn_result["reason"])
    if not fs_result["valid"] and fs_result["reason"]:
        reasons.append(fs_result["reason"])
    combined_reason = "; ".join(reasons) if reasons else None

    # --- Update audit log ---
    db.update_audit_record(
        audit_id,
        file_name_validation_status="PASSED" if fn_result["valid"] else "FAILED",
        file_size_validation_status="PASSED" if fs_result["valid"] else "FAILED",
        validation_failure_reason=combined_reason,
        is_eligible_for_processing=1 if both_passed else 0,
    )

    return {
        "passed": both_passed,
        "patient_id": fn_result["patient_id"],
        "service_date": fn_result["service_date"],
        "unique_doc_num": fn_result["unique_doc_num"],
        "file_size_kb": fs_result["size_kb"],
    }
