"""
Discharge Summary Pipeline — POC Demo Script.

Runs three scenarios to demonstrate the end-to-end pipeline:
    1. Happy Path    — A valid PDF processes through all steps successfully.
    2. Pre-Val Fail  — An invalid file is rejected at the pre-validation gate.
    3. Threshold Fail — A valid PDF fails the structure validation due to a
                        raised acceptance threshold (simulated).

Prerequisites:
    - Place at least one real discharge summary PDF in ``demo/test_pdfs/``
    - Start the mock API server:  python mock_api/server.py
    - Create a ``.env`` file from ``.env.example`` and set your GEMINI_API_KEY

# Sample discharge summaries available at:
# - https://www.physionet.org/content/mimic-iv-note/2.2/ (MIMIC-IV clinical notes, free with registration)
# - Search "sample discharge summary PDF" on medical education sites
# - Medical university open datasets

Usage:
    python demo/run_demo.py
"""

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from src import config
from src import db
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_PDFS_DIR = _PROJECT_ROOT / "demo" / "test_pdfs"
INPUT_DIR = config.INPUT_DIR


# ============================================================================
# Helper: formatted pipeline result printer
# ============================================================================

def print_pipeline_result(result: dict, scenario_name: str) -> None:
    """
    Print the result of a pipeline run in a clear, formatted block.

    Args:
        result:        The dict returned by :func:`pipeline.run_pipeline`.
        scenario_name: Human-readable scenario title for the output header.
    """
    status = result.get("overall_status", "UNKNOWN")
    is_success = status == "COMPLETED"

    # Colour codes (ignored in terminals that don't support ANSI)
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    colour = GREEN if is_success else RED

    print()
    print(f"  {BOLD}Scenario  :{RESET} {scenario_name}")
    print(f"  {BOLD}Status    :{RESET} {colour}{status}{RESET}")
    print(f"  {BOLD}Audit ID  :{RESET} {result.get('audit_id', 'N/A')}")

    if result.get("failure_stage"):
        print(f"  {BOLD}Failed at :{RESET} {YELLOW}{result['failure_stage']}{RESET}")

    time_ms = result.get("processing_time_ms", 0)
    if time_ms > 1000:
        print(f"  {BOLD}Time      :{RESET} {time_ms / 1000:.1f}s ({time_ms}ms)")
    else:
        print(f"  {BOLD}Time      :{RESET} {time_ms}ms")


def print_audit_row(audit: dict) -> None:
    """Print a single audit record as a compact formatted block."""
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print(f"  {CYAN}audit_id{RESET}           : {audit.get('audit_id', 'N/A')[:36]}")
    print(f"  original_file     : {audit.get('original_file_name', 'N/A')}")
    print(f"  pipeline_status   : {audit.get('overall_pipeline_status', 'N/A')}")
    print(f"  failure_stage     : {audit.get('failure_stage') or '—'}")
    print(f"  pre_validation    : name={audit.get('file_name_validation_status') or '—'}"
          f"  size={audit.get('file_size_validation_status') or '—'}")
    print(f"  vision_extraction : {audit.get('raw_text_extraction_status') or '—'}"
          f"  quality={audit.get('raw_text_quality_score') or '—'}")
    print(f"  ontology_fields   : filled={audit.get('structured_fields_filled') or '—'}"
          f"/{audit.get('structured_fields_total') or '—'}")
    print(f"  structure_val     : {audit.get('structure_validation_status') or '—'}"
          f"  ratio={audit.get('filled_to_total_ratio') or '—'}")
    print(f"  hallucination     : {audit.get('hallucination_check_status') or '—'}")
    print(f"  replication       : {audit.get('replication_api_status') or '—'}")
    print(f"  created_at        : {audit.get('created_at', 'N/A')}")


def check_mock_api() -> bool:
    """Check if the mock API server is reachable."""
    try:
        r = requests.get(
            f"http://localhost:{config.MOCK_API_PORT}/health", timeout=3
        )
        return r.status_code == 200
    except Exception:
        return False


def find_test_pdf() -> Path | None:
    """Find the first PDF file in demo/test_pdfs/."""
    TEST_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    for f in TEST_PDFS_DIR.iterdir():
        if f.suffix.lower() == ".pdf" and f.stat().st_size > 0:
            return f
    return None


# ============================================================================
# Demo scenarios
# ============================================================================

def scenario_1_happy_path(test_pdf: Path) -> dict:
    """
    SCENARIO 1 — Happy Path (Success).

    Copies a valid PDF to the input folder with a conforming filename
    and runs the full pipeline. Expected outcome: COMPLETED.
    """
    target_name = "PAT001_20260101_DOC001.pdf"
    target = INPUT_DIR / target_name

    # Copy test PDF to input directory
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(test_pdf), str(target))

    print(f"  Copied {test_pdf.name} -> input/{target_name}")
    print(f"  Running pipeline...")

    result = run_pipeline(target)
    return result


def scenario_2_prevalidation_failure() -> dict:
    """
    SCENARIO 2 — Pre-Validation Failure.

    Creates a 0-byte file with an invalid filename (no underscores,
    not matching the expected pattern). Expected outcome: FAILED at
    PreValidation stage.
    """
    target_name = "INVALID_FILENAME.pdf"
    target = INPUT_DIR / target_name

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")  # 0-byte empty file

    print(f"  Created empty file: input/{target_name}")
    print(f"  Running pipeline...")

    result = run_pipeline(target)
    return result


def scenario_3_threshold_failure(test_pdf: Path) -> dict:
    """
    SCENARIO 3 — Structure Validation Failure (simulated).

    Copies a valid PDF but temporarily raises the acceptance threshold
    to 0.99 (near-impossible to pass). Expected outcome: FAILED at
    ValidationLayer1 stage.
    """
    target_name = "PAT003_20260103_DOC003.pdf"
    target = INPUT_DIR / target_name

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(test_pdf), str(target))

    # Save original threshold and patch to 0.99
    original_threshold = config.ACCEPTANCE_THRESHOLD
    config.ACCEPTANCE_THRESHOLD = 0.99
    print(f"  Copied {test_pdf.name} -> input/{target_name}")
    print(f"  Patched ACCEPTANCE_THRESHOLD: {original_threshold} -> 0.99")
    print(f"  Running pipeline...")

    try:
        result = run_pipeline(target)
    finally:
        # Always restore original threshold
        config.ACCEPTANCE_THRESHOLD = original_threshold
        print(f"  Restored ACCEPTANCE_THRESHOLD: {config.ACCEPTANCE_THRESHOLD}")

    return result


# ============================================================================
# Main demo runner
# ============================================================================

def run_demo() -> None:
    """Execute all three demo scenarios in sequence."""

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}   DISCHARGE SUMMARY PIPELINE — POC DEMO{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()
    print(f"  This demo will run {BOLD}3 scenarios{RESET}:")
    print()
    print(f"    {GREEN}1. Happy Path{RESET}         — Valid PDF, full pipeline success")
    print(f"    {YELLOW}2. Pre-Val Failure{RESET}    — Invalid file rejected at pre-validation")
    print(f"    {RED}3. Threshold Failure{RESET}  — Valid PDF fails structure validation")
    print()
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model     : {config.GEMINI_MODEL}")
    print(f"  Threshold : {config.ACCEPTANCE_THRESHOLD}")
    print(f"  API Port  : {config.MOCK_API_PORT}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    db.init_db()
    print(f"\n  {GREEN}Database initialised{RESET}")

    # Check mock API
    api_ok = check_mock_api()
    if api_ok:
        print(f"  {GREEN}Mock API is reachable{RESET} at http://localhost:{config.MOCK_API_PORT}")
    else:
        print(f"  {YELLOW}WARNING: Mock API is NOT reachable at port {config.MOCK_API_PORT}{RESET}")
        print(f"  {YELLOW}  Start it with: python mock_api/server.py{RESET}")
        print(f"  {YELLOW}  Pipeline will still run but API replication step will fail.{RESET}")

    # Check for test PDFs
    test_pdf = find_test_pdf()
    if test_pdf is None:
        print(f"\n  {RED}ERROR: No test PDF found in demo/test_pdfs/{RESET}")
        print(f"  {RED}Please place at least one discharge summary PDF there.{RESET}")
        print()
        print("  # Sample discharge summaries available at:")
        print("  # - https://www.physionet.org/content/mimic-iv-note/2.2/")
        print("  # - Search 'sample discharge summary PDF' on medical education sites")
        print("  # - Medical university open datasets")
        print()
        sys.exit(1)

    print(f"  Test PDF  : {test_pdf.name} ({test_pdf.stat().st_size / 1024:.1f} KB)")
    print()

    results = []

    # ------------------------------------------------------------------
    # SCENARIO 1 — Happy Path
    # ------------------------------------------------------------------
    print(f"{BOLD}{'─' * 70}{RESET}")
    print(f"  {BOLD}{GREEN}SCENARIO 1: Happy Path (Expected: COMPLETED){RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")

    try:
        r1 = scenario_1_happy_path(test_pdf)
        print_pipeline_result(r1, "Happy Path")
        results.append(("Happy Path", r1))
    except Exception as exc:
        print(f"\n  {RED}SCENARIO 1 CRASHED: {exc}{RESET}")
        results.append(("Happy Path", {"overall_status": "CRASHED", "audit_id": "N/A",
                                        "failure_stage": str(exc), "processing_time_ms": 0}))

    print()
    input("  Press Enter to continue to Scenario 2...")

    # ------------------------------------------------------------------
    # SCENARIO 2 — Pre-Validation Failure
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"  {BOLD}{YELLOW}SCENARIO 2: Pre-Validation Failure (Expected: FAILED){RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")

    try:
        r2 = scenario_2_prevalidation_failure()
        print_pipeline_result(r2, "Pre-Validation Failure")
        results.append(("Pre-Val Failure", r2))
    except Exception as exc:
        print(f"\n  {RED}SCENARIO 2 CRASHED: {exc}{RESET}")
        results.append(("Pre-Val Failure", {"overall_status": "CRASHED", "audit_id": "N/A",
                                             "failure_stage": str(exc), "processing_time_ms": 0}))

    print()
    input("  Press Enter to continue to Scenario 3...")

    # ------------------------------------------------------------------
    # SCENARIO 3 — Threshold Failure
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"  {BOLD}{RED}SCENARIO 3: Structure Validation Failure (Expected: FAILED){RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")

    try:
        r3 = scenario_3_threshold_failure(test_pdf)
        print_pipeline_result(r3, "Threshold Failure (0.99)")
        results.append(("Threshold Failure", r3))
    except Exception as exc:
        print(f"\n  {RED}SCENARIO 3 CRASHED: {exc}{RESET}")
        results.append(("Threshold Failure", {"overall_status": "CRASHED", "audit_id": "N/A",
                                               "failure_stage": str(exc), "processing_time_ms": 0}))

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}   DEMO SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    # Scenario results table
    print(f"\n  {'Scenario':<25} {'Status':<12} {'Failed At':<20} {'Time':<10}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 20} {'─' * 10}")
    for name, r in results:
        status = r.get("overall_status", "UNKNOWN")
        failed_at = r.get("failure_stage") or "—"
        time_s = f"{r.get('processing_time_ms', 0) / 1000:.1f}s"
        print(f"  {name:<25} {status:<12} {failed_at:<20} {time_s:<10}")

    # Pipeline summary from DB
    summary = db.get_pipeline_summary()
    print(f"\n  {BOLD}Pipeline Summary (from SQLite):{RESET}")
    print(f"    Total documents  : {summary['total']}")
    print(f"    Completed        : {GREEN}{summary['completed']}{RESET}")
    print(f"    Failed           : {RED}{summary['failed']}{RESET}")
    print(f"    In Progress      : {summary['in_progress']}")

    # Detailed audit logs
    print(f"\n  {BOLD}Audit Log Details:{RESET}")
    print(f"  {'─' * 66}")
    for name, r in results:
        audit_id = r.get("audit_id")
        if audit_id and audit_id != "N/A":
            audit = db.get_audit_record(audit_id)
            if audit:
                print(f"\n  [{name}]")
                print_audit_row(audit)
                print(f"  {'─' * 66}")

    # File locations
    db_path = config.DATABASE_DIR / "audit.db"
    repl_log = config.LOGS_DIR / "replications.jsonl"
    pipeline_log = config.LOGS_DIR / "pipeline.log"

    print(f"\n  {BOLD}Output Locations:{RESET}")
    print(f"    SQLite DB        : {db_path}")
    print(f"    Pipeline log     : {pipeline_log}")
    print(f"    Replications log : {repl_log}")
    print(f"    Text dumps       : {config.TEXT_DUMPS_DIR}")
    print(f"    Processed PDFs   : {config.PROCESSED_DIR}")
    print(f"    Failed PDFs      : {config.FAILED_DIR}")

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}   DEMO COMPLETE{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    run_demo()
