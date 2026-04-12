"""
Watchdog file watcher for Discharge Summary POC.

Simulates Azure Event Grid ``BlobCreated`` events by monitoring the local
``input/`` folder for new PDF files. When a PDF is detected, the pipeline
is triggered automatically.

Usage:
    python -m src.watcher

The watcher runs until interrupted with Ctrl+C.
"""

import logging
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.config import INPUT_DIR, LOGS_DIR, LOG_LEVEL
from src import db
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Logging setup — console + file
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _configure_logging() -> None:
    """
    Configure the root logger to write to both stdout and a log file.

    Log file: ``logs/pipeline.log``
    Format:   ``timestamp | level | message``
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "pipeline.log"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    # Avoid duplicate handlers on re-import
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ============================================================================
# 1. PDF event handler
# ============================================================================

class PDFHandler(FileSystemEventHandler):
    """
    Watchdog event handler that triggers the pipeline for each new PDF
    dropped into the watched directory.

    Each ``on_created`` call processes independently — watchdog dispatches
    events on separate threads, so concurrent drops are handled naturally.
    """

    def on_created(self, event):
        """
        Called when a new file is created in the watched directory.

        Filters for ``.pdf`` files only, waits briefly for the file to be
        fully written, and then kicks off the pipeline.
        """
        # Ignore directory creation events
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process PDF files (case-insensitive)
        if file_path.suffix.lower() != ".pdf":
            logger.debug("Ignoring non-PDF file: %s", file_path.name)
            return

        logger.info(
            "PDF detected: %s — waiting for file write to complete...",
            file_path.name,
        )

        # Brief delay to ensure the file is fully written to disk
        # (especially important for large PDFs copied over a network)
        time.sleep(0.5)

        # Verify the file still exists (it may have been moved/deleted)
        if not file_path.exists():
            logger.warning(
                "PDF no longer exists (may have been moved): %s",
                file_path.name,
            )
            return

        # Trigger the pipeline
        try:
            logger.info(
                "Triggering pipeline for: %s", file_path.name
            )
            result = run_pipeline(file_path)

            if result["overall_status"] == "COMPLETED":
                logger.info(
                    "Pipeline COMPLETED for %s — audit_id=%s, time=%dms",
                    file_path.name,
                    result["audit_id"],
                    result["processing_time_ms"],
                )
            else:
                logger.warning(
                    "Pipeline FAILED for %s — audit_id=%s, "
                    "failure_stage=%s, time=%dms",
                    file_path.name,
                    result["audit_id"],
                    result.get("failure_stage", "Unknown"),
                    result["processing_time_ms"],
                )

        except Exception as exc:
            logger.exception(
                "Unhandled exception processing %s: %s",
                file_path.name, exc,
            )


# ============================================================================
# 2. Watcher startup
# ============================================================================

def start_watcher(watch_dir: Path = None) -> None:
    """
    Start the file watcher on the specified directory.

    Monitors for new PDF files and triggers the pipeline for each one.
    Runs indefinitely until interrupted with Ctrl+C.

    Args:
        watch_dir: Directory to watch. Defaults to ``INPUT_DIR`` from config.
    """
    _configure_logging()

    if watch_dir is None:
        watch_dir = INPUT_DIR

    # Ensure the watch directory exists
    watch_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the database tables
    db.init_db()
    logger.info("Database initialised")

    # Set up watchdog observer
    handler = PDFHandler()
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)

    observer.start()

    print()
    print("=" * 60)
    print("  DISCHARGE SUMMARY PIPELINE — FILE WATCHER")
    print("=" * 60)
    print(f"  Watching : {watch_dir}")
    print(f"  Log file : {LOGS_DIR / 'pipeline.log'}")
    print(f"  Log level: {LOG_LEVEL}")
    print()
    print("  Drop a PDF into the watched folder to trigger processing.")
    print("  Press Ctrl+C to stop the watcher.")
    print("=" * 60)
    print()

    logger.info("File watcher started — monitoring %s", watch_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        logger.info("Shutdown requested — stopping file watcher...")
        observer.stop()

    observer.join()
    logger.info("File watcher stopped cleanly.")
    print("Watcher stopped. Goodbye.")


# ============================================================================
# 3. Module entry point
# ============================================================================

if __name__ == "__main__":
    start_watcher()
