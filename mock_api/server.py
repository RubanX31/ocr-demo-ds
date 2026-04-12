"""
FastAPI mock server simulating the client's database replication API.

This server receives extracted discharge summary data from the pipeline
and simulates storing it in the client's database. It includes a
configurable simulated failure rate (5%) to test error handling.

Usage:
    python mock_api/server.py
    # or
    uvicorn mock_api.server:app --port 8000
"""

import json
import random
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve config path (mock_api is a sibling of src, not inside it)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MOCK_API_PORT, LOGS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Lifespan event handler (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Print startup banner, then yield to the app."""
    print()
    print("=" * 60)
    print("  MOCK CLIENT REPLICATION API")
    print("=" * 60)
    print(f"  Port       : {MOCK_API_PORT}")
    print(f"  Endpoints  : POST /replicate")
    print(f"               GET  /replications")
    print(f"               GET  /health")
    print(f"  Log file   : {LOGS_DIR / 'replications.jsonl'}")
    print(f"  Failure rate: 5% (simulated)")
    print("=" * 60)
    print()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mock Client Replication API",
    description="Simulates the client's database replication endpoint for the Discharge Summary POC.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Replications log file (JSONL — one JSON object per line)
# ---------------------------------------------------------------------------
REPLICATIONS_LOG = LOGS_DIR / "replications.jsonl"


# ============================================================================
# Request / Response models
# ============================================================================

class ReplicationRequest(BaseModel):
    """Payload sent by the pipeline after successful processing."""
    patientId: str
    uniqueDocNum: str
    serviceDate: str
    summaryText: str
    jsonObject: dict


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/replicate")
def replicate(request: ReplicationRequest):
    """
    Receive processed discharge summary data and simulate database storage.

    Simulates a 5 % failure rate to test the pipeline's error handling.
    On success the payload is appended to ``logs/replications.jsonl``.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # --- Log to console ---
    print()
    print("=" * 60)
    print("  REPLICATION REQUEST RECEIVED")
    print("=" * 60)
    print(f"  Patient ID   : {request.patientId}")
    print(f"  Doc Number   : {request.uniqueDocNum}")
    print(f"  Service Date : {request.serviceDate}")
    print(f"  Summary      : {request.summaryText[:120]}...")
    print(f"  JSON fields  : {len(request.jsonObject)} top-level keys")
    print(f"  Received at  : {now}")
    print("=" * 60)

    # --- Simulate 5 % failure rate ---
    if random.random() < 0.05:
        print("  ** SIMULATED DATABASE ERROR **")
        print("=" * 60)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Simulated database error",
            },
        )

    # --- Persist to JSONL log ---
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "patientId": request.patientId,
        "uniqueDocNum": request.uniqueDocNum,
        "serviceDate": request.serviceDate,
        "summaryText": request.summaryText,
        "jsonObject": request.jsonObject,
        "receivedAt": now,
    }
    with open(REPLICATIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Saved to {REPLICATIONS_LOG}")
    print("  Status: SUCCESS")
    print("=" * 60)

    return {
        "status": "success",
        "message": "Data replicated successfully",
        "patientId": request.patientId,
        "receivedAt": now,
    }


@app.get("/replications")
def get_replications():
    """
    Return all replicated entries from the JSONL log file.

    Useful for inspecting what data the pipeline has sent during a demo.
    """
    if not REPLICATIONS_LOG.exists():
        return []

    entries = []
    with open(REPLICATIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return entries


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "mock_client_api",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }





# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MOCK_API_PORT)
