"""
FastAPI application that coordinates a multi-agent document analysis workflow.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agents.entity_extractor import EntityExtractorAgent
from .agents.sentiment import SentimentAgent
from .agents.summarizer import SummarizerAgent
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResults,
    ResultsResponse,
    UploadResponse,
)
from .services.orchestrator import AnalysisOrchestrator
from .utils.document_parser import UnsupportedDocumentError, extract_text

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("multi-agent")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multi-Agent Document Analysis System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

jobs_store: Dict[str, Dict[str, Any]] = {}
jobs_lock = asyncio.Lock()

def _build_llm() -> Optional[Any]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.info("GROQ_API_KEY not set. Falling back to heuristic agents.")
        return None
    
    # Strip whitespace and quotes that might have been accidentally added
    groq_api_key = groq_api_key.strip().strip('"').strip("'")
    if not groq_api_key:
        logger.info("GROQ_API_KEY is empty after stripping. Falling back to heuristic agents.")
        return None
    
    try:
        from langchain_groq import ChatGroq

        # langchain-groq accepts 'api_key' or 'groq_api_key' - try api_key first
        try:
            return ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                timeout=45,
            )
        except TypeError:
            # Fallback to groq_api_key if api_key doesn't work
            return ChatGroq(
                groq_api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                timeout=45,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unable to initialize Groq client: %s", exc, exc_info=True)
        return None


llm_client = _build_llm()
summarizer_agent = SummarizerAgent(llm_client=llm_client)
entity_agent = EntityExtractorAgent(llm_client=llm_client)
sentiment_agent = SentimentAgent(llm_client=llm_client)
orchestrator = AnalysisOrchestrator(summarizer_agent, entity_agent, sentiment_agent)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    _validate_file_type(file)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File exceeds 10MB limit.",
        )

    job_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or _guess_extension(file.content_type)
    stored_path = UPLOAD_DIR / f"{job_id}{suffix}"
    stored_path.write_bytes(contents)

    async with jobs_lock:
        jobs_store[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "filepath": stored_path,
            "status": "uploaded",
            "results": {},
            "analysis_started_at": None,
            "analysis_finished_at": None,
            "total_processing_time_seconds": None,
            "agents_completed": 0,
            "agents_failed": 0,
            "warning": None,
        }

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="uploaded",
        message="Document uploaded successfully. Use /analyze to start processing.",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> AnalyzeResponse:
    async with jobs_lock:
        job = jobs_store.get(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id.")
        if job["status"] == "processing":
            raise HTTPException(status_code=409, detail="Analysis already in progress.")
        if job["status"] in {"completed", "partial"}:
            raise HTTPException(status_code=409, detail="Analysis already completed.")
        job["status"] = "processing"
        job["analysis_started_at"] = datetime.now(timezone.utc)
        job["results"] = {}
        job["agents_completed"] = 0
        job["agents_failed"] = 0
        job["warning"] = None
        job["total_processing_time_seconds"] = None

    background_tasks.add_task(_launch_analysis, request.job_id)
    return AnalyzeResponse(
        job_id=request.job_id,
        status="processing",
        message="Analysis started. Check /results/{job_id} for updates.",
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
async def fetch_results(job_id: str) -> ResultsResponse:
    async with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id.")

        results = job.get("results") or {}
        response = ResultsResponse(
            job_id=job_id,
            status=job["status"],
            document_name=job["filename"],
            results=AnalysisResults(**results) if results else AnalysisResults(),
            total_processing_time_seconds=job.get("total_processing_time_seconds"),
            agents_completed=job.get("agents_completed", 0),
            agents_failed=job.get("agents_failed", 0),
            warning=job.get("warning"),
        )

    return response


def _validate_file_type(file: UploadFile) -> None:
    allowed = {"application/pdf", "text/plain"}
    if (file.content_type not in allowed) and not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")


def _guess_extension(content_type: str | None) -> str:
    if content_type == "application/pdf":
        return ".pdf"
    return ".txt"


def _launch_analysis(job_id: str) -> None:
    asyncio.run(_process_job(job_id))


async def _process_job(job_id: str) -> None:
    try:
        async with jobs_lock:
            job = jobs_store[job_id]
    except KeyError:
        logger.warning("Received analysis request for unknown job: %s", job_id)
        return

    try:
        document_text = extract_text(job["filepath"])
    except (UnsupportedDocumentError, OSError) as exc:
        await _mark_job_failed(job_id, f"Document parsing failed: {exc}")
        return

    try:
        structured_results, agent_outcomes = await orchestrator.run(document_text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Agent orchestration failed for %s", job_id)
        await _mark_job_failed(job_id, f"Analysis failed: {exc}")
        return
    agents_completed = sum(1 for outcome in agent_outcomes.values() if outcome.success)
    agents_failed = len(agent_outcomes) - agents_completed
    status = "completed" if agents_failed == 0 else "partial"
    warning = None
    if status == "partial":
        warning = "Partial results available - one or more agents failed."

    total_processing_time = max(
        (outcome.processing_time_seconds for outcome in agent_outcomes.values()),
        default=0.0,
    )

    async with jobs_lock:
        job = jobs_store[job_id]
        job["status"] = status
        job["results"] = structured_results
        job["analysis_finished_at"] = datetime.now(timezone.utc)
        job["total_processing_time_seconds"] = round(total_processing_time, 3)
        job["agents_completed"] = agents_completed
        job["agents_failed"] = agents_failed
        job["warning"] = warning


async def _mark_job_failed(job_id: str, message: str) -> None:
    async with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            return
        job["status"] = "failed"
        job["results"] = {
            "summary": {"error": message, "processing_time_seconds": 0.0},
            "entities": {"error": message, "processing_time_seconds": 0.0},
            "sentiment": {"error": message, "processing_time_seconds": 0.0},
        }
        job["warning"] = message
        job["agents_completed"] = 0
        job["agents_failed"] = 3
        job["total_processing_time_seconds"] = 0.0
        job["analysis_finished_at"] = datetime.now(timezone.utc)

