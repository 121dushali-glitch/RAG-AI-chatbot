"""
RAG AI Chatbot — Backend (FastAPI + CPU-only)
Supports: PDF, DOCX, XLSX, CSV, PPTX, TXT, Website URLs
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List

# Ensure local backend modules are importable when running from inside or outside the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Load settings from .env into the process environment BEFORE importing
# rag, since it reads os.getenv(...) at import time to set its module-level config.
from dotenv import load_dotenv
load_dotenv()

from backend.document_processor import DocumentProcessor
from backend.rag_engine import RAGEngine

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Serve frontend
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

processor = DocumentProcessor()
rag_sessions: dict[str, RAGEngine] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_or_create_session(session_id: str) -> RAGEngine:
    """Retrieves an existing session from memory or loads it from disk if available."""
    if session_id not in rag_sessions:
        rag_sessions[session_id] = RAGEngine(str(VECTOR_DIR / session_id))
    return rag_sessions[session_id]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "RAG Chatbot API is running"}


@app.post("/api/session/new")
async def new_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload and process a document into the vector store."""
    allowed_extensions = {
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".csv", ".pptx", ".ppt", ".txt", ".md"
    }
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_extensions)}"
        )

    max_size = 20 * 1024 * 1024  # 20 MB
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit.")

    # Save file
    save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
    save_path.write_bytes(contents)

    # Extract text on a background thread to prevent event loop blocking
    try:
        chunks = await asyncio.to_thread(
            processor.process_file, str(save_path), file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

    # Index into FAISS in a background thread (heavy embedding calculations)
    engine = get_or_create_session(session_id)
    await asyncio.to_thread(engine.add_documents, chunks, source=file.filename)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": len(chunks),
    }


@app.post("/api/url")
async def ingest_url(payload: dict):
    """Fetch a website URL and index its text content."""
    url = payload.get("url", "").strip()
    session_id = payload.get("session_id", "").strip()
    if not url or not session_id:
        raise HTTPException(status_code=400, detail="url and session_id are required.")

    # Fetch webpage on a background thread (prevents blocking during network wait)
    try:
        chunks = await asyncio.to_thread(processor.process_url, url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not fetch URL: {e}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No readable text found at that URL.")

    # Index into FAISS in a background thread
    engine = get_or_create_session(session_id)
    await asyncio.to_thread(engine.add_documents, chunks, source=url)

    return {
        "status": "success",
        "url": url,
        "chunks_indexed": len(chunks),
    }


@app.get("/api/chat/stream")
async def chat_stream(session_id: str, question: str, history: str = "[]"):
    """Stream an answer via Server-Sent Events."""
    if not session_id or not question:
        raise HTTPException(status_code=400, detail="session_id and question required.")

    # Modified: Using get_or_create_session() enables persistence across server restarts
    engine = get_or_create_session(session_id)
    if not engine.has_documents():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed for this session. Please upload a file or URL first."
        )

    try:
        chat_history = json.loads(history)
    except Exception:
        chat_history = []

    async def event_generator():
        try:
            async for token in engine.stream_answer(question, chat_history):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sources")
async def get_sources(session_id: str, question: str):
    """Return the top source chunks used for a question."""
    # Modified: Use get_or_create_session() to read index from disk if the server restarted
    engine = get_or_create_session(session_id)
    if not engine.has_documents():
        return {"sources": []}
        
    sources = engine.retrieve_sources(question)
    return {"sources": sources}


@app.post("/api/reset")
async def reset_session(payload: dict):
    """Clear all documents for a session."""
    session_id = (payload.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    if session_id in rag_sessions:
        rag_sessions[session_id].reset()
        del rag_sessions[session_id]

    import shutil
    # Resolve and confirm the target is actually a subdirectory of VECTOR_DIR
    # before deleting anything — guards against session_id values (e.g. "..")
    session_vec = (VECTOR_DIR / session_id).resolve()
    vector_dir_resolved = VECTOR_DIR.resolve()
    if session_vec != vector_dir_resolved and vector_dir_resolved in session_vec.parents and session_vec.exists():
        shutil.rmtree(session_vec)

    return {"status": "reset"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}