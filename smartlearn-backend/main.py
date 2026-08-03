import os
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services import rag

app = FastAPI(title="SmartLearn Lite API")

# CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [
    origin.strip()
    for origin in allowed_origins_env.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
async def health():
    return {"ok": True}


documents: dict = {}


class ChatRequest(BaseModel):
    chat_id: str = Field(default="day2-demo", min_length=1)
    message: str = Field(..., min_length=2, max_length=2000)


@app.post("/upload")
async def upload_pdf(
    chat_id: str = Query(..., description="Chat session ID"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    # Reject non-PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read into bytes
    pdf_bytes = await file.read()

    # Reject empty file
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Build RAG-ready document record
    try:
        document = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            upload_root="uploads",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # Reject PDF with zero readable text
    total_chars = sum(len(p["text"]) for p in document["pages"])
    if total_chars == 0:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in this PDF. OCR is not supported.",
        )

    # Store in memory
    documents[chat_id] = document

    # Return Day 2-compatible response
    return rag.build_upload_response(document)


@app.get("/documents/{chat_id}/file")
async def get_document_file(chat_id: str):
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No document found for chat_id '{chat_id}'")

    file_path = document.get("saved_pdf_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Saved PDF file not found")

    return FileResponse(file_path, media_type="application/pdf")


@app.post("/chat")
async def chat(request: ChatRequest):
    # Look up stored document
    document = documents.get(request.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF uploaded for chat_id '{request.chat_id}'. "
                   f"Upload a PDF first via POST /upload?chat_id={request.chat_id}.",
        )

    # Day 2-style: feed all pages to LLM directly (no retrieval)
    pages = document["pages"]
    from services.llm import answer_from_pages

    try:
        answer = answer_from_pages(pages, request.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream AI service unavailable: {e}")

    # Extract distinct [Page X] citations
    existing = {p["page"] for p in pages}
    cited = {int(n) for n in re.findall(r"\[Page (\d+)\]", answer)}
    citations = sorted(cited & existing)

    return {"answer": answer, "citations": citations}
