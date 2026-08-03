"""
RAG pipeline for SmartLearn AI - Day 3 Labs A/B/C.

Handles text cleaning, page loading, chunking, embeddings,
FAISS retrieval, answer generation, and evaluation.
"""

import os
import re
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies
# ---------------------------------------------------------------------------

_PdfReader = None
def _get_pdf_reader():
    global _PdfReader
    if _PdfReader is None:
        from pypdf import PdfReader
        _PdfReader = PdfReader
    return _PdfReader


_faiss_mod = None
def _get_faiss():
    global _faiss_mod
    if _faiss_mod is None:
        import faiss
        _faiss_mod = faiss
    return _faiss_mod


_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Text cleaning and page loading  (Lab A)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize extracted PDF text: remove null bytes, soft hyphens,
    repeated whitespace, and noisy line breaks."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_for_rag(file_path, page_limit: int | None = None) -> list[dict]:
    """Read a PDF file page by page, return [{page, text}] records."""
    PdfReader = _get_pdf_reader()
    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        if page_limit and i > page_limit:
            break
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})
    return pages


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Read PDF bytes (from upload), return [{page, text}] records."""
    PdfReader = _get_pdf_reader()
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})
    return pages


# ---------------------------------------------------------------------------
# JSON helpers  (Lab A)
# ---------------------------------------------------------------------------

def save_json(data: Any, path) -> None:
    """Save one Python object to a UTF-8 JSON file, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Any:
    """Read one saved JSON artifact back into Python."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def relative_path_str(path, base) -> str:
    """Return a shorter display path relative to *base*."""
    try:
        return str(Path(path).relative_to(base))
    except (ValueError, TypeError):
        return str(path)


# ---------------------------------------------------------------------------
# Chunking  (Lab A)
# ---------------------------------------------------------------------------

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split one oversized text block into smaller pieces,
    preferring natural boundaries (spaces) over mid-word splits."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # try to break at a space
            gap = 40
            search_end = min(end + gap, len(text))
            last_space = text.rfind(" ", end, search_end)
            if last_space != -1:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end if end > start else start + 1
    return pieces


def chunk_by_paragraph(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split by paragraph boundaries; split long paragraphs if needed."""
    chunks: list[dict] = []
    cid = 0
    for rec in pages:
        pg = rec["page"]
        paragraphs = [p.strip() for p in rec["text"].split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) > chunk_size:
                for piece in slice_long_text(para, chunk_size):
                    chunks.append({
                        "chunk_id": f"chunk-{cid:05d}",
                        "page": pg,
                        "text": piece,
                        "chunk_mode": "paragraph",
                    })
                    cid += 1
            else:
                chunks.append({
                    "chunk_id": f"chunk-{cid:05d}",
                    "page": pg,
                    "text": para,
                    "chunk_mode": "paragraph",
                })
                cid += 1
    return chunks


def chunk_by_characters(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Fixed-size sliding-window chunks with optional overlap."""
    chunks: list[dict] = []
    cid = 0
    step = chunk_size - overlap if overlap > 0 else chunk_size
    step = max(step, 1)
    for rec in pages:
        pg = rec["page"]
        text = rec["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                mode = "character_overlap" if overlap > 0 else "character"
                chunks.append({
                    "chunk_id": f"chunk-{cid:05d}",
                    "page": pg,
                    "text": piece,
                    "chunk_mode": mode,
                })
                cid += 1
            if end >= len(text):
                break
            start += step
    return chunks


def build_chunks(records: list[dict], chunk_mode: str = "character_overlap",
                 chunk_size: int = 700, overlap: int = 120) -> list[dict]:
    """Select the requested chunking strategy and return uniform chunk records."""
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size, overlap)
    elif chunk_mode in ("character", "character_overlap"):
        return chunk_by_characters(records, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown chunk_mode: {chunk_mode}")


# ---------------------------------------------------------------------------
# Embedding  (Lab A)
# ---------------------------------------------------------------------------

def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")


def get_device() -> str:
    """Choose CPU or CUDA for the current machine."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_model_source(model_name: str, artifact_root=None) -> str:
    """Prefer a local cached model folder when it already exists."""
    short_name = model_name.split("/")[-1]
    candidates = []
    if artifact_root:
        candidates.append(Path(artifact_root) / "hf_models" / short_name)
    candidates.append(Path("artifacts") / "rag" / "hf_models" / short_name)
    for c in candidates:
        if (c / "modules.json").exists():
            return str(c)
    return model_name  # fall back to HF download


def load_model(model_name: str, model_cache_dir=None, device: str | None = None):
    """Create or reuse one sentence-transformer model instance."""
    dev = device or get_device()
    source = str(model_cache_dir) if model_cache_dir else model_name
    key = (source, dev)
    if key not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[key] = SentenceTransformer(
            source,
            device=dev,
            model_kwargs={"use_safetensors": False},
        )
    return _model_cache[key]


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                model_cache_dir=None, batch_size: int = 32) -> np.ndarray:
    """Encode texts into normalized float32 vectors."""
    model = load_model(model_name, model_cache_dir)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Artifact management  (Lab A)
# ---------------------------------------------------------------------------

def ensure_artifact_dirs(artifact_root=None) -> dict:
    """Create and return all artifact folders."""
    root = Path(artifact_root) if artifact_root else Path("artifacts")
    dirs = {
        "root": root,
        "raw_pages": root / "raw_pages",
        "chunks": root / "chunks",
        "embeddings": root / "embeddings",
        "reports": root / "reports",
        "chroma": root / "chroma",
        "hf_models": root / "hf_models",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _config_dir_name(chunk_mode, chunk_size, overlap, model_name) -> str:
    return f"{chunk_mode}_c{chunk_size}_o{overlap}_{model_tag(model_name)}"


def artifact_paths_for(document_id, pdf_name, chunk_mode, model_name,
                       chunk_size, overlap, artifact_root=None) -> dict:
    """Decide where pages, chunks, embeddings, manifest, and index should be saved."""
    root = Path(artifact_root) if artifact_root else Path("artifacts")
    cfg = _config_dir_name(chunk_mode, chunk_size, overlap, model_name)
    base = root / document_id / cfg
    mt = model_tag(model_name)
    flat_pages = root / "raw_pages" / f"{document_id}_pages.json"
    flat_chunks = root / "chunks" / f"{document_id}_{chunk_mode}.json"
    flat_embed = root / "embeddings" / f"{document_id}_{chunk_mode}_{mt}.npy"
    flat_manifest = root / "embeddings" / f"{document_id}_{chunk_mode}_{mt}.manifest.json"
    return {
        "config_dir": base,
        "raw_pages": base / "raw_pages.json",
        "chunks": base / "chunks.json",
        "embeddings": base / "embeddings.npy",
        "manifest": base / "manifest.json",
        "index": base / "index.faiss",
        "flat_raw_pages": flat_pages,
        "flat_chunks": flat_chunks,
        "flat_embeddings": flat_embed,
        "flat_manifest": flat_manifest,
    }


def ensure_artifacts(document_id, pdf_name, pages, chunk_mode="character_overlap",
                     model_name="sentence-transformers/all-MiniLM-L6-v2",
                     chunk_size=700, overlap=120, batch_size=32,
                     artifact_root=None) -> dict:
    """Build or reuse the full pages -> chunks -> embeddings -> manifest bundle."""
    paths = artifact_paths_for(document_id, pdf_name, chunk_mode, model_name,
                               chunk_size, overlap, artifact_root)
    # Reuse if manifest exists and signature matches
    if paths["manifest"].exists():
        manifest = load_json(paths["manifest"])
        if (manifest.get("chunk_mode") == chunk_mode
                and manifest.get("chunk_size") == chunk_size
                and manifest.get("overlap") == overlap
                and manifest.get("model_name") == model_name):
            chunks = load_json(paths["chunks"])
            embeddings = np.load(paths["embeddings"])
            raw_pages = load_json(paths["raw_pages"])
            # also save flat copies for Lab B compatibility
            save_json(raw_pages, paths["flat_raw_pages"])
            save_json(chunks, paths["flat_chunks"])
            return {
                "manifest": manifest,
                "chunks": chunks,
                "embeddings": embeddings,
                "pages": raw_pages,
            }

    # Build from scratch
    ensure_artifact_dirs(artifact_root)
    chunks = build_chunks(pages, chunk_mode=chunk_mode,
                          chunk_size=chunk_size, overlap=overlap)
    chunk_texts = [c["text"] for c in chunks]
    if chunk_texts:
        embeddings = embed_texts(chunk_texts, model_name=model_name,
                                 batch_size=batch_size)
    else:
        embeddings = np.zeros((0, 384), dtype=np.float32)

    device = get_device()
    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 384,
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(pages, paths["raw_pages"])
    save_json(chunks, paths["chunks"])
    np.save(paths["embeddings"], embeddings)
    save_json(manifest, paths["manifest"])
    # flat copies for Lab B compatibility
    save_json(pages, paths["flat_raw_pages"])
    save_json(chunks, paths["flat_chunks"])
    np.save(paths["flat_embeddings"], embeddings)
    save_json(manifest, paths["flat_manifest"])

    return {
        "manifest": manifest,
        "chunks": chunks,
        "embeddings": embeddings,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# FAISS  (Lab B)
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """Build a searchable FAISS index from normalized embeddings."""
    faiss = _get_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path) -> None:
    """Write the binary .faiss file to disk."""
    faiss = _get_faiss()
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_faiss_index(index_path):
    """Load a saved .faiss index back into memory."""
    faiss = _get_faiss()
    return faiss.read_index(str(index_path))


# ---------------------------------------------------------------------------
# Document preparation  (Lab B)
# ---------------------------------------------------------------------------

def prepare_rag_document(document_id, filename, pages,
                         chunk_mode="character_overlap",
                         chunk_size=700, overlap=120,
                         model_name="sentence-transformers/all-MiniLM-L6-v2",
                         batch_size=32, artifact_root=None) -> dict:
    """Build a server-side document record with chunks, embeddings,
    FAISS index, and empty history."""
    bundle = ensure_artifacts(
        document_id, filename, pages,
        chunk_mode=chunk_mode, model_name=model_name,
        chunk_size=chunk_size, overlap=overlap,
        batch_size=batch_size, artifact_root=artifact_root,
    )
    embeddings = bundle["embeddings"]
    chunks = bundle["chunks"]

    # Build / save FAISS index
    paths = artifact_paths_for(document_id, filename, chunk_mode, model_name,
                               chunk_size, overlap, artifact_root)
    if embeddings.shape[0] > 0:
        index = build_faiss_index(embeddings)
        save_faiss_index(index, paths["index"])
    else:
        index = None

    model_source = resolve_model_source(model_name, artifact_root)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": chunks,
        "history": [],
        "artifacts": {
            "index": paths["index"],
            "chunks": paths["chunks"],
            "embeddings": paths["embeddings"],
            "raw_pages": paths["raw_pages"],
            "manifest": paths["manifest"],
        },
        "chunk_size": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 384,
        "model_name": model_name,
        "model_source": model_source,
        "index": index,
    }


# ---------------------------------------------------------------------------
# Retrieval  (Lab B)
# ---------------------------------------------------------------------------

def keyword_set(text: str) -> set:
    """Lightweight lexical tokens for simple reranking."""
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]{2,}", text)}


def search_bundle(question: str, bundle: dict, top_k: int = 3,
                  candidate_pool: int = 60, batch_size: int = 1,
                  history: list | None = None) -> list[dict]:
    """Retrieve top-k chunks from an in-memory index bundle."""
    chunks = bundle["chunks"]
    embeddings = bundle["embeddings"]
    index = bundle.get("index")
    if index is None:
        index = build_faiss_index(embeddings)

    q_vec = embed_texts([question], model_name=bundle["model_name"],
                        model_cache_dir=bundle.get("model_source"), batch_size=1)
    k = min(top_k, len(chunks))
    scores, indices = index.search(q_vec, k)

    hits = []
    q_keywords = keyword_set(question)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        overlap = len(q_keywords & keyword_set(chunk["text"]))
        hits.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score),
            "keyword_overlap": overlap,
            "rank": rank,
        })
    # Light lexical rerank: sort by score + keyword_overlap
    hits.sort(key=lambda h: (h["score"] + 0.01 * h["keyword_overlap"]), reverse=True)
    return hits[:top_k]


def search_document(question: str, document: dict, top_k: int = 3,
                    candidate_pool: int = 60,
                    history: list | None = None) -> list[dict]:
    """Retrieve top-k chunks from a prepared document record."""
    index = document.get("index")
    if index is None:
        index_path = document["artifacts"]["index"]
        if Path(index_path).exists():
            index = load_faiss_index(index_path)
            document["index"] = index
        else:
            # rebuild
            embeddings = np.load(document["artifacts"]["embeddings"])
            index = build_faiss_index(embeddings)
            document["index"] = index

    bundle = {
        "chunks": document["chunks"],
        "embeddings": np.load(document["artifacts"]["embeddings"]),
        "index": index,
        "model_name": document["model_name"],
        "model_source": document.get("model_source"),
    }
    return search_bundle(question, bundle, top_k=top_k,
                         candidate_pool=candidate_pool, history=history)


# ---------------------------------------------------------------------------
# Local answer extraction  (Lab B)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split retrieved chunk text into candidate answer sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return one short local answer sentence with a page tag."""
    if not hits:
        return "The document does not provide enough information."
    q_keywords = keyword_set(question)
    best_sent = ""
    best_score = -1
    best_page = hits[0]["page"]
    for hit in hits:
        for sent in split_sentences(hit["text"]):
            score = len(q_keywords & keyword_set(sent))
            if score > best_score:
                best_score = score
                best_sent = sent
                best_page = hit["page"]
    if not best_sent:
        best_sent = hits[0]["text"][:200]
    return f"{best_sent} [Page {best_page}]"


# ---------------------------------------------------------------------------
# Answer generation  (Lab B/C)
# ---------------------------------------------------------------------------

def extract_citations(answer: str, hits: list | None = None) -> list[int]:
    """Extract numeric PDF page citations from an answer string."""
    cited = {int(n) for n in re.findall(r"\[Page (\d+)\]", answer)}
    if hits:
        existing = {h["page"] for h in hits}
        cited &= existing
    return sorted(cited)


def build_sources(hits: list[dict]) -> list[dict]:
    """Build frontend-friendly source objects from retrieval hits."""
    return [
        {
            "chunk_id": h["chunk_id"],
            "page": h["page"],
            "score": round(h["score"], 4),
            "preview": h["text"][:150],
        }
        for h in hits
    ]


def _call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    """Call OpenRouter LLM and return the response text."""
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    actual_model = os.getenv("OPENROUTER_MODEL", model)
    resp = client.chat.completions.create(
        model=actual_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


_RAG_SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text chunks. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the chunks, say the document does not provide "
    "enough information. Never invent a page number."
)


def build_grounded_user_prompt(question: str, hits: list[dict],
                               history: list | None = None) -> str:
    """Build a grounded prompt string for LLM answer generation."""
    context_parts = [
        f"[Page {h['page']}] {h['text']}" for h in hits
    ]
    context = "\n\n".join(context_parts)
    prompt = f"PDF evidence:\n{context}\n\nQuestion: {question}"
    if history:
        history_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}" for t in history[-3:]
        )
        prompt = f"Previous turns:\n{history_text}\n\n{prompt}"
    return prompt


def answer_document(document: dict, question: str, top_k: int = 3,
                    candidate_pool: int = 60,
                    answer_model: str = "openrouter/free") -> dict:
    """Answer one question from a prepared document using retrieval."""
    hits = search_document(question, document, top_k=top_k,
                           candidate_pool=candidate_pool)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            user_prompt = build_grounded_user_prompt(question, hits)
            answer = _call_llm(_RAG_SYSTEM_PROMPT, user_prompt, answer_model)
        except Exception:
            answer = best_sentence_answer(question, hits)
    else:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)
    return {"answer": answer, "citations": citations, "sources": sources}


def append_history(document: dict, question: str, result: dict) -> list[dict]:
    """Append one completed turn to the document's in-memory history."""
    document.setdefault("history", []).append({
        "question": question,
        "answer": result["answer"],
        "citations": result.get("citations", []),
    })
    return document["history"]


def answer_document_turn(document: dict, question: str, top_k: int = 3,
                         candidate_pool: int = 60,
                         answer_model: str = "poolside/laguna-s-2.1:free") -> dict:
    """Answer one question and append to history. Returns result + updated history."""
    hits = search_document(question, document, top_k=top_k,
                           candidate_pool=candidate_pool,
                           history=document.get("history"))
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            user_prompt = build_grounded_user_prompt(
                question, hits, history=document.get("history"))
            answer = _call_llm(_RAG_SYSTEM_PROMPT, user_prompt, answer_model)
        except Exception:
            answer = best_sentence_answer(question, hits)
    else:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)
    result = {"answer": answer, "citations": citations, "sources": sources}
    result["history"] = append_history(document, question, result)
    return result


def answer_chat_turn(document: dict, message: str, top_k: int = 3,
                     candidate_pool: int = 60,
                     answer_model: str = "poolside/laguna-s-2.1:free") -> dict:
    """Answer a chat turn: fresh retrieval + history update."""
    return answer_document_turn(document, message, top_k=top_k,
                                candidate_pool=candidate_pool,
                                answer_model=answer_model)


# ---------------------------------------------------------------------------
# Lab C: upload record helpers
# ---------------------------------------------------------------------------

def prepare_rag_chat_record(chat_id: str, filename: str,
                            pdf_bytes: bytes | None = None,
                            pages: list | None = None,
                            upload_root=None,
                            chunk_mode="character_overlap",
                            chunk_size=700, overlap=120,
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            batch_size=32, artifact_root=None) -> dict:
    """Build the documents[chat_id] record after upload."""
    if pages is None and pdf_bytes is not None:
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)
    if pages is None:
        raise ValueError("Either pdf_bytes or pages must be provided")

    # Save uploaded PDF to disk
    saved_pdf_path = None
    if pdf_bytes is not None:
        up_dir = Path(upload_root) if upload_root else Path("uploads")
        up_dir.mkdir(parents=True, exist_ok=True)
        saved_pdf_path = up_dir / f"{chat_id}.pdf"
        saved_pdf_path.write_bytes(pdf_bytes)

    doc = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )
    doc["chat_id"] = chat_id
    doc["saved_pdf_path"] = str(saved_pdf_path) if saved_pdf_path else None
    doc["rag"] = {
        "document_id": chat_id,
        "index_path": str(doc["artifacts"]["index"]),
        "model_name": model_name,
    }
    return doc


def build_upload_response(document: dict) -> dict:
    """Build the Day 2-compatible upload success JSON."""
    total_chars = sum(len(p["text"]) for p in document["pages"])
    return {
        "status": "ok",
        "filename": document["filename"],
        "pages": len(document["pages"]),
        "characters": total_chars,
    }


# ---------------------------------------------------------------------------
# Evaluation  (Lab B)
# ---------------------------------------------------------------------------

def normalize_for_match(text: str) -> str:
    """Normalize text for simple string-based scoring."""
    return re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().strip()


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Check whether any acceptable answer appears in *text*."""
    norm = normalize_for_match(text)
    return any(normalize_for_match(a) in norm for a in answers)


def evaluate_questions(eval_set: list[dict], documents_by_name: dict,
                       top_k: int = 3, candidate_pool: int = 60):
    """Run a small retrieval + answer evaluation, return a DataFrame."""
    import pandas as pd
    rows = []
    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        answers = item["answers"]
        doc = documents_by_name.get(pdf_name)
        if doc is None:
            rows.append({
                "pdf_name": pdf_name,
                "question": question,
                "answers": answers,
                "retrieved_pages": [],
                "local_answer": "document not found",
                "retrieval_hit": False,
                "answer_hit": False,
            })
            continue

        hits = search_document(question, doc, top_k=top_k,
                               candidate_pool=candidate_pool)
        retrieved_pages = sorted({h["page"] for h in hits})
        local_answer = best_sentence_answer(question, hits)

        # retrieval_hit: any gold answer string appears in retrieved chunks
        chunk_text = " ".join(h["text"] for h in hits)
        retrieval_hit = contains_any_answer(chunk_text, answers)
        # answer_hit: any gold answer appears in the local answer
        answer_hit = contains_any_answer(local_answer, answers)

        rows.append({
            "pdf_name": pdf_name,
            "question": question,
            "answers": answers,
            "retrieved_pages": retrieved_pages,
            "local_answer": local_answer[:200],
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
        })
    return pd.DataFrame(rows)
