import hashlib
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import Base, engine, get_db
from app.llm import LLMError, llm_from_env
from app.models import Chunk, Document
from app.page import PAGE
from app.rag.answer import NO_ANSWER, answer_question
from app.rag.embed import HashingEmbedder
from app.rag.ingest import EmptyDocument, UnsupportedFormat, ingest
from app.rag.retrieve import Index

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.12"))  # ниже этой близости считаем, что в документах нет ответа, и не зовём модель


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    app.state.embedder = HashingEmbedder()
    app.state.index = Index(app.state.embedder)
    app.state.llm = llm_from_env()
    yield


app = FastAPI(
    title="RAG Docs Bot",
    version="1.0.0",
    description="Вопросы по загруженным документам: гибридный поиск (BM25 + векторы), ответ модели со ссылками на источники, отказ при отсутствии ответа.",
    lifespan=lifespan,
)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_ids: list[int] | None = None
    history: list[dict] = Field(default_factory=list, max_length=6, description="предыдущие реплики [{role, content}]")


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[int] | None = None


def doc_out(d: Document) -> dict:
    return {"id": d.id, "filename": d.filename, "pages": d.pages, "chunks": d.chunks_count, "created_at": d.created_at}


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok", "llm": app.state.llm.name}


@app.post("/documents", status_code=201, tags=["documents"], summary="Загрузить документ (txt, md, pdf, docx)")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    digest = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.sha256 == digest))
    if existing:
        raise HTTPException(409, f"This file is already uploaded (document id {existing.id})")
    try:
        chunks, pages = ingest(file.filename or "file", data)
    except UnsupportedFormat as exc:
        raise HTTPException(415, str(exc))
    except EmptyDocument as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        raise HTTPException(422, "The file is damaged or cannot be parsed")

    vectors = app.state.embedder.embed([c.text for c in chunks])
    doc = Document(filename=file.filename or "file", sha256=digest, pages=pages, chunks_count=len(chunks))
    db.add(doc)
    db.flush()
    db.add_all([Chunk(document_id=doc.id, idx=i, page=c.page, text=c.text, vector=np.asarray(vectors[i], dtype=np.float32).tobytes()) for i, c in enumerate(chunks)])
    db.commit()
    app.state.index.invalidate()
    return doc_out(doc)


@app.get("/documents", tags=["documents"])
def list_documents(db: Session = Depends(get_db)):
    return [doc_out(d) for d in db.scalars(select(Document).order_by(Document.id))]


@app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    db.delete(doc)
    db.commit()
    app.state.index.invalidate()


@app.post("/search", tags=["qa"], summary="Только поиск фрагментов (без модели): удобно для отладки качества")
def search(data: SearchIn, db: Session = Depends(get_db)):
    hits = app.state.index.search(db, data.query, data.top_k, data.document_ids)
    return [{"document": h.document, "page": h.page, "cosine": round(h.cosine, 3), "bm25": round(h.bm25, 2), "rrf": round(h.rrf, 4), "text": h.text} for h in hits]


@app.post("/ask", tags=["qa"], summary="Задать вопрос по документам")
def ask(data: AskIn, db: Session = Depends(get_db)):
    for turn in data.history:
        if turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
            raise HTTPException(422, "history items must look like {role: user|assistant, content: str}")
    hits = app.state.index.search(db, data.question, data.top_k, data.document_ids)
    if not hits or max(h.cosine for h in hits) < MIN_RELEVANCE:
        return {"answer": NO_ANSWER, "citations": [], "grounded": False, "refused": False, "model": None, "usage": {"input_tokens": 0, "output_tokens": 0}, "retrieved": len(hits)}
    try:
        result = answer_question(app.state.llm, data.question, hits, data.history)
    except LLMError as exc:
        raise HTTPException(503 if exc.retryable else 502, str(exc))
    return {
        "answer": result.answer, "citations": result.citations, "grounded": result.grounded, "refused": result.refused, "model": result.model,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}, "retrieved": len(hits),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index_page():
    return PAGE
