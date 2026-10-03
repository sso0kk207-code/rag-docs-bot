"""Гибридный поиск: BM25 (точные слова: названия, числа, термины) + векторная близость, слияние рангов методом RRF."""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.rag.embed import Embedder
from app.rag.text import tokens

RRF_K = 60
CANDIDATES = 50


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    document: str
    page: int | None
    text: str
    cosine: float   # векторная близость: по ней решаем, есть ли в документах что-то по теме
    bm25: float
    rrf: float


class Index:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._data = None

    def invalidate(self) -> None:
        self._data = None

    def _load(self, db: Session) -> dict:
        if self._data is None:
            rows = db.execute(select(Chunk.id, Chunk.document_id, Chunk.page, Chunk.text, Chunk.vector, Document.filename).join(Document, Document.id == Chunk.document_id).order_by(Chunk.id)).all()
            toks = [Counter(tokens(r.text)) for r in rows]
            n = len(rows)
            df: Counter = Counter()
            for c in toks:
                df.update(c.keys())
            self._data = {
                "rows": rows,
                "matrix": np.vstack([np.frombuffer(r.vector, dtype=np.float32) for r in rows]) if rows else np.zeros((0, self.embedder.dim), dtype=np.float32),
                "tf": toks,
                "len": np.array([sum(c.values()) for c in toks], dtype=np.float32),
                "idf": {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()},
            }
        return self._data

    def search(self, db: Session, query: str, top_k: int = 5, document_ids: list[int] | None = None) -> list[Hit]:
        d = self._load(db)
        rows = d["rows"]
        if not rows:
            return []
        mask = np.ones(len(rows), dtype=bool) if not document_ids else np.array([r.document_id in set(document_ids) for r in rows])
        if not mask.any():
            return []

        cosine = d["matrix"] @ self.embedder.embed([query])[0]
        avg_len = d["len"].mean() or 1.0
        bm25 = np.zeros(len(rows), dtype=np.float32)
        for term in set(tokens(query)):
            idf = d["idf"].get(term)
            if idf is None:
                continue
            for i, tf_counter in enumerate(d["tf"]):
                f = tf_counter.get(term, 0)
                if f:
                    bm25[i] += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * d["len"][i] / avg_len))

        cosine_m, bm25_m = np.where(mask, cosine, -np.inf), np.where(mask, bm25, -np.inf)
        fused: dict[int, float] = {}
        for scores in (cosine_m, bm25_m):
            for rank, i in enumerate(np.argsort(-scores)[:CANDIDATES]):
                if mask[i] and (scores is cosine_m or bm25[i] > 0):  # нулевое BM25-совпадение рангов не получает
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [Hit(rows[i].id, rows[i].document_id, rows[i].filename, rows[i].page, rows[i].text, float(cosine[i]), float(bm25[i]), fused[i]) for i in best]
