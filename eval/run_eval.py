"""Оценка качества поиска на размеченных вопросах: python -m eval.run_eval

Метрики (только для вопросов, ответ на которые есть в документах):
  hit@k  - доля вопросов, где нужный документ входит в k первых результатов
  hit@k+ - то же, но в k первых результатах есть фрагмент с ожидаемой фразой (поиск нашёл именно ответ, а не просто нужный документ)
  MRR    - средняя величина 1/ранг первого правильного фрагмента
Для вопросов без ответа в документах: доля вопросов, которые система корректно отсекла порогом релевантности.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite://")

from sqlalchemy.orm import Session  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import MIN_RELEVANCE  # noqa: E402
from app.models import Chunk, Document  # noqa: E402
from app.rag.embed import HashingEmbedder  # noqa: E402
from app.rag.ingest import ingest  # noqa: E402
from app.rag.retrieve import Index  # noqa: E402

ROOT = Path(__file__).parent.parent


def load_corpus(db: Session, embedder: HashingEmbedder) -> None:
    import hashlib

    import numpy as np

    for path in sorted((ROOT / "sample_docs").glob("*.md")):
        data = path.read_bytes()
        chunks, pages = ingest(path.name, data)
        doc = Document(filename=path.name, sha256=hashlib.sha256(data).hexdigest(), pages=pages, chunks_count=len(chunks))
        db.add(doc)
        db.flush()
        vectors = embedder.embed([c.text for c in chunks])
        db.add_all([Chunk(document_id=doc.id, idx=i, page=c.page, text=c.text, vector=np.asarray(vectors[i], dtype=np.float32).tobytes()) for i, c in enumerate(chunks)])
    db.commit()


def evaluate(ks=(1, 3, 5)) -> dict:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    embedder = HashingEmbedder()
    index = Index(embedder)
    cases = [json.loads(line) for line in (ROOT / "eval" / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with Session(engine) as db:
        load_corpus(db, embedder)
        answerable = [c for c in cases if c["doc"]]
        hit_doc = {k: 0 for k in ks}
        hit_text = {k: 0 for k in ks}
        rr, misses = 0.0, []
        for c in answerable:
            hits = index.search(db, c["q"], top_k=max(ks))
            for k in ks:
                top = hits[:k]
                hit_doc[k] += any(h.document == c["doc"] for h in top)
                hit_text[k] += any(h.document == c["doc"] and c["must"].lower() in h.text.lower() for h in top)
            rank = next((i for i, h in enumerate(hits, 1) if h.document == c["doc"] and c["must"].lower() in h.text.lower()), None)
            rr += 1 / rank if rank else 0
            if rank is None or rank > 3:
                misses.append((c["q"], rank))
        unanswerable = [c for c in cases if not c["doc"]]
        rejected = 0
        for c in unanswerable:
            hits = index.search(db, c["q"], top_k=4)
            rejected += (not hits) or max(h.cosine for h in hits) < MIN_RELEVANCE
        n = len(answerable)
    return {
        "questions": n,
        **{f"hit_doc@{k}": round(hit_doc[k] / n, 3) for k in ks},
        **{f"hit_answer@{k}": round(hit_text[k] / n, 3) for k in ks},
        "mrr": round(rr / n, 3),
        "unanswerable": len(unanswerable), "correctly_rejected": rejected, "min_relevance": MIN_RELEVANCE, "misses": misses,
    }


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0)
