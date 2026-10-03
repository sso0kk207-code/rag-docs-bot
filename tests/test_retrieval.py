import json
from pathlib import Path

import pytest

from app.rag.embed import HashingEmbedder
from app.rag.text import stem, tokens
from eval.run_eval import evaluate

CASES = [json.loads(line) for line in (Path(__file__).parent.parent / "eval" / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
ANSWERABLE = [c for c in CASES if c["doc"]]


def test_russian_word_forms_share_a_stem():
    assert len({stem("отпуск"), stem("отпуска"), stem("отпуске"), stem("отпуском")}) == 1
    assert "и" not in tokens("отпуск и больничный") and "отпус" in tokens("в отпуске")


def test_embeddings_are_normalized_and_deterministic():
    e = HashingEmbedder()
    v = e.embed(["Сколько дней отпуска", "Сколько дней отпуска"])
    assert abs(float((v[0] ** 2).sum()) - 1) < 1e-5 and (v[0] == v[1]).all()
    assert float(e.embed([""])[0].sum()) == 0  # пустой текст не ломает нормализацию


def test_similar_texts_are_closer_than_unrelated_ones():
    e = HashingEmbedder()
    a, b, c = e.embed(["сколько дней отпуска положено", "ежегодный отпуск 28 дней", "лимит на гостиницу в командировке"])
    assert float(a @ b) > float(a @ c)


@pytest.mark.parametrize("case", ANSWERABLE, ids=[c["q"][:40] for c in ANSWERABLE])
def test_right_fragment_is_in_top_3(client, corpus, case):
    hits = client.post("/search", json={"query": case["q"], "top_k": 3}).json()
    assert any(h["document"] == case["doc"] and case["must"].lower() in h["text"].lower() for h in hits), [h["document"] for h in hits]


def test_hybrid_search_finds_exact_numbers_and_terms(client, corpus):
    # точное совпадение редкого токена - сильная сторона BM25
    assert client.post("/search", json={"query": "security@company.example"}).json()[0]["document"] == "it_security.md"
    assert client.post("/search", json={"query": "60 000 рублей"}).json()[0]["document"] == "onboarding.md"


def test_document_filter_limits_search_scope(client, corpus):
    only_it = [corpus["it_security.md"]]
    hits = client.post("/search", json={"query": "сколько дней отпуска", "document_ids": only_it, "top_k": 5}).json()
    assert hits and all(h["document"] == "it_security.md" for h in hits)
    assert client.post("/search", json={"query": "отпуск", "document_ids": [999]}).json() == []


def test_search_on_empty_index(client):
    assert client.post("/search", json={"query": "что угодно"}).json() == []


def test_search_scores_are_exposed_for_debugging(client, corpus):
    top = client.post("/search", json={"query": "длина пароля"}).json()[0]
    assert {"cosine", "bm25", "rrf", "text", "document"} <= top.keys() and top["bm25"] > 0


def test_retrieval_quality_does_not_regress():
    """Порог качества на размеченном наборе: если кто-то испортит нарезку или токенизацию, тест упадёт."""
    m = evaluate()
    assert m["hit_doc@3"] >= 0.95 and m["hit_answer@3"] >= 0.9 and m["mrr"] >= 0.85
    assert m["correctly_rejected"] == m["unanswerable"]
