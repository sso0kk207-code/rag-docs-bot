"""Демо: python scripts/demo.py http://127.0.0.1:8000
Без ANTHROPIC_API_KEY сервис работает в режиме extractive: отвечает цитатами из найденных фрагментов (без LLM)."""
from pathlib import Path

from _demo import client, section, show

section("1. Загружаем учебные документы компании")
for path in sorted(Path("sample_docs").glob("*.md")):
    r = client.post("/documents", files={"file": (path.name, path.read_bytes(), "text/markdown")})
    print(f"   {path.name:<22} HTTP {r.status_code}  фрагментов: {r.json().get('chunks')}")
print()
r = client.post("/documents", files={"file": ("copy.md", Path("sample_docs/onboarding.md").read_bytes(), "text/markdown")})
print(f"   повторная загрузка того же файла -> HTTP {r.status_code}: {r.json()['detail']}")
print()

section("2. Поиск без модели: что нашлось и почему (BM25 + векторная близость, слияние RRF)")
hits = client.post("/search", json={"query": "сколько дней отпуска положено", "top_k": 3}).json()
print("$ curl -X POST /search -d '{\"query\": \"сколько дней отпуска положено\"}'")
for h in hits:
    print(f"   {h['document']:<20} cosine={h['cosine']:<6} bm25={h['bm25']:<6} rrf={h['rrf']}")
    print(f"      {h['text'][:90]}...")
print()

section("3. Вопрос по документам: ответ со ссылками на источники")
show("POST", "/ask", json_body={"question": "Какой лимит на гостиницу за ночь в командировке?"}, max_lines=24)

section("4. Вопрос, ответа на который в документах нет: модель не вызывается")
show("POST", "/ask", json_body={"question": "Какая зарплата у генерального директора?"}, max_lines=16)
