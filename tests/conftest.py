import os

os.environ.update({"DATABASE_URL": "sqlite://", "LLM_PROVIDER": "extractive"})

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Base, engine
from app.llm import LLMResult
from app.main import app

SAMPLES = Path(__file__).parent.parent / "sample_docs"


class FakeLLM:
    """Записывает вызовы и отвечает заданной функцией: тесты не зависят от сети и ключей."""

    name = "fake"

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or (lambda system, messages: "Ответ по источнику [1].")

    def complete(self, system, messages, max_tokens=2048):
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        out = self.reply(system, messages)
        return out if isinstance(out, LLMResult) else LLMResult(text=out, model="fake-model", input_tokens=120, output_tokens=15, stop_reason="end_turn")


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as c:
        c.llm = FakeLLM()
        app.state.llm = c.llm
        yield c


def upload(client, name: str, data: bytes, ctype: str = "text/plain"):
    return client.post("/documents", files={"file": (name, data, ctype)})


@pytest.fixture()
def corpus(client):
    """Все пять учебных документов загружены."""
    ids = {}
    for path in sorted(SAMPLES.glob("*.md")):
        r = upload(client, path.name, path.read_bytes())
        assert r.status_code == 201, r.text
        ids[path.name] = r.json()["id"]
    return ids
