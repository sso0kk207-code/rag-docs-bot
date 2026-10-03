"""Режим без LLM: отвечает самыми подходящими предложениями из найденных источников (с теми же ссылками [n]).

Нужен для запуска без ключа API, для демонстрации и тестов. Это не языковая модель: связного пересказа он не делает.
"""
import re

from app.llm import LLMResult
from app.rag.text import tokens

SOURCE = re.compile(r'<source id="(\d+)"[^>]*>(.*?)</source>', re.S)
QUESTION = re.compile(r"<question>(.*?)</question>", re.S)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM:
    name = "extractive (без LLM)"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2048) -> LLMResult:
        prompt = messages[-1]["content"]
        question = QUESTION.search(prompt)
        q = set(tokens(question.group(1))) if question else set()
        scored = []
        for sid, body in SOURCE.findall(prompt):
            for sentence in SENTENCE.split(" ".join(body.split())):
                overlap = len(q & set(tokens(sentence)))
                if overlap:
                    scored.append((overlap, int(sid), sentence.strip()))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:3]
        text = " ".join(f"{s} [{sid}]" for _, sid, s in top) if top else "В найденных фрагментах нет прямого ответа."
        return LLMResult(text=text, model="extractive", stop_reason="end_turn")
