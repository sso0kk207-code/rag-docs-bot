"""Эмбеддинги. По умолчанию - хэширование признаков (feature hashing): работает офлайн, без ключей и загрузки моделей.

Это лексический «вектор»: он ловит общие слова и словосочетания, но не смысловые синонимы. Для настоящей семантики
(«отпуск» ~ «каникулы») реализуйте тот же интерфейс `Embedder` поверх API эмбеддингов или sentence-transformers и храните
векторы в pgvector: остальной код (`VectorIndex`, поиск, API) не меняется.
"""
import hashlib
from typing import Protocol

import numpy as np

from app.rag.text import tokens

DIM = 1024


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokens(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # униграммы и биграммы

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0  # знак по старшему биту уменьшает влияние коллизий
                out[row, h % self.dim] += sign
            out[row] = np.sign(out[row]) * np.sqrt(np.abs(out[row]))  # сублинейный вес: повтор слова даёт убывающий эффект
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)
