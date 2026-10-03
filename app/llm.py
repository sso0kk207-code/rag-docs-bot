"""Интерфейс к языковой модели. Остальной код зависит только от `LLM.complete()`, поэтому провайдера можно заменить адаптером."""
import os
from dataclasses import dataclass
from typing import Protocol

import anthropic


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    refused: bool = False


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2048) -> LLMResult: ...


DEFAULT_MODEL = "claude-opus-5-5"
# Запасная модель: если классификатор безопасности отклонит запрос, API перезапустит его на ней в рамках того же вызова
DEFAULT_FALLBACK_MODEL = "claude-opus-4-8"


class AnthropicLLM:
    """Claude через официальный SDK.

    - thinking не передаём: на claude-opus-5-5 рассуждение включено всегда, глубиной управляет effort;
    - temperature / top_p / budget_tokens на этой модели не поддерживаются (400), поэтому их нет;
    - отказ по политике безопасности приходит как stop_reason="refusal" (HTTP 200) и обрабатывается явно.
    """

    def __init__(self, model: str | None = None, client: anthropic.Anthropic | None = None, effort: str | None = None, fallback: bool | None = None):
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.getenv("LLM_EFFORT", "low")  # для коротких ответов в чате достаточно low
        self.fallback = (os.getenv("LLM_REFUSAL_FALLBACK", "1") == "1") if fallback is None else fallback
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        self.client = client or anthropic.Anthropic()  # ключ из ANTHROPIC_API_KEY или профиля `ant auth login`
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2048) -> LLMResult:
        params = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages, output_config={"effort": self.effort})
        try:
            if self.fallback:
                response = self.client.beta.messages.create(betas=["server-side-fallback-2026-06-01"], fallbacks=[{"model": self.fallback_model}], **params)
            else:
                response = self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited: {exc.message}", retryable=True) from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Invalid or missing Anthropic credentials", retryable=False) from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Bad request: {exc.message}", retryable=False) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}", retryable=exc.status_code >= 500) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Network error while calling the LLM", retryable=True) from exc

        usage = getattr(response, "usage", None)
        base = dict(model=response.model, input_tokens=getattr(usage, "input_tokens", 0), output_tokens=getattr(usage, "output_tokens", 0), stop_reason=response.stop_reason)
        if response.stop_reason == "refusal":
            return LLMResult(text="", refused=True, **base)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResult(text=text, **base)


def llm_from_env() -> LLM:
    """LLM_PROVIDER: anthropic | extractive | auto (anthropic, если задан ANTHROPIC_API_KEY, иначе extractive)."""
    provider = os.getenv("LLM_PROVIDER", "auto")
    if provider == "anthropic" or (provider == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        return AnthropicLLM()
    from app.rag.extractive import ExtractiveLLM

    return ExtractiveLLM()
