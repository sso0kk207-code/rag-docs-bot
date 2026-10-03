"""Адаптер Claude проверяется на фейковом клиенте: смотрим, какой именно запрос уходит в SDK."""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.llm import AnthropicLLM, LLMError


def response(text="Ответ [1]", stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(model=model, stop_reason=stop_reason, content=[SimpleNamespace(type="thinking", text="hidden"), SimpleNamespace(type="text", text=text)], usage=SimpleNamespace(input_tokens=300, output_tokens=42))


class FakeClient:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result, self.error = result or response(), error
        self.messages = SimpleNamespace(create=self._create("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create("beta")))

    def _create(self, kind):
        def create(**kwargs):
            self.calls.append((kind, kwargs))
            if self.error:
                raise self.error
            return self.result

        return create


def llm(client, **kw):
    return AnthropicLLM(client=client, **kw)


def test_request_shape_for_claude_opus_5_5():
    client = FakeClient()
    llm(client, fallback=False).complete("sys", [{"role": "user", "content": "q"}], max_tokens=500)
    kind, kw = client.calls[0]
    assert kind == "messages" and kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 500
    assert kw["system"] == "sys" and kw["messages"] == [{"role": "user", "content": "q"}]
    assert kw["output_config"] == {"effort": "low"}
    # параметры, которые эта модель отклоняет (400): их быть не должно
    assert not {"temperature", "top_p", "top_k", "thinking"} & kw.keys()


def test_refusal_fallback_is_enabled_by_default_and_can_be_disabled():
    client = FakeClient()
    llm(client).complete("s", [{"role": "user", "content": "q"}])
    kind, kw = client.calls[0]
    assert kind == "beta" and kw["betas"] == ["server-side-fallback-2026-06-01"] and kw["fallbacks"] == [{"model": "claude-opus-4-8"}]
    off = FakeClient()
    llm(off, fallback=False).complete("s", [{"role": "user", "content": "q"}])
    assert off.calls[0][0] == "messages" and "fallbacks" not in off.calls[0][1]


def test_only_text_blocks_are_returned_and_usage_is_reported():
    r = llm(FakeClient()).complete("s", [{"role": "user", "content": "q"}])
    assert r.text == "Ответ [1]" and (r.input_tokens, r.output_tokens) == (300, 42) and r.refused is False


def test_refusal_stop_reason_is_surfaced():
    r = llm(FakeClient(response(text="", stop_reason="refusal"))).complete("s", [{"role": "user", "content": "q"}])
    assert r.refused is True and r.text == ""


def test_effort_and_model_are_configurable(monkeypatch):
    monkeypatch.setenv("LLM_EFFORT", "high")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    client = FakeClient()
    llm(client).complete("s", [{"role": "user", "content": "q"}])
    kw = client.calls[0][1]
    assert kw["model"] == "claude-sonnet-5-5" and kw["output_config"]["effort"] == "high"


def api_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(status, request=req), body=None)


@pytest.mark.parametrize(
    "error, retryable",
    [
        (api_error(anthropic.RateLimitError, 429), True),
        (api_error(anthropic.InternalServerError, 500), True),
        (api_error(anthropic.AuthenticationError, 401), False),
        (api_error(anthropic.BadRequestError, 400), False),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), True),
    ],
)
def test_sdk_errors_are_mapped_with_retry_flag(error, retryable):
    with pytest.raises(LLMError) as exc:
        llm(FakeClient(error=error)).complete("s", [{"role": "user", "content": "q"}])
    assert exc.value.retryable is retryable


def test_provider_selection_from_environment(monkeypatch):
    from app.llm import llm_from_env

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    assert llm_from_env().name.startswith("extractive")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_from_env().name.startswith("extractive")  # без ключа сервис работает в режиме без LLM
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert llm_from_env().name == "anthropic:claude-opus-5-5"
