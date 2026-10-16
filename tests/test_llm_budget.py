from types import SimpleNamespace

import pytest

from reflex.llm_client import LLMCallBudgetExceeded, LLMClient


class _FakeCompletions:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.fail:
            raise RuntimeError("synthetic upstream failure")
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        )


class _FakeOpenAI:
    def __init__(self, *, fail: bool = False):
        self.chat = SimpleNamespace(completions=_FakeCompletions(fail=fail))


def test_hard_call_budget_and_role_accounting(monkeypatch):
    client = LLMClient(
        host="https://example.invalid/v1",
        key="test-only",
        model="fake",
        max_retries=1,
        call_budget=2,
    )
    fake = _FakeOpenAI()
    monkeypatch.setattr(client, "_client", lambda vision=False: fake)

    assert client.chat("diagnose", usage_role="critic") == "OK"
    assert client.chat("patch", usage_role="actor") == "OK"
    with pytest.raises(LLMCallBudgetExceeded):
        client.chat("over budget", usage_role="actor")

    stats = client.stats()
    assert stats["attempted"] == 2
    assert stats["calls"] == 2
    assert stats["failed"] == 0
    assert stats["attempts_by_role"] == {"critic": 1, "actor": 1}
    assert stats["by_role"] == {"critic": 1, "actor": 1}
    assert stats["remaining_calls"] == 0
    assert stats["prompt_tokens"] == 22
    assert stats["completion_tokens"] == 14
    trace = client.call_trace()
    assert [row["call_index"] for row in trace] == [1, 2]
    assert [row["role"] for row in trace] == ["critic", "actor"]
    assert all(row["status"] == "success" for row in trace)
    assert trace[0]["prompt_tokens"] == 11
    assert trace[0]["completion_tokens"] == 7


def test_failed_attempt_consumes_budget(monkeypatch):
    client = LLMClient(
        host="https://example.invalid/v1",
        key="test-only",
        model="fake",
        max_retries=1,
        retry_backoff=0,
        call_budget=1,
    )
    fake = _FakeOpenAI(fail=True)
    monkeypatch.setattr(client, "_client", lambda vision=False: fake)

    with pytest.raises(RuntimeError, match="synthetic upstream failure"):
        client.chat("fail", usage_role="repair")

    stats = client.stats()
    assert stats["attempted"] == 1
    assert stats["calls"] == 0
    assert stats["failed"] == 1
    assert stats["failures_by_role"] == {"repair": 1}
    assert stats["remaining_calls"] == 0
    trace = client.call_trace()
    assert trace[0]["status"] == "failure"
    assert "synthetic upstream failure" in trace[0]["error"]


def test_vllm_thinking_flag_is_forwarded_to_shared_vision_model(monkeypatch):
    client = LLMClient(
        host="https://example.invalid/v1",
        key="test-only",
        model="fake-text",
        vision_model="fake-vision",
        extra_body={"enable_thinking": False},
        max_retries=1,
    )
    fake = _FakeOpenAI()
    monkeypatch.setattr(client, "_client", lambda vision=False: fake)

    assert client.chat(
        "diagnose",
        image_b64s=["aW1hZ2U="],
        usage_role="critic",
    ) == "OK"
    sent = fake.chat.completions.last_kwargs["extra_body"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enable_thinking" not in sent
