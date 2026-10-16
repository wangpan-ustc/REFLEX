"""LLM client for OpenAI-compatible endpoints.

Routes multimodal requests (with images) to `vision_model` (if configured)
and text-only requests to `model`, enabling decoupled Critic/Actor model
selection across OpenAI-compatible APIs.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsage:
    attempted: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failed: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    by_role: dict[str, int] = field(default_factory=dict)
    attempts_by_role: dict[str, int] = field(default_factory=dict)
    failures_by_role: dict[str, int] = field(default_factory=dict)


class LLMCallBudgetExceeded(RuntimeError):
    """Raised before an outbound generation when the hard call cap is spent."""


@dataclass
class LLMClient:
    host: str                          # e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
    key: str = field(repr=False)
    model: str                         # default text model
    vision_model: str | None = None    # used when an image is present
    vision_host: str | None = None     # separate endpoint for vision_model (defaults to host)
    vision_key: str | None = field(default=None, repr=False)      # separate key for vision endpoint (defaults to key)
    timeout: float = 300.0
    temperature: float = 0.8
    max_tokens: int | None = None      # cap generation (incl. thinking tokens)
    max_retries: int = 2
    retry_backoff: float = 3.0
    call_budget: int | None = None     # successful + failed generation attempts
    trust_env: bool = True             # whether httpx inherits proxy env vars
    extra_body: dict[str, Any] = field(default_factory=dict)
    vision_extra_body: dict[str, Any] = field(default_factory=dict)
    # Keys in `extra_body` that are TEXT-MODEL-ONLY and should be stripped
    # before any call routed to `vision_model`. DashScope's qwen-vl-max
    # rejects `enable_thinking`, for example.
    text_only_extra_keys: tuple[str, ...] = (
        "enable_thinking",
        "thinking_budget",
    )
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    provider: str = "openai"           # OpenAI-compatible API
    usage: LLMUsage = field(default_factory=LLMUsage)
    debug: bool = False
    _text_openai_client: Any = field(default=None, init=False, repr=False)
    _vision_openai_client: Any = field(default=None, init=False, repr=False)
    _openai_client_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _usage_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _call_trace: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @staticmethod
    def _normalize_host(h: str) -> str:
        # Allow shorthand hosts like "api.deepseek.com" to become full URL.
        if not h.startswith("http"):
            h = "https://" + h
        if not h.rstrip("/").endswith("/v1"):
            # Most providers want /v1 at the end; if user already gave a
            # full custom path (DashScope's `compatible-mode/v1`), leave it.
            if "compatible-mode" not in h and not h.rstrip("/").endswith("/v1"):
                h = h.rstrip("/") + "/v1"
        return h.rstrip("/")

    def __post_init__(self):
        if self.provider not in {"openai"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not self.host:
            raise ValueError("host is required for provider='openai'")
        if not self.key:
            raise ValueError("key is required for provider='openai'")
        self.base_url = self._normalize_host(self.host)
        self.vision_base_url = (
            self._normalize_host(self.vision_host) if self.vision_host else self.base_url
        )

    # ------------------------------------------------------------------
    def _client(self, vision: bool = False):
        try:
            from openai import OpenAI
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "openai and httpx are required. `pip install openai>=1.30 httpx`."
            ) from e
        # Reuse the underlying httpx connection pool across calls.
        same_endpoint = (
            self.vision_base_url == self.base_url
            and (self.vision_key or self.key) == self.key
        )
        with self._openai_client_lock:
            if vision and same_endpoint:
                vision = False
            if vision:
                if self._vision_openai_client is None:
                    http_client = None
                    if not self.trust_env:
                        http_client = httpx.Client(timeout=self.timeout, trust_env=False)
                    self._vision_openai_client = OpenAI(
                        api_key=self.vision_key or self.key,
                        base_url=self.vision_base_url,
                        timeout=self.timeout,
                        max_retries=0,
                        http_client=http_client,
                    )
                return self._vision_openai_client
            if self._text_openai_client is None:
                http_client = None
                if not self.trust_env:
                    http_client = httpx.Client(timeout=self.timeout, trust_env=False)
                self._text_openai_client = OpenAI(
                    api_key=self.key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=0,
                    http_client=http_client,
                )
            return self._text_openai_client

    def _reserve_call(
        self,
        role: str,
        *,
        target_model: str | None = None,
        is_vision_call: bool = False,
    ) -> int:
        """Atomically reserve one generation attempt against the hard budget."""
        role = role or "unspecified"
        with self._usage_lock:
            if self.call_budget is not None and self.usage.attempted >= self.call_budget:
                raise LLMCallBudgetExceeded(
                    f"LLM call budget exhausted: {self.usage.attempted}/{self.call_budget}"
                )
            self.usage.attempted += 1
            call_index = self.usage.attempted
            self.usage.attempts_by_role[role] = (
                self.usage.attempts_by_role.get(role, 0) + 1
            )
            self._call_trace.append(
                {
                    "call_index": call_index,
                    "role": role,
                    "model": target_model or self.model,
                    "is_vision_call": bool(is_vision_call),
                    "status": "reserved",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "started_unix_seconds": time.time(),
                    "elapsed_seconds": None,
                    "error": None,
                }
            )
            return call_index

    def _record_success(
        self,
        call_index: int,
        target_model: str,
        role: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        role = role or "unspecified"
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += int(prompt_tokens or 0)
            self.usage.completion_tokens += int(completion_tokens or 0)
            self.usage.by_model[target_model] = (
                self.usage.by_model.get(target_model, 0) + 1
            )
            self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
            trace = self._call_trace[call_index - 1]
            trace.update(
                {
                    "model": target_model,
                    "status": "success",
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "elapsed_seconds": time.time() - trace["started_unix_seconds"],
                }
            )

    def _record_failure(self, call_index: int, role: str, error: Exception) -> None:
        role = role or "unspecified"
        with self._usage_lock:
            self.usage.failed += 1
            self.usage.failures_by_role[role] = (
                self.usage.failures_by_role.get(role, 0) + 1
            )
            trace = self._call_trace[call_index - 1]
            trace.update(
                {
                    "status": "failure",
                    "elapsed_seconds": time.time() - trace["started_unix_seconds"],
                    "error": f"{type(error).__name__}: {self._safe_error(error)}"[:500],
                }
            )

    def _safe_error(self, error) -> str:
        """Keep configured credentials out of diagnostics and persisted traces."""
        message = str(error)
        secrets = [self.key, self.vision_key]
        secrets.extend(value for name, value in self.extra_headers.items()
                       if name.lower() in {"authorization", "x-api-key", "api-key"})
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def remaining_calls(self) -> int | None:
        """Return remaining generation attempts, or ``None`` if uncapped."""
        with self._usage_lock:
            if self.call_budget is None:
                return None
            return max(0, self.call_budget - self.usage.attempted)

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        system: str | None,
        user_text: str,
        image_b64s: list[str] | None,
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if image_b64s:
            content: list[dict] = [{"type": "text", "text": user_text}]
            for b64 in image_b64s:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": user_text})
        return msgs

    # ------------------------------------------------------------------







    # ------------------------------------------------------------------
    def chat(
        self,
        user: str,
        system: str | None = None,
        image_b64s: list[str] | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
        extra_body: dict[str, Any] | None = None,
        usage_role: str = "actor",
        force_vision_model: bool = False,
    ) -> str:
        if messages is None:
            messages = self._build_messages(system, user, image_b64s)

        # Route to vision model when image is present and one is configured.
        is_vision_call = bool((image_b64s or force_vision_model) and self.vision_model)
        target_model = self.vision_model if is_vision_call else self.model

        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if is_vision_call:
            # Use vision-specific extras, while inheriting chat-template
            # controls that apply to a shared multimodal checkpoint. Provider
            # request keys such as DashScope's ``enable_thinking`` may be
            # rejected by a dedicated vision endpoint, but vLLM consumes the
            # equivalent value through ``chat_template_kwargs``.
            merged_extra = dict(self.vision_extra_body)
            merged_extra.update(extra_body or {})
            if "enable_thinking" in self.extra_body:
                merged_extra.setdefault(
                    "chat_template_kwargs",
                    {
                        "enable_thinking": bool(
                            self.extra_body["enable_thinking"]
                        )
                    },
                )
            for k in self.text_only_extra_keys:
                merged_extra.pop(k, None)
        else:
            merged_extra = {**self.extra_body, **(extra_body or {})}
            # vLLM ignores DashScope's `enable_thinking` and instead controls
            # reasoning via chat template kwargs. Send both so the same flag
            # works on either provider.
            if "enable_thinking" in merged_extra:
                merged_extra.setdefault(
                    "chat_template_kwargs",
                    {"enable_thinking": bool(merged_extra["enable_thinking"])},
                )
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            call_index = self._reserve_call(
                usage_role,
                target_model=target_model,
                is_vision_call=is_vision_call,
            )
            try:
                client = self._client(vision=is_vision_call)
                resp = client.chat.completions.create(**kwargs)
                u = getattr(resp, "usage", None)
                self._record_success(
                    call_index,
                    target_model,
                    usage_role,
                    int(getattr(u, "prompt_tokens", 0) or 0) if u is not None else 0,
                    int(getattr(u, "completion_tokens", 0) or 0) if u is not None else 0,
                )
                msg = resp.choices[0].message
                # qwen-thinking models may return reasoning_content + content;
                # we only consume the final answer.
                content = getattr(msg, "content", None) or ""
                return content
            except LLMCallBudgetExceeded:
                raise
            except Exception as e:
                last_err = e
                self._record_failure(call_index, usage_role, e)
                if self.debug:
                    print(f"[LLM] attempt {attempt+1} failed on {target_model}: {self._safe_error(e)}")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries on {target_model}: {self._safe_error(last_err)}"
        )

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "attempted": self.usage.attempted,
            "calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "failed": self.usage.failed,
            "by_model": dict(self.usage.by_model),
            "by_role": dict(self.usage.by_role),
            "attempts_by_role": dict(self.usage.attempts_by_role),
            "failures_by_role": dict(self.usage.failures_by_role),
            "call_budget": self.call_budget,
            "remaining_calls": self.remaining_calls(),
        }

    def call_trace(self) -> list[dict[str, Any]]:
        """Return a prompt-free audit trace for every reserved generation."""
        with self._usage_lock:
            return [dict(row) for row in self._call_trace]
