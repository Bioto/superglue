"""superglue — Python bindings for the superglue Rust core.

Observability: the native module initialises Rust ``tracing`` on import. Set ``RUST_LOG``
before importing (e.g. ``export RUST_LOG=info`` or ``RUST_LOG=superglue=debug``) to control
log level; syntax follows ``tracing_subscriber``'s ``EnvFilter``.

Multi-provider models use ``provider:model`` (e.g. ``openai:gpt-4o-mini``,
``anthropic:claude-sonnet-4-20250514``). Set ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.
or pass ``api_keys`` on ``Client``. Use ``upload_file`` / ``message_with_file_bytes`` for
attachments; ``stream()`` runs tool rounds when tools are registered. See
``projects/superglue/docs/multi-provider.md``.

Quick-start
-----------

    import superglue
    from typing import Annotated, Literal

    client = superglue.Client(
        api_key="sk-...",
        model="gpt-5.4-nano-2026-03-17-mini",
        system_prompt="You are a helpful assistant.",
        max_retries=5,
        requests_per_second=10,
    )

    def get_weather(
        location: Annotated[str, "City name, e.g. 'London'"],
        unit: Literal["celsius", "fahrenheit"] = "celsius",
    ) -> dict:
        '''Get the current weather for a location.'''
        return {"location": location, "temperature": 22, "unit": unit}

    # Schema, name, and description are all inferred automatically.
    client.register_tool(get_weather)

    result = client.complete("What is the weather in Paris?")
    print(result.content)
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import json
import os
from types import SimpleNamespace
from typing import Any, Callable, Generic, Literal, TypeVar

import httpx

from ._superglue import (  # noqa: F401  (re-exported for users)
    AgentEngine as _RustAgentEngine,
    AgentSpec as _RustAgentSpec,
    BatchRequest as _RustBatchRequest,
    BatchResponse,
    BatchResult,
    Client as _RustClient,
    UploadedFile as _RustUploadedFile,
    CompletionOutcome,
    Conversation as _RustConversation,
    ResponseOutcome,
    ResponseStreamOutcome,
    StreamOutcome,
    _RustStatusEmitter,
    version,
)

__version__ = version()
UploadedFile = _RustUploadedFile
from ._introspect import schema_from_fn, wrap_for_rust

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class GuardrailStage:
    """String constants for guardrail stage names.

    Pass these to :meth:`Client.register_guardrail`, :meth:`Client.add_blocklist`,
    and :meth:`Client.add_max_length` to control when the guardrail fires.

    Stages
    ------
    INPUT:
        Runs on the user message **before** the first LLM API call.
        A block propagates immediately as a ``GuardrailError``.
    OUTPUT:
        Runs on the final assistant response **after** the tool loop.
        A block triggers an LLM retry loop (up to ``max_output_retries``).
    BOTH:
        Registers the guardrail on both input and output (default).

    Example
    -------
    ::

        def no_profanity(content: str) -> str:
            if "badword" in content:
                raise ValueError("profanity detected")
            return content

        client.register_guardrail(no_profanity, stage=GuardrailStage.INPUT)
    """

    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


class HookStage:
    """String constants for hook stage names.

    Pass these to :meth:`Client.register_hook` to choose when your hook fires.

    Stages
    ------
    PRE_COMPLETION:
        Before each LLM API call. Content: last user message. **Observation only.**
    POST_COMPLETION:
        After each LLM response, before tool dispatch. Content: assistant text. **Observation only.**
    PRE_TOOL:
        Before a tool is invoked. Content: JSON-serialized args. **Mutating** — return a
        new content string (or dict with ``"content"`` key) to override the args.
    POST_TOOL:
        After a tool returns. Content: JSON-serialized result. **Mutating** — return a
        new content string to override the result seen by the model.
    ON_RETRY:
        When an HTTP retry is attempted. Content: error description. **Observation only.**
    PRE_BATCH_ITEM:
        Before each item in a :meth:`Client.batch` call. Content: prompt. **Observation only.**
    POST_BATCH_ITEM:
        After each batch item completes. Content: response text or error. **Observation only.**

    Example
    -------
    ::

        def log_completion(ctx):
            print(f"[{ctx['stage']}] {ctx['content'][:80]}")

        client.register_hook(HookStage.PRE_COMPLETION, log_completion)
    """

    PRE_COMPLETION = "pre_completion"
    POST_COMPLETION = "post_completion"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_RETRY = "on_retry"
    PRE_BATCH_ITEM = "pre_batch_item"
    POST_BATCH_ITEM = "post_batch_item"


@dataclasses.dataclass
class ProcessEvent:
    """Typed process event emitted during LLM / tool execution."""

    kind: str
    request_id: str
    round: int = 0
    model: str = ""
    tool_call_count: int = 0
    usage: dict[str, int] | None = None
    estimated_cost_usd: float | None = None
    error_type: str | None = None
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    timestamp_ms: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProcessEvent:
        usage = raw.get("usage")
        return cls(
            kind=str(raw.get("kind", "")),
            request_id=str(raw.get("request_id", "")),
            round=int(raw.get("round", 0)),
            model=str(raw.get("model", "")),
            tool_call_count=int(raw.get("tool_call_count", 0)),
            usage=dict(usage) if isinstance(usage, dict) else None,
            estimated_cost_usd=raw.get("estimated_cost_usd"),
            error_type=raw.get("error_type"),
            metadata=dict(raw.get("metadata") or {}),
            timestamp_ms=int(raw.get("timestamp_ms", 0)),
        )


class StatusEmitter:
    """Fan-out dispatcher for :class:`ProcessEvent` observers (gluellm-compatible)."""

    def __init__(self) -> None:
        self._rust = _RustStatusEmitter()

    def subscribe(self, callback: Callable[[ProcessEvent], None]) -> None:
        def _bridge(raw: dict[str, Any]) -> None:
            callback(ProcessEvent.from_dict(raw))

        self._rust.subscribe(_bridge)

    def bind_observers(
        self,
        *,
        on_status: Callable[[ProcessEvent], None] | None = None,
        sinks: list[Callable[[ProcessEvent], None]] | None = None,
    ) -> None:
        if on_status is not None:
            self.subscribe(on_status)
        for sink in sinks or []:
            self.subscribe(sink)


@dataclasses.dataclass
class BatchRequest:
    """A single item in a batch completion call.

    Parameters
    ----------
    prompt:
        The user message to send.
    id:
        Optional caller-supplied identifier. Auto-generated if ``None``.
    system_prompt:
        Per-request system prompt that overrides the client-level system
        prompt for this item only.
    """

    prompt: str
    id: str | None = None
    system_prompt: str | None = None


class Client:
    """superglue client.

    Parameters
    ----------
    api_key:
        OpenAI-compatible API key.
    model:
        Model name, e.g. ``"gpt-5.4-nano-2026-03-17-mini"`` (default).
    base_url:
        API base URL (default: ``https://api.openai.com``).
    system_prompt:
        Optional system prompt prepended to every request.
    max_tool_rounds:
        Maximum number of tool-call / answer iterations (default 16).
    temperature:
        Sampling temperature forwarded to the model.
    max_completion_tokens:
        Maximum tokens in the response.
    seed:
        Random seed for reproducibility.

    Retry
    -----
    max_retries:
        Maximum retry attempts on transient errors — 429, 502, 503, 504,
        and transport timeouts (default 3). Set to ``0`` to disable retries.
    retry_initial_delay_ms:
        Initial backoff delay in milliseconds before the first retry (default 1000).
    retry_max_delay_ms:
        Upper bound on backoff delay in milliseconds (default 6000).
    retry_multiplier:
        Exponential backoff multiplier applied after each attempt (default 3.0).

    Rate limiting
    -------------
    requests_per_second:
        Optional QPS cap applied across all requests made by this client.
        ``None`` means unlimited (default). Set to e.g. ``10`` to cap at 10 req/s.

    Timeouts
    --------
    timeout_secs:
        Total per-request timeout in seconds — connect + response body (default 60).
    connect_timeout_secs:
        Maximum time to establish a TCP connection in seconds (default 30).

    Connection pool
    ---------------
    pool_max_idle_per_host:
        Maximum number of idle keep-alive connections retained per host in the
        connection pool (default 50). Increase for high-throughput workloads that
        send many concurrent requests to the same host.
    pool_idle_timeout_secs:
        Seconds before an idle pooled connection is evicted. ``None`` uses
        reqwest's built-in default of 90 s (default ``None``).

    Examples
    --------
    Default (sensible retry + no rate limit)::

        client = superglue.Client(api_key="sk-...", model="gpt-5.4-nano-2026-03-17-mini")

    Aggressive retry with rate cap::

        client = superglue.Client(
            api_key="sk-...",
            model="gpt-5.4-nano-2026-03-17",
            max_retries=5,
            retry_initial_delay_ms=100,
            retry_multiplier=1.5,
            requests_per_second=10,
            timeout_secs=30,
        )
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.4-nano-2026-03-17-mini",
        base_url: str = "https://api.openai.com",
        system_prompt: str | None = None,
        max_tool_rounds: int = 16,
        # retry
        max_retries: int = 3,
        retry_initial_delay_ms: int = 1000,
        retry_max_delay_ms: int = 6000,
        retry_multiplier: float = 3.0,
        # rate limiting
        requests_per_second: int | None = None,
        # timeouts
        timeout_secs: int = 60,
        connect_timeout_secs: int = 30,
        # guardrails — max retry attempts when the LLM output is rejected
        max_output_retries: int = 3,
        # connection pool
        pool_max_idle_per_host: int = 50,
        pool_idle_timeout_secs: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        status_emitter: StatusEmitter | None = None,
        model_fallback_models: list[str] | None = None,
        api_keys: dict[str, str] | None = None,
        requests_per_second_for: dict[str, int] | None = None,
        max_upload_bytes: int | None = None,
        tool_mode: str = "standard",  # standard | dynamic | code
        tool_route_model: str | None = None,
        condense_tool_messages: bool = False,
        aaak_tool_condensing: bool = False,
        summarize_context_enabled: bool = False,
        summarize_context_threshold: int = 20,
        summarize_context_keep_recent: int = 6,
        aaak_compression_enabled: bool = False,
        aaak_compression_model: str | None = None,
    ) -> None:
        self._rust = _RustClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            max_tool_rounds=max_tool_rounds,
            max_retries=max_retries,
            retry_initial_delay_ms=retry_initial_delay_ms,
            retry_max_delay_ms=retry_max_delay_ms,
            retry_multiplier=retry_multiplier,
            requests_per_second=requests_per_second,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            max_output_retries=max_output_retries,
            pool_max_idle_per_host=pool_max_idle_per_host,
            pool_idle_timeout_secs=pool_idle_timeout_secs,
            reasoning_effort=reasoning_effort,
            status_emitter=status_emitter._rust if status_emitter else None,
            model_fallback_models=model_fallback_models,
            api_keys=api_keys,
            requests_per_second_for=requests_per_second_for,
            max_upload_bytes=max_upload_bytes,
            tool_mode=tool_mode,
            tool_route_model=tool_route_model,
            condense_tool_messages=condense_tool_messages,
            aaak_tool_condensing=aaak_tool_condensing,
            summarize_context_enabled=summarize_context_enabled,
            summarize_context_threshold=summarize_context_threshold,
            summarize_context_keep_recent=summarize_context_keep_recent,
            aaak_compression_enabled=aaak_compression_enabled,
            aaak_compression_model=aaak_compression_model,
        )

    def upload_file(
        self,
        path: str,
        purpose: str = "user_data",
        provider: str | None = None,
    ) -> _RustUploadedFile:
        """Upload a file to the provider Files API (or inline metadata for Anthropic)."""
        return self._rust.upload_file(path, purpose, provider)

    @staticmethod
    def message_with_file_bytes(
        filename: str,
        file_bytes: bytes,
        text: str | None = None,
    ) -> Any:
        """Build a user message with inline file bytes for chat."""
        return _RustClient.message_with_file_bytes(filename, file_bytes, text)

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        fn: Callable | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        static_tool: bool = False,
    ) -> None:
        """Register a callable as a tool.

        **Minimal form** — let superglue infer everything from the function:

            client.register_tool(get_weather)

        superglue will:

        * Use ``fn.__name__`` as the tool name.
        * Use the first line of ``fn.__doc__`` as the description.
        * Build the JSON Schema from type annotations (``str``, ``int``,
          ``Literal[...]``, ``Optional[...]``, ``Annotated[X, "desc"]``, …).
        * Parse ``Args:`` sections in Google-style docstrings for per-parameter
          descriptions.

        **Override selectively** — provide any combination of explicit values:

            client.register_tool(get_weather, description="Custom description")

        **Fully explicit** (legacy style; still supported):

            client.register_tool(
                fn=get_weather,
                name="get_weather",
                description="...",
                parameters={...},
            )

        Function calling conventions
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Two styles are supported:

        *Typed parameters* (recommended — fully introspectable):

            def get_weather(location: str, unit: str = "celsius") -> dict:
                '''Get weather for a location.

                Args:
                    location: City name, e.g. 'Paris'
                    unit: Temperature unit
                '''
                return {"temp": 22}

        *Legacy args dict* (schema must be supplied manually):

            def get_weather(args: dict) -> dict:
                return {"temp": args.get("location")}
        """
        if fn is None:
            raise TypeError("register_tool() requires a callable as the first argument")

        inferred_name, inferred_desc, inferred_params = schema_from_fn(fn)

        resolved_name = name or inferred_name
        resolved_desc = description or inferred_desc
        resolved_params = parameters or inferred_params

        # Wrap typed-kwarg functions so Rust always sees fn(args: dict) -> dict
        rust_fn = wrap_for_rust(fn)

        self._rust.register_tool(
            name=resolved_name,
            description=resolved_desc,
            parameters=resolved_params,
            fn=rust_fn,
            static_tool=static_tool,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def register_hook(
        self,
        stage: str,
        handler: Callable[[dict[str, Any]], str | dict | None],
        *,
        name: str = "hook",
        error_strategy: str = "skip",
    ) -> None:
        """Register a lifecycle hook that fires at the given pipeline stage.

        Parameters
        ----------
        stage:
            One of the :class:`HookStage` string constants (e.g.
            ``HookStage.PRE_COMPLETION``).
        handler:
            Callable with signature ``fn(ctx: dict) -> str | dict | None``.

            The ``ctx`` dict has keys:

            * ``"stage"`` (str) — which stage fired
            * ``"content"`` (str) — stage-specific payload (see :class:`HookStage`)
            * ``"metadata"`` (dict) — extra keys, e.g. ``"tool_name"`` for tool stages

            Return values:

            * ``None`` — leave content unchanged (observation-only behaviour)
            * ``str`` — replace the content string (mutating behaviour)
            * ``dict`` with key ``"content"`` — replace the content string

        name:
            Human-readable label used in warning and error messages.
        error_strategy:
            ``"skip"`` *(default)* — log and continue on handler error.
            ``"abort"`` — propagate the error and cancel the pipeline call.

        Examples
        --------
        Log every LLM call (observation-only)::

            def log_pre(ctx):
                print(f"[pre_completion] {ctx['content'][:80]}")

            client.register_hook(HookStage.PRE_COMPLETION, log_pre)

        Mutate tool arguments before invocation::

            import json

            def uppercase_location(ctx):
                args = json.loads(ctx["content"])
                args["location"] = args.get("location", "").upper()
                return json.dumps(args)

            client.register_hook(HookStage.PRE_TOOL, uppercase_location)
        """
        self._rust.register_hook(
            stage,
            handler,
            name=name,
            error_strategy=error_strategy,
        )

    def tools_registry_ptr(self) -> int:
        """Opaque pointer to the internal tool registry (for extension modules)."""
        return self._rust.tools_registry_ptr()

    def hooks_registry_ptr(self) -> int:
        """Opaque pointer to the internal hook registry (for extension modules)."""
        return self._rust.hooks_registry_ptr()

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    def register_guardrail(
        self,
        handler: Callable[[str], str | None],
        *,
        stage: str = GuardrailStage.BOTH,
        name: str = "guardrail",
    ) -> None:
        """Register a custom Python guardrail callable.

        Parameters
        ----------
        handler:
            Callable with signature ``fn(content: str) -> str | None``.

            * Return ``None`` or the same string → allow content unchanged.
            * Return a different string → allow with transformation (e.g. redaction).
            * Raise any exception → block the content (the exception message is used
              as the reason string in the resulting ``GuardrailError``).

        stage:
            One of :attr:`GuardrailStage.INPUT`, :attr:`GuardrailStage.OUTPUT`,
            or :attr:`GuardrailStage.BOTH` (default).
        name:
            Human-readable label for logging and error messages.

        Examples
        --------
        Block profanity on input::

            def no_profanity(content: str) -> str | None:
                if "badword" in content.lower():
                    raise ValueError("profanity detected")
                return content

            client.register_guardrail(no_profanity, stage=GuardrailStage.INPUT)

        Redact PII on output::

            import re

            def redact_ssn(content: str) -> str:
                return re.sub(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN]", content)

            client.register_guardrail(redact_ssn, stage=GuardrailStage.OUTPUT)
        """
        self._rust.register_guardrail(handler, stage=stage, name=name)

    def add_blocklist(
        self,
        patterns: list[str],
        *,
        action: str = "block",
        stage: str = GuardrailStage.BOTH,
        name: str = "blocklist",
    ) -> None:
        """Register the built-in blocklist guardrail.

        Parameters
        ----------
        patterns:
            List of regex strings. Any content matching one of these patterns
            triggers the ``action``.
        action:
            ``"block"`` *(default)* — reject content matching any pattern.
            ``"redact"`` — replace matching text with ``[REDACTED]`` and allow.
        stage:
            One of :attr:`GuardrailStage.INPUT`, :attr:`GuardrailStage.OUTPUT`,
            or :attr:`GuardrailStage.BOTH` (default).
        name:
            Human-readable label for error messages.

        Examples
        --------
        Block input containing forbidden words::

            client.add_blocklist(["password", "secret"], stage=GuardrailStage.INPUT)

        Redact phone numbers from output::

            client.add_blocklist(
                [r"\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b"],
                action="redact",
                stage=GuardrailStage.OUTPUT,
            )
        """
        self._rust.add_blocklist_guardrail(patterns, action=action, stage=stage, name=name)

    def add_max_length(
        self,
        *,
        max_input: int | None = None,
        max_output: int | None = None,
        strategy: str = "block",
        name: str = "max_length",
    ) -> None:
        """Register the built-in maximum-length guardrail.

        Parameters
        ----------
        max_input:
            Maximum character count for user messages. ``None`` means unlimited.
        max_output:
            Maximum character count for assistant responses. ``None`` means unlimited.
        strategy:
            ``"block"`` *(default)* — reject content that exceeds the limit.
            ``"truncate"`` — silently truncate to the limit and allow.
        name:
            Human-readable label for error messages.

        Examples
        --------
        Block very long input messages::

            client.add_max_length(max_input=4096)

        Truncate long outputs::

            client.add_max_length(max_output=2000, strategy="truncate")
        """
        self._rust.add_max_length_guardrail(
            max_input=max_input, max_output=max_output, strategy=strategy, name=name
        )

    def add_pii_guardrail(
        self,
        *,
        stage: str = GuardrailStage.BOTH,
        name: str = "pii_redact",
    ) -> None:
        """Register the built-in PII redaction guardrail.

        Automatically redacts emails, US phone numbers, SSNs, and credit card
        numbers by replacing them with ``[REDACTED]``. This guardrail always
        allows content through — it never blocks.

        Parameters
        ----------
        stage:
            One of :attr:`GuardrailStage.INPUT`, :attr:`GuardrailStage.OUTPUT`,
            or :attr:`GuardrailStage.BOTH` (default).
        name:
            Human-readable label for error messages.
        """
        self._rust.add_pii_guardrail(stage=stage, name=name)

    def system_one(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``state`` with TypeSafe System One.

        ``questions`` is a map of id to ``{type, instructions, criteria?}``.
        """
        return self._rust.system_one(state, questions, model=model)

    def add_typesafe_guardrail(
        self,
        policy: dict[str, Any],
        *,
        stage: str = "both",
        name: str = "typesafe",
    ) -> None:
        """Register a TypeSafe System One guardrail.

        ``policy`` is ``{questions, rules?, model?}``. Fail-closed on errors.
        """
        self._rust.add_typesafe_guardrail(policy, stage=stage, name=name)

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        /,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionOutcome:
        """Run a blocking chat completion (with tool loop if tools are registered).

        Parameters
        ----------
        prompt:
            User message to send.
        timeout_secs:
            Override the total per-request timeout (connect + body) for this
            call only. Uses the client-level default when ``None``.
        connect_timeout_secs:
            Override the TCP connect timeout for this call only.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``CompletionOutcome.request_id``.

        Returns
        -------
        CompletionOutcome
            ``.content``    — assistant text response (``str | None``)
            ``.rounds``     — number of LLM calls made
            ``.usage``      — token usage dict (``prompt``, ``completion``, ``total``)
            ``.request_id`` — correlation ID for this request
        """
        return self._rust.complete(
            prompt,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
            reasoning_effort=reasoning_effort,
        )

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        /,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionOutcome:
        """Run a completion with an explicit OpenAI-style ``messages`` list.

        Returns a :class:`CompletionOutcome` whose ``messages`` attribute is the
        caller-visible transcript after the turn (suitable to pass back in on the
        next call).
        """
        return self._rust.complete_messages(
            messages,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def stream(
        self,
        prompt: str,
        /,
        *,
        on_token: Callable[[str], None],
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
    ) -> StreamOutcome:
        """Run a streaming chat completion.

        Each token is delivered to *on_token* as it arrives.

        Parameters
        ----------
        prompt:
            User message to send.
        on_token:
            Callback invoked for each content token.
        timeout_secs:
            Override the total per-request timeout for this call only.
        connect_timeout_secs:
            Override the TCP connect timeout for this call only.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``StreamOutcome.request_id``.

        Returns
        -------
        StreamOutcome
            ``.content``       — full assembled response text
            ``.finish_reason`` — OpenAI finish reason (``"stop"``, ``"length"``, …)
            ``.usage``         — token usage dict (may be ``None`` for some models)
            ``.request_id``    — correlation ID for this request
        """
        return self._rust.stream(
            prompt,
            on_token,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def complete_response(
        self,
        prompt: str,
        /,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ResponseOutcome:
        """OpenAI Responses API completion (with tool loop when tools are registered)."""
        return self._rust.complete_response(
            prompt,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
            reasoning_effort=reasoning_effort,
        )

    def stream_response(
        self,
        prompt: str,
        /,
        *,
        on_token: Callable[[str], None],
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
    ) -> ResponseStreamOutcome:
        """Stream a Responses API completion; each text delta is passed to ``on_token``."""
        return self._rust.stream_response(
            prompt,
            on_token,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def connect_mcp_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        prefix: str | None = None,
    ) -> None:
        """Connect an MCP server over stdio and register its tools on this client."""
        self._rust.connect_mcp_stdio(command, args, env, prefix)

    def connect_mcp_http(self, url: str, prefix: str | None = None) -> None:
        """Connect an MCP server over streamable HTTP and register its tools."""
        self._rust.connect_mcp_http(url, prefix)

    def batch(
        self,
        requests: list[str] | list[BatchRequest],
        *,
        max_concurrent: int = 5,
        error_strategy: str = "continue",
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
    ) -> BatchResponse:
        """Run multiple prompts concurrently as a batch.

        Parameters
        ----------
        requests:
            A list of plain strings **or** :class:`BatchRequest` objects.
            Plain strings are wrapped in ``BatchRequest`` automatically.
        max_concurrent:
            Maximum number of simultaneous in-flight API calls (default 5).
        error_strategy:
            How to handle per-request failures:

            * ``"continue"`` *(default)* — include all results; failures have
              ``success=False`` and a non-empty ``error``.
            * ``"skip"`` — return only successful results; drop failures silently.
            * ``"fail_fast"`` — raise ``RuntimeError`` on the first failure.
        timeout_secs:
            Override the total per-request timeout for every item in this
            batch. Uses the client-level default when ``None``.
        connect_timeout_secs:
            Override the TCP connect timeout for every item in this batch.

        Returns
        -------
        BatchResponse
            ``.results``        — list of :class:`BatchResult` (one per request)
            ``.total_requests`` — total input count (before any filtering)
            ``.successful``     — count of items with ``success=True``
            ``.failed``         — count of items with ``success=False``
            ``.elapsed_secs``   — total wall-clock time for the batch
            ``.total_usage``    — aggregated token usage dict (or ``None``)
        """
        rust_requests: list[_RustBatchRequest] = []
        for r in requests:
            if isinstance(r, str):
                rust_requests.append(_RustBatchRequest(prompt=r))
            elif isinstance(r, BatchRequest):
                rust_requests.append(
                    _RustBatchRequest(
                        prompt=r.prompt,
                        id=r.id,
                        system_prompt=r.system_prompt,
                    )
                )
            else:
                raise TypeError(
                    f"batch() expects str or BatchRequest items, got {type(r).__name__!r}"
                )

        return self._rust.batch(
            rust_requests,
            max_concurrent=max_concurrent,
            error_strategy=error_strategy,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
        )

    # ------------------------------------------------------------------
    # Agent convenience
    # ------------------------------------------------------------------

    def run_agent(
        self,
        spec: "AgentSpec",
        user_message: str,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionOutcome:
        """Run an :class:`AgentSpec` against ``user_message``.

        The agent's compiled system prompt is injected automatically. Tools,
        hooks, and guardrails registered on this :class:`Client` are shared.

        Parameters
        ----------
        spec:
            The agent's configuration (persona, goals, constraints, …).
        user_message:
            The user's message to send to the agent.
        timeout_secs:
            Per-call total request timeout override.
        connect_timeout_secs:
            Per-call TCP connect timeout override.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``CompletionOutcome.request_id``.

        Returns
        -------
        CompletionOutcome
        """
        return self._rust.run_agent(
            spec._rust,
            user_message,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._rust)


Client.response = Client.complete  # gluellm alias


class Conversation:
    """Multi-turn chat session sharing a :class:`Client`'s tools and settings."""

    def __init__(self, client: Client) -> None:
        if not isinstance(client, Client):
            raise TypeError("Conversation requires a superglue.Client instance")
        self._rust = _RustConversation(client._rust)

    def push_user(self, text: str) -> None:
        self._rust.push_user(text)

    def push_assistant_text(self, text: str) -> None:
        self._rust.push_assistant_text(text)

    @property
    def messages(self) -> list[Any]:
        return self._rust.messages

    def complete(
        self,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
    ) -> CompletionOutcome:
        return self._rust.complete(
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def __repr__(self) -> str:
        return repr(self._rust)


# ---------------------------------------------------------------------------
# AgentSpec — declarative agent configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AgentSpec:
    """Declarative configuration for a superglue agent.

    Mirrors ``gluellm.Agent`` — pure data, no execution logic.

    Parameters
    ----------
    name:
        Unique identifier used in logs and traces.
    persona:
        Personality and role description. Becomes: "You are {persona}."
    goals:
        High-level objectives (each rendered as a bullet under ``## Goals``).
    constraints:
        Hard rules the agent must not violate (``## Constraints``).
    model:
        LLM model override (e.g. ``"gpt-5.4-nano-2026-03-17"``). Empty string keeps the
        client's default model.
    max_tool_rounds:
        Maximum tool-call iterations per :meth:`Agent.run` call (default 16).
    max_output_retries:
        Maximum output-guardrail retry attempts (default 3).
    system_prompt:
        When set, skips compilation of persona/goals/constraints and uses this
        verbatim as the system prompt.

    Examples
    --------
    >>> spec = AgentSpec(
    ...     name="Researcher",
    ...     persona="an expert research assistant",
    ...     goals=["Provide accurate, well-sourced answers"],
    ...     constraints=["Never speculate without labelling it clearly"],
    ... )
    >>> spec.compile_system_prompt()
    'You are an expert research assistant.\\n\\n## Goals\\n...'
    """

    name: str
    persona: str
    goals: list[str] = dataclasses.field(default_factory=list)
    constraints: list[str] = dataclasses.field(default_factory=list)
    model: str = ""
    max_tool_rounds: int = 16
    max_output_retries: int = 3
    system_prompt: str | None = None
    reasoning_effort: ReasoningEffort | None = None

    def compile_system_prompt(self) -> str:
        """Return the compiled system prompt string.

        If :attr:`system_prompt` is set, it is returned unchanged. Otherwise
        the persona, goals, and constraints are formatted into a structured
        prompt.
        """
        rust_spec = self._to_rust()
        return rust_spec.compile_system_prompt()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_rust(self) -> _RustAgentSpec:
        return _RustAgentSpec(
            name=self.name,
            persona=self.persona,
            goals=self.goals,
            constraints=self.constraints,
            model=self.model,
            max_tool_rounds=self.max_tool_rounds,
            max_output_retries=self.max_output_retries,
            system_prompt=self.system_prompt,
            reasoning_effort=self.reasoning_effort,
        )

    # Keep a cached property so Client.run_agent can reach the Rust object.
    @property
    def _rust(self) -> _RustAgentSpec:
        return self._to_rust()


# ---------------------------------------------------------------------------
# Agent — execution wrapper (mirrors gluellm's AgentExecutor)
# ---------------------------------------------------------------------------

class Agent:
    """Execution wrapper for a superglue :class:`AgentSpec`.

    ``Agent`` owns its own HTTP connection pool, tool registry, hooks, and
    guardrails — independent from any :class:`Client`. This mirrors
    ``gluellm.AgentExecutor``.

    Parameters
    ----------
    spec:
        Agent configuration (persona, goals, constraints, model, …).
    api_key:
        OpenAI-compatible API key.
    base_url:
        API base URL (default ``"https://api.openai.com"``).
    model:
        Fallback model when ``spec.model`` is empty (default
        ``"gpt-5.4-nano-2026-03-17-mini"``).
    max_retries:
        Maximum HTTP retry attempts (default 3).
    retry_initial_delay_ms:
        Initial retry back-off in milliseconds (default 1000).
    retry_max_delay_ms:
        Maximum retry back-off in milliseconds (default 6000).
    retry_multiplier:
        Exponential back-off multiplier (default 3.0).
    requests_per_second:
        Steady-state QPS cap; ``None`` means unlimited (default ``None``).
    timeout_secs:
        Total request timeout in seconds (default 60).
    connect_timeout_secs:
        TCP connect timeout in seconds (default 15).
    pool_max_idle_per_host:
        Maximum idle keep-alive connections per host in the pool (default 8).
    pool_idle_timeout_secs:
        Seconds before an idle pooled connection is evicted; ``None`` uses the
        reqwest default of 90 s (default ``None``).

    Examples
    --------
    >>> spec = AgentSpec(name="Helper", persona="a helpful assistant")
    >>> agent = Agent(spec, api_key="sk-...")
    >>> result = agent.run("Hello!")
    >>> print(result.content)
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com",
        model: str = "gpt-5.4-nano-2026-03-17-mini",
        max_retries: int = 3,
        retry_initial_delay_ms: int = 1000,
        retry_max_delay_ms: int = 6000,
        retry_multiplier: float = 3.0,
        requests_per_second: int | None = None,
        timeout_secs: int = 60,
        connect_timeout_secs: int = 30,
        pool_max_idle_per_host: int = 50,
        pool_idle_timeout_secs: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        status_emitter: StatusEmitter | None = None,
    ) -> None:
        self._spec = spec
        self._engine = _RustAgentEngine(
            spec=spec._to_rust(),
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_retries=max_retries,
            retry_initial_delay_ms=retry_initial_delay_ms,
            retry_max_delay_ms=retry_max_delay_ms,
            retry_multiplier=retry_multiplier,
            requests_per_second=requests_per_second,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            pool_max_idle_per_host=pool_max_idle_per_host,
            pool_idle_timeout_secs=pool_idle_timeout_secs,
            reasoning_effort=reasoning_effort or spec.reasoning_effort,
            status_emitter=status_emitter._rust if status_emitter else None,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        fn: Callable,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        static_tool: bool = False,
    ) -> None:
        """Register a Python callable as a tool.

        ``name``, ``description``, and ``parameters`` are inferred from the
        function's signature and docstring when not provided.

        Parameters
        ----------
        fn:
            The Python callable to register. Must accept a single ``dict``
            argument (the model-supplied arguments) and return a ``dict``.
        name:
            Tool name (defaults to ``fn.__name__``).
        description:
            Tool description shown to the model (defaults to the first line
            of ``fn``'s docstring).
        parameters:
            JSON Schema dict for the tool's parameters (inferred via type
            annotations when omitted).
        """
        inferred_name, inferred_desc, inferred_params = schema_from_fn(fn)
        resolved_name = name or inferred_name
        resolved_desc = description or inferred_desc
        resolved_params = parameters or inferred_params
        rust_fn = wrap_for_rust(fn)
        self._engine.register_tool(
            name=resolved_name,
            description=resolved_desc,
            parameters=resolved_params,
            fn=rust_fn,
            static_tool=static_tool,
        )

    def register_hook(
        self,
        stage: str,
        handler: Callable,
        *,
        name: str = "hook",
        error_strategy: str = "skip",
    ) -> None:
        """Register a lifecycle hook.

        Parameters
        ----------
        stage:
            One of ``"pre_completion"``, ``"post_completion"``, ``"pre_tool"``,
            ``"post_tool"``, ``"on_retry"``, ``"pre_batch_item"``,
            ``"post_batch_item"``.
        handler:
            Callable ``fn(ctx: dict) -> str | None`` receiving a context dict
            with keys ``stage``, ``content``, ``metadata``.
        name:
            Human-readable name for logging (default ``"hook"``).
        error_strategy:
            ``"skip"`` (default) or ``"abort"``.
        """
        self._engine.register_hook(
            stage=stage, handler=handler, name=name, error_strategy=error_strategy
        )

    def register_guardrail(
        self,
        handler: Callable,
        *,
        stage: str = "both",
        name: str = "guardrail",
    ) -> None:
        """Register a custom guardrail callable.

        Parameters
        ----------
        handler:
            Callable ``fn(stage: str, content: str) -> str | bool | None``.
            Return ``False`` or raise ``ValueError`` to **block**. Return a
            string to **transform** the content. Return ``None`` / ``True``
            to **allow** unchanged.
        stage:
            ``"input"``, ``"output"``, or ``"both"`` (default).
        name:
            Human-readable name for logging (default ``"guardrail"``).
        """
        self._engine.register_guardrail(handler=handler, stage=stage, name=name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionOutcome:
        """Run a single-turn completion (with tool loop).

        Parameters
        ----------
        user_message:
            The user's message.
        timeout_secs:
            Per-call total request timeout override.
        connect_timeout_secs:
            Per-call TCP connect timeout override.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``CompletionOutcome.request_id``.

        Returns
        -------
        CompletionOutcome
            ``.content``    — final assistant text.
            ``.rounds``     — number of tool-call rounds.
            ``.usage``      — token usage dict or ``None``.
            ``.request_id`` — correlation ID for this request.
        """
        return self._engine.run(
            user_message,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def stream(
        self,
        user_message: str,
        *,
        on_token: Callable[[str], None],
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
    ) -> StreamOutcome:
        """Stream a completion, calling ``on_token`` for each content delta.

        Parameters
        ----------
        user_message:
            The user's message.
        on_token:
            Callable invoked with each content token string as it arrives.
        timeout_secs:
            Per-call total request timeout override.
        connect_timeout_secs:
            Per-call TCP connect timeout override.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``StreamOutcome.request_id``.

        Returns
        -------
        StreamOutcome
            ``.content``       — full accumulated text.
            ``.finish_reason`` — why the stream ended.
            ``.usage``         — token usage dict or ``None``.
            ``.request_id``    — correlation ID for this request.
        """
        return self._engine.stream(
            user_message,
            on_token,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def __repr__(self) -> str:
        return repr(self._engine    )


def _merge_status_emitter(
    *,
    status_emitter: StatusEmitter | None,
    on_status: Callable[[ProcessEvent], None] | None,
    sinks: list[Callable[[ProcessEvent], None]] | None,
) -> StatusEmitter:
    emitter = status_emitter or StatusEmitter()
    emitter.bind_observers(on_status=on_status, sinks=sinks)
    return emitter


# ---------------------------------------------------------------------------
# Executors — composable, hook-wrapped execution units
# ---------------------------------------------------------------------------

_StructuredT = TypeVar("_StructuredT")


class Executor(abc.ABC, Generic[_StructuredT]):
    """Abstract base class for superglue executors.

    An ``Executor`` wraps an underlying completion strategy (plain client or
    agent) and exposes a single :meth:`execute` method. It also supports
    **executor-level hooks** that fire *before* and *after* the LLM call,
    independently of the in-pipeline hooks registered on the underlying
    :class:`Client` or :class:`Agent`.

    Subclasses implement :meth:`_execute_internal`.

    Hook protocol
    -------------
    Hooks are callables with signature ``fn(stage: str, content: str) -> str | None``.
    ``stage`` is one of ``"pre_executor"`` / ``"post_executor"``. Returning
    ``None`` keeps the content unchanged; returning a string replaces it.

    Parameters
    ----------
    hooks:
        List of callables invoked at the executor boundary. Applied in order.
    """

    def __init__(self, *, hooks: list[Callable] | None = None, on_status: Callable[[ProcessEvent], None] | None = None, sinks: list[Callable[[ProcessEvent], None]] | None = None) -> None:
        self._hooks: list[Callable] = hooks or []
        self._on_status = on_status
        self._sinks = sinks or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        user_message: str,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionOutcome:
        """Execute a query, running pre/post executor hooks around it.

        Parameters
        ----------
        user_message:
            The user's message / query.
        timeout_secs:
            Per-call total request timeout override.
        connect_timeout_secs:
            Per-call TCP connect timeout override.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``CompletionOutcome.request_id``.

        Returns
        -------
        CompletionOutcome
        """
        processed = self._run_hooks("pre_executor", user_message)
        result = self._execute_internal(
            processed,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )
        final_content = self._run_hooks("post_executor", result.content or "")
        if final_content != (result.content or ""):
            result = _patch_outcome(result, final_content)
        return result

    def add_hook(self, fn: Callable) -> None:
        """Append a hook callable to this executor.

        Parameters
        ----------
        fn:
            ``fn(stage: str, content: str) -> str | None``
        """
        self._hooks.append(fn)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _execute_internal(
        self,
        user_message: str,
        *,
        timeout_secs: int | None,
        connect_timeout_secs: int | None,
        request_id: str | None,
    ) -> CompletionOutcome:
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_hooks(self, stage: str, content: str) -> str:
        for fn in self._hooks:
            try:
                result = fn(stage, content)
                if isinstance(result, str):
                    content = result
            except Exception:
                pass
        return content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hooks={len(self._hooks)})"


@dataclasses.dataclass
class _PatchedOutcome:
    """Duck-typed replacement for CompletionOutcome when hooks modify content."""
    content: str | None
    rounds: int
    usage: Any
    request_id: str = ""
    messages: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def _patch_outcome(outcome: CompletionOutcome, new_content: str) -> Any:
    """Return a duck-typed CompletionOutcome with content replaced.

    ``CompletionOutcome`` is an immutable Rust-backed type; when executor
    hooks modify the text we wrap it in a plain Python object that exposes
    the same attributes.
    """
    msgs = getattr(outcome, "messages", [])
    if not isinstance(msgs, list):
        msgs = list(msgs) if msgs is not None else []
    return _PatchedOutcome(
        content=new_content,
        rounds=outcome.rounds,
        usage=outcome.usage,
        request_id=getattr(outcome, "request_id", ""),
        messages=list(msgs),
    )


# ---------------------------------------------------------------------------
# SimpleExecutor — plain Client, no AgentSpec
# ---------------------------------------------------------------------------

class SimpleExecutor(Executor):
    """Executor that wraps a :class:`Client` directly.

    This mirrors ``gluellm.SimpleExecutor``. All parameters accepted by
    :class:`Client` are forwarded verbatim.

    Parameters
    ----------
    api_key:
        OpenAI-compatible API key.
    model:
        Model to use (default ``"gpt-5.4-nano-2026-03-17-mini"``).
    system_prompt:
        Optional system prompt.
    hooks:
        Executor-level hooks (pre/post executor boundary).
    **client_kwargs:
        Any remaining keyword arguments are forwarded to :class:`Client`
        (e.g. ``max_retries``, ``requests_per_second``, ``timeout_secs``,
        ``pool_max_idle_per_host``, …).

    Examples
    --------
    >>> ex = SimpleExecutor(api_key="sk-...", system_prompt="You are a math tutor.")
    >>> result = ex.execute("What is the integral of x²?")
    >>> print(result.content)
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.4-nano-2026-03-17-mini",
        system_prompt: str | None = None,
        hooks: list[Callable] | None = None,
        on_status: Callable[[ProcessEvent], None] | None = None,
        sinks: list[Callable[[ProcessEvent], None]] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        status_emitter: StatusEmitter | None = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(hooks=hooks, on_status=on_status, sinks=sinks)
        emitter = _merge_status_emitter(
            status_emitter=status_emitter,
            on_status=on_status,
            sinks=sinks,
        )
        if reasoning_effort is not None:
            client_kwargs["reasoning_effort"] = reasoning_effort
        self._client = Client(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            status_emitter=emitter,
            **client_kwargs,
        )

    # delegate tool / hook / guardrail registration to the underlying client
    def register_tool(self, fn: Callable, /, **kwargs: Any) -> None:
        """Register a tool on the underlying :class:`Client`."""
        self._client.register_tool(fn, **kwargs)

    def register_hook(self, stage: str, handler: Callable, **kwargs: Any) -> None:
        """Register a pipeline hook on the underlying :class:`Client`."""
        self._client.register_hook(stage, handler, **kwargs)

    def register_guardrail(self, handler: Callable, **kwargs: Any) -> None:
        """Register a guardrail on the underlying :class:`Client`."""
        self._client.register_guardrail(handler, **kwargs)

    def _execute_internal(
        self,
        user_message: str,
        *,
        timeout_secs: int | None,
        connect_timeout_secs: int | None,
        request_id: str | None,
    ) -> CompletionOutcome:
        return self._client.complete(
            user_message,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def __repr__(self) -> str:
        return f"SimpleExecutor(hooks={len(self._hooks)})"


# ---------------------------------------------------------------------------
# AgentExecutor — wraps an Agent
# ---------------------------------------------------------------------------

class AgentExecutor(Executor):
    """Executor that drives a superglue :class:`Agent`.

    This mirrors ``gluellm.AgentExecutor``. The :class:`Agent` owns its own
    HTTP pool, tool registry, hooks, and guardrails.

    Parameters
    ----------
    agent:
        A pre-configured :class:`Agent` instance.
    hooks:
        Executor-level hooks (pre/post executor boundary).

    Examples
    --------
    >>> spec = AgentSpec(name="Analyst", persona="a data analyst")
    >>> agent = Agent(spec, api_key="sk-...")
    >>> agent.register_tool(query_database)
    >>> ex = AgentExecutor(agent=agent)
    >>> result = ex.execute("How many users signed up last week?")
    >>> print(result.content)
    """

    def __init__(self, *, agent: Agent, hooks: list[Callable] | None = None, on_status: Callable[[ProcessEvent], None] | None = None, sinks: list[Callable[[ProcessEvent], None]] | None = None) -> None:
        super().__init__(hooks=hooks, on_status=on_status, sinks=sinks)
        self._agent = agent

    @property
    def agent(self) -> Agent:
        """The underlying :class:`Agent`."""
        return self._agent

    def _execute_internal(
        self,
        user_message: str,
        *,
        timeout_secs: int | None,
        connect_timeout_secs: int | None,
        request_id: str | None,
    ) -> CompletionOutcome:
        return self._agent.run(
            user_message,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def __repr__(self) -> str:
        return (
            f"AgentExecutor(agent={self._agent!r}, hooks={len(self._hooks)})"
        )


# ---------------------------------------------------------------------------
# AgentStructuredExecutor — parses structured JSON output into a dataclass /
# dict, using the agent's compiled system prompt + tool loop.
# ---------------------------------------------------------------------------

class AgentStructuredExecutor(Executor, Generic[_StructuredT]):
    """Executor that returns parsed structured output alongside the raw text.

    The LLM is instructed to respond **only** with a JSON object matching
    ``response_schema``. If ``response_schema`` is a :mod:`dataclasses`
    dataclass or a class with a ``model_validate`` method (Pydantic), the
    raw JSON is automatically parsed into an instance. Otherwise the raw
    ``dict`` is returned in :attr:`CompletionOutcome.content` (as JSON
    string) and ``structured`` holds the parsed ``dict``.

    Parameters
    ----------
    agent:
        A pre-configured :class:`Agent` instance.
    response_schema:
        The expected output type. Accepts:

        * A Pydantic ``BaseModel`` subclass — parsed via ``model_validate``.
        * A Python dataclass — parsed via ``dataclass(**json_dict)``.
        * ``dict`` — returns the raw parsed dict.
    hooks:
        Executor-level hooks (pre/post executor boundary).

    Examples
    --------
    >>> import dataclasses
    >>> @dataclasses.dataclass
    ... class SentimentResult:
    ...     label: str          # "positive" | "neutral" | "negative"
    ...     confidence: float
    ...     reasoning: str
    ...
    >>> spec = AgentSpec(name="Sentiment", persona="a sentiment analysis expert")
    >>> agent = Agent(spec, api_key="sk-...")
    >>> ex = AgentStructuredExecutor(agent=agent, response_schema=SentimentResult)
    >>> result = ex.execute("The product is absolutely fantastic!")
    >>> print(result.structured)      # SentimentResult(label='positive', ...)
    >>> print(result.content)         # raw JSON string
    """

    def __init__(
        self,
        *,
        agent: Agent,
        response_schema: type[_StructuredT],
        hooks: list[Callable] | None = None,
        on_status: Callable[[ProcessEvent], None] | None = None,
        sinks: list[Callable[[ProcessEvent], None]] | None = None,
    ) -> None:
        super().__init__(hooks=hooks, on_status=on_status, sinks=sinks)
        self._agent = agent
        self._schema = response_schema

    @property
    def agent(self) -> Agent:
        return self._agent

    def execute(  # type: ignore[override]
        self,
        user_message: str,
        *,
        timeout_secs: int | None = None,
        connect_timeout_secs: int | None = None,
        request_id: str | None = None,
    ) -> StructuredOutcome[_StructuredT]:
        """Execute and return a :class:`StructuredOutcome`.

        Parameters
        ----------
        user_message:
            The user's message / query.
        timeout_secs:
            Per-call total request timeout override.
        connect_timeout_secs:
            Per-call TCP connect timeout override.
        request_id:
            Optional correlation ID. A UUID v4 is auto-generated when omitted.
            Returned on ``StructuredOutcome.request_id``.

        Returns
        -------
        StructuredOutcome[_StructuredT]
            ``.content``    — raw JSON string from the model.
            ``.structured`` — parsed instance of ``response_schema``.
            ``.rounds``     — tool-call rounds used.
            ``.usage``      — token usage dict or ``None``.
            ``.request_id`` — correlation ID for this request.
        """
        processed = self._run_hooks("pre_executor", user_message)
        raw = self._execute_internal(
            processed,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )
        # post hooks on raw JSON text
        hooked = self._run_hooks("post_executor", raw.content or "")
        parsed = self._parse(hooked)
        return StructuredOutcome(
            content=hooked,
            structured=parsed,
            rounds=raw.rounds,
            usage=raw.usage,
            request_id=getattr(raw, "request_id", ""),
        )

    def _execute_internal(
        self,
        user_message: str,
        *,
        timeout_secs: int | None,
        connect_timeout_secs: int | None,
        request_id: str | None,
    ) -> CompletionOutcome:
        schema_hint = _build_schema_hint(self._schema)
        augmented = (
            f"{user_message}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n{schema_hint}"
        )
        return self._agent.run(
            augmented,
            timeout_secs=timeout_secs,
            connect_timeout_secs=connect_timeout_secs,
            request_id=request_id,
        )

    def _parse(self, text: str) -> _StructuredT:
        # strip markdown fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
        data = json.loads(clean)
        # Pydantic model
        if hasattr(self._schema, "model_validate"):
            return self._schema.model_validate(data)  # type: ignore[return-value]
        # dataclass
        if dataclasses.is_dataclass(self._schema) and isinstance(self._schema, type):
            return self._schema(**data)  # type: ignore[return-value]
        # plain dict or other
        return data  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"AgentStructuredExecutor(agent={self._agent!r}, "
            f"schema={self._schema.__name__}, hooks={len(self._hooks)})"
        )


# ---------------------------------------------------------------------------
# StructuredOutcome — return value of AgentStructuredExecutor.execute()
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StructuredOutcome(Generic[_StructuredT]):
    """Return value of :meth:`AgentStructuredExecutor.execute`.

    Attributes
    ----------
    content:
        Raw JSON string from the model.
    structured:
        Parsed instance of the ``response_schema`` type.
    rounds:
        Number of tool-call rounds used.
    usage:
        Token usage dict (``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``) or ``None`` when the provider omits it.
    request_id:
        Correlation ID for this request (UUID v4 auto-generated or
        caller-supplied via ``execute(request_id=...)``.
    """

    content: str | None
    structured: _StructuredT | None
    rounds: int = 0
    usage: dict | None = None
    request_id: str = ""

    def __repr__(self) -> str:
        return (
            f"StructuredOutcome(structured={self.structured!r}, "
            f"rounds={self.rounds}, request_id={self.request_id!r})"
        )


# ---------------------------------------------------------------------------
# Schema hint builder
# ---------------------------------------------------------------------------

def _build_schema_hint(schema: type) -> str:
    """Return a compact JSON schema hint string for the given type."""
    # Pydantic
    if hasattr(schema, "model_json_schema"):
        return json.dumps(schema.model_json_schema(), indent=2)
    # dataclass — build a simple field map
    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        fields = {
            f.name: (f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type)))
            for f in dataclasses.fields(schema)
        }
        return json.dumps({"type": "object", "properties": fields}, indent=2)
    return '{"type": "object"}'


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def complete(
    prompt: str,
    /,
    *,
    api_key: str | None = None,
    model: str = "gpt-5.4-nano-2026-03-17-mini",
    base_url: str = "https://api.openai.com",
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    timeout_secs: int | None = None,
    connect_timeout_secs: int | None = None,
    request_id: str | None = None,
    **client_kwargs: Any,
) -> CompletionOutcome:
    """Run a one-shot chat completion without constructing a :class:`Client`.

    A throwaway :class:`Client` is created for this call. Any ``tools`` are
    registered on it before the request is sent. Pass extra keyword arguments
    to customise retry, rate-limiting, and connection-pool behaviour — they
    are forwarded verbatim to :class:`Client`.

    Parameters
    ----------
    prompt:
        User message to send.
    api_key:
        OpenAI-compatible API key. Falls back to the ``OPENAI_API_KEY``
        environment variable when ``None``.
    model:
        Model to use (default ``"gpt-5.4-nano-2026-03-17-mini"``).
    base_url:
        API base URL (default ``"https://api.openai.com"``).
    system_prompt:
        Optional system prompt prepended to every request.
    tools:
        List of Python callables to register as tools. Schema, name, and
        description are inferred automatically from type annotations and
        docstrings.
    timeout_secs:
        Total per-request timeout override for this call.
    connect_timeout_secs:
        TCP connect timeout override for this call.
    request_id:
        Optional correlation ID. A UUID v4 is auto-generated when omitted.
    **client_kwargs:
        Extra keyword arguments forwarded to :class:`Client` (e.g.
        ``max_retries``, ``requests_per_second``, ``pool_max_idle_per_host``).

    Returns
    -------
    CompletionOutcome
        ``.content``    — assistant text response (``str | None``)
        ``.rounds``     — number of LLM calls made
        ``.usage``      — token usage dict (``prompt``, ``completion``, ``total``)
        ``.request_id`` — correlation ID for this request

    Examples
    --------
    Simple one-shot completion::

        import superglue
        result = superglue.complete("What is the capital of France?")
        print(result.content)

    With tools::

        def get_weather(location: str) -> dict:
            '''Return current weather for *location*.'''
            return {"location": location, "temp_c": 18}

        result = superglue.complete(
            "What is the weather in Berlin?",
            tools=[get_weather],
        )
        print(result.content)
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = Client(
        api_key=resolved_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        **client_kwargs,
    )
    for fn in tools or []:
        client.register_tool(fn)
    return client.complete(
        prompt,
        timeout_secs=timeout_secs,
        connect_timeout_secs=connect_timeout_secs,
        request_id=request_id,
    )


def stream(
    prompt: str,
    /,
    *,
    on_token: Callable[[str], None],
    api_key: str | None = None,
    model: str = "gpt-5.4-nano-2026-03-17-mini",
    base_url: str = "https://api.openai.com",
    system_prompt: str | None = None,
    timeout_secs: int | None = None,
    connect_timeout_secs: int | None = None,
    request_id: str | None = None,
    **client_kwargs: Any,
) -> StreamOutcome:
    """Run a one-shot streaming completion without constructing a :class:`Client`.

    A throwaway :class:`Client` is created for this call. Each content token
    is delivered to *on_token* as it arrives from the model.

    Parameters
    ----------
    prompt:
        User message to send.
    on_token:
        Callable invoked for each content token string as it arrives.
    api_key:
        OpenAI-compatible API key. Falls back to the ``OPENAI_API_KEY``
        environment variable when ``None``.
    model:
        Model to use (default ``"gpt-5.4-nano-2026-03-17-mini"``).
    base_url:
        API base URL (default ``"https://api.openai.com"``).
    system_prompt:
        Optional system prompt prepended to every request.
    timeout_secs:
        Total per-request timeout override for this call.
    connect_timeout_secs:
        TCP connect timeout override for this call.
    request_id:
        Optional correlation ID. A UUID v4 is auto-generated when omitted.
    **client_kwargs:
        Extra keyword arguments forwarded to :class:`Client` (e.g.
        ``max_retries``, ``requests_per_second``).

    Returns
    -------
    StreamOutcome
        ``.content``       — full assembled response text
        ``.finish_reason`` — OpenAI finish reason (``"stop"``, ``"length"``, …)
        ``.usage``         — token usage dict (may be ``None`` for some models)
        ``.request_id``    — correlation ID for this request

    Examples
    --------
    Print tokens as they arrive::

        import superglue
        result = superglue.stream(
            "Tell me a short story.",
            on_token=lambda tok: print(tok, end="", flush=True),
        )
        print()  # newline after streaming ends
        print(f"Finish reason: {result.finish_reason}")
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = Client(
        api_key=resolved_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        **client_kwargs,
    )
    return client.stream(
        prompt,
        on_token=on_token,
        timeout_secs=timeout_secs,
        connect_timeout_secs=connect_timeout_secs,
        request_id=request_id,
    )


def batch(
    requests: list[str] | list[BatchRequest],
    /,
    *,
    api_key: str | None = None,
    model: str = "gpt-5.4-nano-2026-03-17-mini",
    base_url: str = "https://api.openai.com",
    system_prompt: str | None = None,
    max_concurrent: int = 5,
    error_strategy: str = "continue",
    timeout_secs: int | None = None,
    connect_timeout_secs: int | None = None,
    **client_kwargs: Any,
) -> BatchResponse:
    """Run multiple prompts concurrently without constructing a :class:`Client`.

    A throwaway :class:`Client` is created for this call. Plain strings are
    automatically wrapped in :class:`BatchRequest`.

    Parameters
    ----------
    requests:
        A list of plain strings **or** :class:`BatchRequest` objects.
    api_key:
        OpenAI-compatible API key. Falls back to the ``OPENAI_API_KEY``
        environment variable when ``None``.
    model:
        Model to use (default ``"gpt-5.4-nano-2026-03-17-mini"``).
    base_url:
        API base URL (default ``"https://api.openai.com"``).
    system_prompt:
        Optional system prompt prepended to every request.
    max_concurrent:
        Maximum number of simultaneous in-flight API calls (default 5).
    error_strategy:
        How to handle per-request failures:

        * ``"continue"`` *(default)* — include all results; failures have
          ``success=False`` and a non-empty ``error``.
        * ``"skip"`` — return only successful results; drop failures silently.
        * ``"fail_fast"`` — raise ``RuntimeError`` on the first failure.
    timeout_secs:
        Total per-request timeout override for every item in this batch.
    connect_timeout_secs:
        TCP connect timeout override for every item in this batch.
    **client_kwargs:
        Extra keyword arguments forwarded to :class:`Client` (e.g.
        ``max_retries``, ``requests_per_second``).

    Returns
    -------
    BatchResponse
        ``.results``        — list of :class:`BatchResult` (one per request)
        ``.total_requests`` — total input count (before any filtering)
        ``.successful``     — count of items with ``success=True``
        ``.failed``         — count of items with ``success=False``
        ``.elapsed_secs``   — total wall-clock time for the batch
        ``.total_usage``    — aggregated token usage dict (or ``None``)

    Examples
    --------
    Batch of plain strings::

        import superglue
        response = superglue.batch(
            ["What is 1+1?", "What is 2+2?", "What is 3+3?"],
            max_concurrent=3,
        )
        for r in response.results:
            print(r.content)

    Mixed :class:`BatchRequest` objects with per-item system prompts::

        from superglue import BatchRequest
        response = superglue.batch(
            [
                BatchRequest(prompt="Summarise this text.", id="req-1"),
                BatchRequest(
                    prompt="Translate to French.",
                    system_prompt="You are a professional translator.",
                ),
            ],
        )
        print(f"Successful: {response.successful}/{response.total_requests}")
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = Client(
        api_key=resolved_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        **client_kwargs,
    )
    return client.batch(
        requests,
        max_concurrent=max_concurrent,
        error_strategy=error_strategy,
        timeout_secs=timeout_secs,
        connect_timeout_secs=connect_timeout_secs,
    )


# ---------------------------------------------------------------------------
# GlueLLM-compatible async API (for migrating from gluellm)
# ---------------------------------------------------------------------------


_ALLOWED_CLIENT_KWARGS = frozenset({
    "max_tool_rounds",
    "max_retries",
    "retry_initial_delay_ms",
    "retry_max_delay_ms",
    "retry_multiplier",
    "requests_per_second",
    "timeout_secs",
    "connect_timeout_secs",
    "max_output_retries",
    "reasoning_effort",
    "pool_max_idle_per_host",
    "pool_idle_timeout_secs",
    "model_fallback_models",
})


def _filter_client_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if k in _ALLOWED_CLIENT_KWARGS}


def _strip_provider(model: str | None) -> str | None:
    """Strip ``provider:`` prefix (e.g. ``openai:gpt-5.4-nano-2026-03-17`` → ``gpt-5.4-nano-2026-03-17``)."""
    if model is None:
        return None
    if ":" in model:
        return model.split(":", 1)[1]
    return model


def _timeout_secs_from_any(timeout: Any) -> dict[str, Any]:
    if timeout is None:
        return {}
    try:
        return {"timeout_secs": int(float(timeout))}
    except (TypeError, ValueError):
        return {}


def _messages_to_prompt(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages)
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            parts.append(str(m))
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def _parse_structured_json(text: str, response_format: type[Any] | None) -> Any:
    if response_format is None:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    data = json.loads(clean)
    if hasattr(response_format, "model_validate"):
        return response_format.model_validate(data)
    if dataclasses.is_dataclass(response_format) and isinstance(response_format, type):
        return response_format(**data)
    return data


@dataclasses.dataclass
class EmbedResult:
    """Result of :meth:`GlueLLM.embed` (gluellm-compatible shape)."""

    embeddings: list[list[float]]


@dataclasses.dataclass
class CompleteResult:
    """Result of :meth:`GlueLLM.complete` (gluellm-compatible shape)."""

    final_response: str
    content: str


@dataclasses.dataclass
class GenerateResult:
    """Result of :meth:`GlueLLM.generate` (gluellm-compatible shape)."""

    text: str
    content: str


@dataclasses.dataclass
class StructuredResult:
    """Result of :meth:`GlueLLM.structured_complete` (gluellm-compatible shape)."""

    structured_output: Any
    final_response: str


class GlueLLM:
    """Async facade matching common **gluellm** ``GlueLLM`` call patterns.

    Embeddings use the OpenAI HTTP API via ``httpx``. Chat/completions use the
    Rust-backed :class:`Client` in a worker thread (``asyncio.to_thread``).

    Parameters
    ----------
    model:
        Chat model id; ``provider:model`` prefixes are stripped.
    embedding_model:
        Embedding model id for :meth:`embed`; defaults to ``text-embedding-3-small``.
    api_key:
        OpenAI-compatible API key, or ``OPENAI_API_KEY`` from the environment.
    base_url:
        API base URL (default ``https://api.openai.com``).
    system_prompt:
        Default system prompt for completions.
    **client_kwargs:
        Forwarded to :class:`Client` (retries, rate limits, pool settings, …).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        system_prompt: str | None = None,
        status_emitter: StatusEmitter | None = None,
        **client_kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "API key is required. Pass api_key=... or set OPENAI_API_KEY."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._default_model = _strip_provider(model) or "gpt-5.4-nano-2026-03-17-mini"
        self._embedding_model = _strip_provider(embedding_model) or "text-embedding-3-small"
        self._default_system_prompt = system_prompt
        if status_emitter is not None:
            client_kwargs["status_emitter"] = status_emitter
        self._client_kwargs = _filter_client_kwargs(client_kwargs)
        self._status_emitter = client_kwargs.get("status_emitter")
        self._client = Client(
            api_key=resolved_key,
            model=self._default_model,
            base_url=base_url,
            system_prompt=system_prompt,
            status_emitter=client_kwargs.get("status_emitter"),
            **{k: v for k, v in client_kwargs.items() if k != "status_emitter"},
        )
        self._embed_http: httpx.AsyncClient | None = None

    def _embed_client(self) -> httpx.AsyncClient:
        if self._embed_http is None or self._embed_http.is_closed:
            self._embed_http = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
        return self._embed_http

    async def aclose(self) -> None:
        """Close the shared embeddings HTTP client, if open."""
        if self._embed_http is not None:
            await self._embed_http.aclose()
            self._embed_http = None

    async def __aenter__(self) -> GlueLLM:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _client_for(self, *, model: str | None, system_prompt: str | None) -> Client:
        m = _strip_provider(model) if model else self._default_model
        sp = system_prompt if system_prompt is not None else self._default_system_prompt
        if m == self._default_model and sp == self._default_system_prompt:
            return self._client
        return Client(
            api_key=self._api_key,
            model=m or self._default_model,
            base_url=self._base_url,
            system_prompt=sp,
            status_emitter=self._status_emitter,
            **self._client_kwargs,
        )

    async def complete(self, *args: Any, **kwargs: Any) -> CompleteResult:
        """Run a chat completion (gluellm-compatible kwargs)."""
        if args:
            user_message = str(args[0])
        else:
            user_message = kwargs.pop("user_message", None)
            if user_message is None:
                raise TypeError("complete() requires user_message=... or a positional prompt string")
        kwargs.pop("temperature", None)
        kwargs.pop("max_tokens", None)
        timeout = kwargs.pop("timeout", None)
        model = kwargs.pop("model", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        # ignore remaining gluellm-only kwargs
        cli = self._client_for(model=model, system_prompt=None)
        outcome = await asyncio.to_thread(
            cli.complete,
            user_message,
            **_timeout_secs_from_any(timeout),
            reasoning_effort=reasoning_effort,
        )
        text = outcome.content or ""
        return CompleteResult(final_response=text, content=text)

    async def generate(self, *args: Any, **kwargs: Any) -> GenerateResult:
        """Single-shot text generation (maps to :meth:`Client.complete`)."""
        if args:
            prompt = str(args[0])
        else:
            prompt = str(kwargs.pop("prompt", ""))
        kwargs.pop("temperature", None)
        kwargs.pop("max_tokens", None)
        timeout = kwargs.pop("timeout", None)
        outcome = await asyncio.to_thread(
            self._client.complete,
            prompt,
            **_timeout_secs_from_any(timeout),
        )
        text = outcome.content or ""
        return GenerateResult(text=text, content=text)

    async def chat(
        self,
        messages: Any = None,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return an OpenAI-shaped object with ``choices[0].message.content``."""
        _ = max_tokens, kwargs
        prompt = _messages_to_prompt(messages)
        cli = self._client_for(model=model, system_prompt=None)
        outcome = await asyncio.to_thread(cli.complete, prompt)
        text = (outcome.content or "").strip()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    async def embed(self, texts: str | list[str], **kwargs: Any) -> EmbedResult:
        """OpenAI-compatible embeddings (HTTP, not the Rust core)."""
        _ = kwargs
        if isinstance(texts, str):
            inputs: list[str] = [texts]
        else:
            inputs = list(texts)
        url = f"{self._base_url}/v1/embeddings"
        http = self._embed_client()
        r = await http.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._embedding_model, "input": inputs},
        )
        r.raise_for_status()
        payload = r.json()
        rows = sorted(payload["data"], key=lambda d: d["index"])
        return EmbedResult(embeddings=[row["embedding"] for row in rows])

    async def structured_complete(
        self,
        *,
        user_message: str | None = None,
        response_format: Any = None,
        system_prompt: str | None = None,
        model: str | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> StructuredResult:
        _ = kwargs
        if user_message is None:
            raise TypeError("structured_complete() requires user_message=...")
        cli = self._client_for(model=model, system_prompt=system_prompt)
        augmented = user_message
        if response_format is not None:
            augmented = (
                f"{user_message}\n\nRespond ONLY with a valid JSON object matching this schema:\n"
                f"{_build_schema_hint(response_format)}"
            )
        outcome = await asyncio.to_thread(
            cli.complete,
            augmented,
            **_timeout_secs_from_any(timeout),
        )
        text = outcome.content or ""
        parsed = _parse_structured_json(text, response_format)
        return StructuredResult(structured_output=parsed, final_response=text)


async def structured_complete(
    *,
    user_message: str | None = None,
    response_format: Any = None,
    system_prompt: str | None = None,
    model: str | None = None,
    timeout: Any = None,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com",
    **kwargs: Any,
) -> StructuredResult:
    """Async module-level structured completion (gluellm ``api.structured_complete`` compatible)."""
    for k in list(kwargs):
        if k in (
            "correlation_id",
            "connect_timeout",
            "request_timeout",
            "execute_tools",
            "max_tool_iterations",
        ):
            kwargs.pop(k, None)
    extra = _filter_client_kwargs(kwargs)
    llm = GlueLLM(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        **extra,
    )
    return await llm.structured_complete(
        user_message=user_message,
        response_format=response_format,
        system_prompt=system_prompt,
        model=model,
        timeout=timeout,
    )


GlueLLM.response = GlueLLM.complete
GlueLLM.structured_response = GlueLLM.structured_complete
