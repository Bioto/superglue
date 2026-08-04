from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessEventKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LLM_CALL_START: _ClassVar[ProcessEventKind]
    LLM_CALL_END: _ClassVar[ProcessEventKind]
    LLM_CALL_ERROR: _ClassVar[ProcessEventKind]
    TOOL_CALL_START: _ClassVar[ProcessEventKind]
    TOOL_CALL_END: _ClassVar[ProcessEventKind]
    TOOL_ROUTE: _ClassVar[ProcessEventKind]
    REASONING_DELTA: _ClassVar[ProcessEventKind]
LLM_CALL_START: ProcessEventKind
LLM_CALL_END: ProcessEventKind
LLM_CALL_ERROR: ProcessEventKind
TOOL_CALL_START: ProcessEventKind
TOOL_CALL_END: ProcessEventKind
TOOL_ROUTE: ProcessEventKind
REASONING_DELTA: ProcessEventKind

class ToolSpec(_message.Message):
    __slots__ = ("name", "parameters_schema_json", "description", "static_tool")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_SCHEMA_JSON_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    STATIC_TOOL_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameters_schema_json: str
    description: str
    static_tool: bool
    def __init__(self, name: _Optional[str] = ..., parameters_schema_json: _Optional[str] = ..., description: _Optional[str] = ..., static_tool: _Optional[bool] = ...) -> None: ...

class FunctionCall(_message.Message):
    __slots__ = ("name", "arguments")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    arguments: str
    def __init__(self, name: _Optional[str] = ..., arguments: _Optional[str] = ...) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ("id", "kind", "function")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    function: FunctionCall
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., function: _Optional[_Union[FunctionCall, _Mapping]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "tool_calls", "tool_call_id", "name", "refusal", "provider_blocks_json")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_BLOCKS_JSON_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    tool_call_id: str
    name: str
    refusal: str
    provider_blocks_json: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ..., tool_call_id: _Optional[str] = ..., name: _Optional[str] = ..., refusal: _Optional[str] = ..., provider_blocks_json: _Optional[str] = ...) -> None: ...

class ChatOptions(_message.Message):
    __slots__ = ("base_url", "api_key", "model", "max_tool_rounds", "system_prompt", "temperature", "top_p", "max_completion_tokens", "presence_penalty", "frequency_penalty", "stop", "parallel_tool_calls", "logprobs", "top_logprobs", "seed", "store", "service_tier", "reasoning_effort", "reasoning_summary", "extra_json", "tool_mode", "tool_route_model", "condense_tool_messages", "aaak_tool_condensing", "summarize_context_enabled", "summarize_context_threshold", "summarize_context_keep_recent", "aaak_compression_enabled", "aaak_compression_model")
    BASE_URL_FIELD_NUMBER: _ClassVar[int]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MAX_TOOL_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    MAX_COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    STORE_FIELD_NUMBER: _ClassVar[int]
    SERVICE_TIER_FIELD_NUMBER: _ClassVar[int]
    REASONING_EFFORT_FIELD_NUMBER: _ClassVar[int]
    REASONING_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    EXTRA_JSON_FIELD_NUMBER: _ClassVar[int]
    TOOL_MODE_FIELD_NUMBER: _ClassVar[int]
    TOOL_ROUTE_MODEL_FIELD_NUMBER: _ClassVar[int]
    CONDENSE_TOOL_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    AAAK_TOOL_CONDENSING_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZE_CONTEXT_ENABLED_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZE_CONTEXT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZE_CONTEXT_KEEP_RECENT_FIELD_NUMBER: _ClassVar[int]
    AAAK_COMPRESSION_ENABLED_FIELD_NUMBER: _ClassVar[int]
    AAAK_COMPRESSION_MODEL_FIELD_NUMBER: _ClassVar[int]
    base_url: str
    api_key: str
    model: str
    max_tool_rounds: int
    system_prompt: str
    temperature: float
    top_p: float
    max_completion_tokens: int
    presence_penalty: float
    frequency_penalty: float
    stop: str
    parallel_tool_calls: bool
    logprobs: bool
    top_logprobs: int
    seed: int
    store: bool
    service_tier: str
    reasoning_effort: str
    reasoning_summary: str
    extra_json: str
    tool_mode: str
    tool_route_model: str
    condense_tool_messages: bool
    aaak_tool_condensing: bool
    summarize_context_enabled: bool
    summarize_context_threshold: int
    summarize_context_keep_recent: int
    aaak_compression_enabled: bool
    aaak_compression_model: str
    def __init__(self, base_url: _Optional[str] = ..., api_key: _Optional[str] = ..., model: _Optional[str] = ..., max_tool_rounds: _Optional[int] = ..., system_prompt: _Optional[str] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., max_completion_tokens: _Optional[int] = ..., presence_penalty: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., stop: _Optional[str] = ..., parallel_tool_calls: _Optional[bool] = ..., logprobs: _Optional[bool] = ..., top_logprobs: _Optional[int] = ..., seed: _Optional[int] = ..., store: _Optional[bool] = ..., service_tier: _Optional[str] = ..., reasoning_effort: _Optional[str] = ..., reasoning_summary: _Optional[str] = ..., extra_json: _Optional[str] = ..., tool_mode: _Optional[str] = ..., tool_route_model: _Optional[str] = ..., condense_tool_messages: _Optional[bool] = ..., aaak_tool_condensing: _Optional[bool] = ..., summarize_context_enabled: _Optional[bool] = ..., summarize_context_threshold: _Optional[int] = ..., summarize_context_keep_recent: _Optional[int] = ..., aaak_compression_enabled: _Optional[bool] = ..., aaak_compression_model: _Optional[str] = ...) -> None: ...

class CompletionRequest(_message.Message):
    __slots__ = ("messages", "options")
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    options: ChatOptions
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., options: _Optional[_Union[ChatOptions, _Mapping]] = ...) -> None: ...

class Usage(_message.Message):
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., total_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., reasoning_tokens: _Optional[int] = ...) -> None: ...

class CompletionOutcome(_message.Message):
    __slots__ = ("content", "rounds", "usage", "finish_reason")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ROUNDS_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    content: str
    rounds: int
    usage: Usage
    finish_reason: str
    def __init__(self, content: _Optional[str] = ..., rounds: _Optional[int] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class ResponsesRequest(_message.Message):
    __slots__ = ("messages", "options", "previous_response_id", "instructions", "input_json")
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    INPUT_JSON_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    options: ChatOptions
    previous_response_id: str
    instructions: str
    input_json: str
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., options: _Optional[_Union[ChatOptions, _Mapping]] = ..., previous_response_id: _Optional[str] = ..., instructions: _Optional[str] = ..., input_json: _Optional[str] = ...) -> None: ...

class ResponsesOutcome(_message.Message):
    __slots__ = ("response_id", "content", "rounds", "usage", "request_id", "model_used", "raw_output_json", "messages")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ROUNDS_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_USED_FIELD_NUMBER: _ClassVar[int]
    RAW_OUTPUT_JSON_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    content: str
    rounds: int
    usage: Usage
    request_id: str
    model_used: str
    raw_output_json: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, response_id: _Optional[str] = ..., content: _Optional[str] = ..., rounds: _Optional[int] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ..., request_id: _Optional[str] = ..., model_used: _Optional[str] = ..., raw_output_json: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class HookEvent(_message.Message):
    __slots__ = ("stage", "content", "metadata", "request_id", "timestamp_ms")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STAGE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    stage: str
    content: str
    metadata: _containers.ScalarMap[str, str]
    request_id: str
    timestamp_ms: int
    def __init__(self, stage: _Optional[str] = ..., content: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., request_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class RunRecord(_message.Message):
    __slots__ = ("request_id", "messages", "hook_events", "outcome", "started_at_ms", "finished_at_ms", "process_events", "options_json", "model_used", "previous_response_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    HOOK_EVENTS_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESS_EVENTS_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_JSON_FIELD_NUMBER: _ClassVar[int]
    MODEL_USED_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    hook_events: _containers.RepeatedCompositeFieldContainer[HookEvent]
    outcome: CompletionOutcome
    started_at_ms: int
    finished_at_ms: int
    process_events: _containers.RepeatedCompositeFieldContainer[ProcessEvent]
    options_json: str
    model_used: str
    previous_response_id: str
    def __init__(self, request_id: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., hook_events: _Optional[_Iterable[_Union[HookEvent, _Mapping]]] = ..., outcome: _Optional[_Union[CompletionOutcome, _Mapping]] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., process_events: _Optional[_Iterable[_Union[ProcessEvent, _Mapping]]] = ..., options_json: _Optional[str] = ..., model_used: _Optional[str] = ..., previous_response_id: _Optional[str] = ...) -> None: ...

class ProcessEvent(_message.Message):
    __slots__ = ("kind", "request_id", "round", "model", "tool_call_count", "usage", "estimated_cost_usd", "error_type", "timestamp_ms", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    KIND_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_COUNT_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_COST_USD_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    kind: ProcessEventKind
    request_id: str
    round: int
    model: str
    tool_call_count: int
    usage: Usage
    estimated_cost_usd: float
    error_type: str
    timestamp_ms: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, kind: _Optional[_Union[ProcessEventKind, str]] = ..., request_id: _Optional[str] = ..., round: _Optional[int] = ..., model: _Optional[str] = ..., tool_call_count: _Optional[int] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ..., estimated_cost_usd: _Optional[float] = ..., error_type: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StreamChunk(_message.Message):
    __slots__ = ("delta", "usage", "finish_reason", "reasoning_delta")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    REASONING_DELTA_FIELD_NUMBER: _ClassVar[int]
    delta: str
    usage: Usage
    finish_reason: str
    reasoning_delta: str
    def __init__(self, delta: _Optional[str] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ..., finish_reason: _Optional[str] = ..., reasoning_delta: _Optional[str] = ...) -> None: ...
