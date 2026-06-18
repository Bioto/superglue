"""Function introspection helpers: annotations + docstring → JSON Schema tool spec.

Supports two function styles:

  # Style A — explicit typed parameters (recommended)
  def get_weather(location: str, unit: str = "celsius") -> dict:
      '''Get the current weather.

      Args:
          location: City name, e.g. 'Paris'
          unit: One of celsius or fahrenheit
      '''

  # Style B — single args dict (legacy; schema must be supplied separately)
  def get_weather(args: dict) -> dict:
      ...
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import re
import types
import typing
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints


# ---------------------------------------------------------------------------
# Type annotation → JSON Schema
# ---------------------------------------------------------------------------

def _annotation_to_schema(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X]  /  X | None
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {}

    # Literal["a", "b"]
    if origin is Literal:
        values = list(args)
        types_in_literal = {type(v) for v in values}
        json_type = "string"
        if types_in_literal == {int}:
            json_type = "integer"
        elif types_in_literal == {float}:
            json_type = "number"
        return {"type": json_type, "enum": values}

    # list / List[X]
    if origin is list:
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _annotation_to_schema(args[0])
        return schema

    # dict / Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Annotated[X, "description"] — extract inner type; description handled later
    if origin is typing.Annotated:
        return _annotation_to_schema(args[0])

    # Bare types
    _MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    if annotation in _MAP:
        return {"type": _MAP[annotation]}

    return {}


# ---------------------------------------------------------------------------
# Docstring parsing (Google style)
# ---------------------------------------------------------------------------

def _parse_docstring(fn: Callable) -> tuple[str, dict[str, str]]:
    """Return ``(description, {param_name: description})`` from a Google-style docstring."""
    raw = inspect.getdoc(fn) or ""
    if not raw:
        return "", {}

    lines = raw.splitlines()
    # First non-empty line(s) up to the Args: section = description
    desc_lines: list[str] = []
    in_args = False
    param_descs: dict[str, str] = {}
    current_param: str | None = None

    args_header = re.compile(r"^\s*Args\s*:\s*$", re.IGNORECASE)
    param_line = re.compile(r"^\s{4}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)")
    continuation = re.compile(r"^\s{8,}(.*)")
    section_header = re.compile(r"^\s*\w[\w\s]*:\s*$")

    for line in lines:
        if args_header.match(line):
            in_args = True
            continue
        if in_args:
            if section_header.match(line) and not param_line.match(line):
                # Another section started
                in_args = False
                current_param = None
                continue
            m = param_line.match(line)
            if m:
                current_param = m.group(1)
                param_descs[current_param] = m.group(2).strip()
                continue
            m2 = continuation.match(line)
            if m2 and current_param:
                param_descs[current_param] = (
                    param_descs[current_param] + " " + m2.group(1).strip()
                ).strip()
                continue
            current_param = None
        else:
            desc_lines.append(line)

    # Description = everything before Args: section, stripped
    desc = "\n".join(desc_lines).strip()
    # Use only the first paragraph as short description
    first_para = desc.split("\n\n")[0].replace("\n", " ").strip()
    return first_para, param_descs


# ---------------------------------------------------------------------------
# Annotated[X, "description"] helper
# ---------------------------------------------------------------------------

def _annotated_description(annotation: Any) -> str | None:
    """If annotation is ``Annotated[X, 'desc']``, return the string metadata."""
    if get_origin(annotation) is typing.Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, str):
                return meta
    return None


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def is_legacy_style(fn: Callable) -> bool:
    """Return True if fn takes a single ``args: dict`` parameter (legacy style)."""
    try:
        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        if len(params) == 1 and params[0].name == "args":
            ann = params[0].annotation
            if ann is inspect.Parameter.empty or ann is dict:
                return True
    except (ValueError, TypeError):
        pass
    return False


def schema_from_fn(fn: Callable) -> tuple[str, str, dict]:
    """Return ``(name, description, parameters_schema)`` inferred from *fn*.

    Works for both style A (typed kwargs) and style B (legacy ``args: dict``).
    For style B the returned schema is a bare ``{"type": "object"}`` — callers
    should supply an explicit schema in that case.
    """
    name = fn.__name__
    description, param_descs = _parse_docstring(fn)
    if not description:
        description = name.replace("_", " ").strip()

    if is_legacy_style(fn):
        # Legacy: can't introspect parameter names from args dict
        return name, description, {"type": "object"}

    # Typed parameters
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Return annotation not a property
        if pname == "return":
            continue

        annotation = hints.get(pname, param.annotation)

        # Is the param optional (has a default)?
        has_default = param.default is not inspect.Parameter.empty

        # Handle Optional[X] / X | None → not required
        origin = get_origin(annotation)
        inner_args = get_args(annotation)
        is_optional_union = (
            origin is types.UnionType or origin is typing.Union
        ) and type(None) in inner_args

        prop = _annotation_to_schema(annotation)

        # Per-param description: first from Annotated metadata, then docstring
        inline_desc = _annotated_description(annotation)
        doc_desc = param_descs.get(pname, "")
        prop_desc = inline_desc or doc_desc
        if prop_desc:
            prop["description"] = prop_desc

        properties[pname] = prop

        if not has_default and not is_optional_union:
            required.append(pname)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return name, description, schema


def _run_coroutine_blocking(*, factory: Callable[[], Any]) -> Any:
    """Run ``factory()`` (must return an awaitable) from sync Rust/tool threads.

    Uses :func:`asyncio.run` when no loop is running; if a loop is already running
    (rare for tool callbacks), runs the factory in a worker thread with its own
    event loop so the coroutine is created and consumed on the same thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    def _in_thread() -> Any:
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


def wrap_for_rust(fn: Callable) -> Callable:
    """Wrap *fn* so it always receives and returns a plain ``dict``.

    - If *fn* is already legacy-style (``fn(args: dict) -> dict``): no-op.
    - If *fn* takes typed parameters: calls ``fn(**args)`` and returns the result.
    - If *fn* is a coroutine function, the coroutine is run to completion before returning.
    """
    if is_legacy_style(fn):
        return fn

    if inspect.iscoroutinefunction(fn):

        def _wrapper_async(args: dict) -> Any:
            return _run_coroutine_blocking(factory=lambda: fn(**args))

        return _wrapper_async

    def _wrapper(args: dict) -> Any:
        return fn(**args)

    return _wrapper
