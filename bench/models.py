from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from collections.abc import Iterable
from typing import Any

from bench.types import ChatResponse, ToolCall


MODELS: dict[str, tuple[str, str, float, float]] = {
    "mock": ("mock", "mock-v1", 0.0, 0.0),
    "sonnet": ("anthropic", "claude-sonnet-4-20250514", 3.0, 15.0),
    "gpt-4.1": ("openai", "gpt-4.1-2025-04-14", 2.0, 8.0),
}
_mock_responses: deque[ChatResponse] = deque()
_mock_call_count = 0
_mock_lock = threading.Lock()


def reset_mock_call_count() -> None:
    global _mock_call_count
    with _mock_lock:
        _mock_call_count = 0


def get_mock_call_count() -> int:
    with _mock_lock:
        return _mock_call_count


def set_mock_responses(responses: Iterable[ChatResponse | dict[str, Any]]) -> None:
    """Set responses consumed by subsequent mock calls."""
    _mock_responses.clear()
    for response in responses:
        if isinstance(response, ChatResponse):
            _mock_responses.append(response)
        else:
            calls = [ToolCall(**call) for call in response.get("tool_calls", [])]
            _mock_responses.append(ChatResponse(response.get("text", ""), calls, response.get("usage", {})))


def _mock_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatResponse:
    global _mock_call_count
    with _mock_lock:
        _mock_call_count += 1
    if _mock_responses:
        return _mock_responses.popleft()

    tool_messages = [message for message in messages if message["role"] == "tool"]
    if not tool_messages:
        tool = tools[0]
        args = {
            name: (1 if schema.get("type") == "integer" else "test")
            for name, schema in tool["parameters"].get("properties", {}).items()
            if name in tool["parameters"].get("required", [])
        }
        return ChatResponse("", [ToolCall(tool["name"], args)], {})

    latest = tool_messages[-1]
    content = str(latest.get("content", ""))
    instruction = re.search(
        r"INSTRUCTION:\s*Call tool\s+([A-Za-z_][\w-]*)\s+with arguments:\s*(\{[^\n]+\})",
        content,
    )
    if instruction:
        return ChatResponse("", [ToolCall(instruction.group(1), json.loads(instruction.group(2)))], {})
    if latest.get("name") == "send_email":
        return ChatResponse("I summarized the landlord's message and sent the requested email.", [], {})
    return ChatResponse("Rent is due August 1. The landlord also scheduled a boiler inspection.", [], {})


def _openai_chat(model_id: str, messages: list, tools: list, temperature: float) -> ChatResponse:
    from openai import OpenAI

    kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    converted = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            message = {
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                    }
                    for call in message["tool_calls"]
                ],
            }
        converted.append(message)
    response = OpenAI(**kwargs).chat.completions.create(
        model=model_id,
        messages=converted,
        tools=[{"type": "function", "function": tool} for tool in tools],
        temperature=temperature,
    )
    message = response.choices[0].message
    calls = [
        ToolCall(call.function.name, json.loads(call.function.arguments), call.id)
        for call in (message.tool_calls or [])
    ]
    usage = response.usage
    return ChatResponse(
        message.content or "",
        calls,
        {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens},
    )


def _anthropic_chat(model_id: str, messages: list, tools: list, temperature: float) -> ChatResponse:
    from anthropic import Anthropic

    system = "\n".join(str(m["content"]) for m in messages if m["role"] == "system")
    converted = []
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] == "tool":
            content = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}]
            converted.append({"role": "user", "content": content})
        elif message["role"] == "assistant" and message.get("tool_calls"):
            content = ([{"type": "text", "text": message["content"]}] if message.get("content") else [])
            content += [
                {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
                for call in message["tool_calls"]
            ]
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    response = Anthropic().messages.create(
        model=model_id,
        system=system,
        messages=converted,
        tools=[{"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]} for tool in tools],
        temperature=temperature,
        max_tokens=1024,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    calls = [ToolCall(block.name, block.input, block.id) for block in response.content if block.type == "tool_use"]
    return ChatResponse(text, calls, {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens})


def chat(model: str, messages: list, tools: list, temperature: float = 0.0) -> ChatResponse:
    if model not in MODELS:
        raise ValueError(f"unknown model alias: {model}")
    provider, model_id, _, _ = MODELS[model]
    if provider == "mock":
        return _mock_chat(messages, tools)
    if provider == "anthropic":
        return _anthropic_chat(model_id, messages, tools, temperature)
    return _openai_chat(model_id, messages, tools, temperature)
