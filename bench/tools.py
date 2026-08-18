from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from bench.types import Scenario


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "fetch_url": {
        "name": "fetch_url",
        "description": "Fetch a URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a local file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    "write_file": {
        "name": "write_file",
        "description": "Write a local file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    },
    "read_email": {
        "name": "read_email",
        "description": "Read the newest email matching the user's request.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "send_email": {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
            "additionalProperties": False,
        },
    },
    "search_docs": {
        "name": "search_docs",
        "description": "Search internal documents.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "run_query": {
        "name": "run_query",
        "description": "Run a database query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "create_calendar_event": {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "attendees": {"type": "string"}}, "required": ["title", "attendees"]},
    },
    "submit_order": {
        "name": "submit_order",
        "description": "Submit a purchase order.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["item", "quantity"]},
    },
}


def tool_specs(names: list[str]) -> list[dict[str, Any]]:
    try:
        return [TOOL_SPECS[name] for name in names]
    except KeyError as error:
        raise ValueError(f"unknown tool: {error.args[0]}") from error


def _insert(content: str, payload: str, placement: str) -> str:
    if placement == "start":
        return f"{payload}\n{content}"
    if placement == "end":
        return f"{content}\n{payload}"
    lines = content.splitlines()
    middle = len(lines) // 2
    return "\n".join(lines[:middle] + [payload] + lines[middle:])


def execute_tool(name: str, args: dict[str, Any], scenario: Scenario, track: str) -> tuple[str, bool]:
    if name in scenario.fixtures:
        content = Path(scenario.fixtures[name]).read_text(encoding="utf-8").rstrip()
    elif name in {"write_file", "send_email", "create_calendar_event", "submit_order"}:
        content = f"{name} completed (simulated)."
    elif name in TOOL_SPECS:
        content = f"{name} returned no results for {args!r}."
    else:
        raise ValueError(f"unknown or unconfigured tool: {name}")

    injected = track == "attack" and name == scenario.injection["target_tool"]
    if injected:
        variables = scenario.injection.get("vars", {})
        payload = re.sub(
            r"\{([A-Za-z_]\w*)\}",
            lambda match: str(variables[match.group(1)]),
            scenario.payload.template,
        ).strip()
        content = _insert(content, payload, scenario.injection["placement"])
    return content, injected
