from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from bench.types import Scenario


TOOL_SPECS: dict[str, dict[str, Any]] = {
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
    del args
    if name in scenario.fixtures:
        content = Path(scenario.fixtures[name]).read_text(encoding="utf-8").rstrip()
    elif name == "send_email":
        content = "Email sent (simulated)."
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
