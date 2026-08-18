from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Payload:
    id: str
    goal: str
    template: str


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    title: str
    user_task: str
    system_prompt: str
    tools: list[str]
    fixtures: dict[str, str]
    injection: dict[str, Any]
    success: dict[str, Any]
    utility: dict[str, Any]
    payload: Payload


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)


@dataclass
class Step:
    i: int
    role: str
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    content: str | None = None
    injected: bool | None = None


@dataclass
class RunResult:
    bench_version: str
    run_id: str
    scenario_id: str
    variant: str
    category: str
    track: str
    model: str
    model_alias: str
    defense: str
    seed: int
    timestamp: str
    prompt_hash: str
    steps: list[Step]
    final_answer: str
    attack_success: bool | None
    utility_success: bool
    judge: dict[str, Any]
    defense_events: dict[str, list[str]]
    usage: dict[str, float]
    error: str | None
