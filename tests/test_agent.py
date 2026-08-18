import json
from dataclasses import fields

from bench import __version__
from bench.agent import MAX_ITERATIONS, run_agent
from bench.loader import load_scenario
from bench.models import set_mock_responses
from bench.run import main
from bench.types import ChatResponse, RunResult, ToolCall


def test_attack_runs_end_to_end(tmp_path):
    output = tmp_path / "smoke.jsonl"
    assert main([
        "--scenario", "exfil_email_canary", "--model", "mock", "--defense", "none",
        "--track", "attack", "--out", str(output),
    ]) == 0
    lines = output.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert set(data) == {field.name for field in fields(RunResult)}
    assert data["bench_version"] == __version__
    assert data["attack_success"] is True
    assert any(step.get("injected") is True for step in data["steps"])


def test_benign_omits_injection(tmp_path):
    output = tmp_path / "benign.jsonl"
    main(["--scenario", "exfil_email_canary", "--model", "mock", "--track", "benign", "--out", str(output)])
    data = json.loads(output.read_text())
    assert data["attack_success"] is None
    assert not any(step.get("injected") is True for step in data["steps"])
    assert data["utility_success"] is True


def test_scripted_mock_and_iteration_cap():
    scenario = load_scenario("exfil_email_canary")
    set_mock_responses([ChatResponse("", [ToolCall("read_email", {})])] * MAX_ITERATIONS)
    result = run_agent(scenario, "mock", "benign")
    assert len([step for step in result.steps if step.role == "assistant"]) == MAX_ITERATIONS
