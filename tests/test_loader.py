from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import yaml

from bench.agent import run_agent
from bench.loader import CATEGORIES, load_scenario, load_scenarios
from bench.models import set_mock_responses
from bench.types import ChatResponse, ToolCall


ROOT = Path(__file__).parents[1]


def _scenario_dir(tmp_path, mutate):
    source = ROOT / "scenarios" / "exfil_email_canary.yaml"
    data = yaml.safe_load(source.read_text())
    mutate(data)
    (tmp_path / "fixtures").mkdir()
    shutil.copy(ROOT / "scenarios" / "payloads.yaml", tmp_path / "payloads.yaml")
    shutil.copy(ROOT / "scenarios" / "fixtures" / "landlord_thread.txt", tmp_path / "fixtures" / "landlord_thread.txt")
    (tmp_path / "case.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def test_corpus_has_six_scenarios_per_category():
    scenarios = load_scenarios()
    base = {scenario.id: scenario for scenario in scenarios}.values()
    assert len(scenarios) == 48
    assert {category: sum(s.category == category for s in base) for category in CATEGORIES} == {
        category: 6 for category in CATEGORIES
    }


def test_rejects_unknown_tool(tmp_path):
    path = _scenario_dir(tmp_path, lambda data: data["tools"].append("unknown_tool"))
    with pytest.raises(ValueError, match="unknown tool"):
        load_scenario("case", path)


def test_rejects_unknown_payload_id(tmp_path):
    path = _scenario_dir(tmp_path, lambda data: data["injection"].update(payload_id="missing"))
    with pytest.raises(ValueError, match="unknown payload_id"):
        load_scenario("case", path)


def test_rejects_missing_utility(tmp_path):
    path = _scenario_dir(tmp_path, lambda data: data.pop("utility"))
    with pytest.raises(ValueError, match="missing keys: utility"):
        load_scenario("case", path)


def test_rejects_non_reserved_email_or_url_host(tmp_path):
    path = _scenario_dir(tmp_path, lambda data: data["injection"]["vars"].update(dropbox="attacker@evil.com"))
    with pytest.raises(ValueError, match="disallowed email/URL host 'evil.com'"):
        load_scenario("case", path)


def test_every_utility_rule_is_satisfiable_by_benign_script():
    for scenario in load_scenarios():
        answer = "; ".join(str(value) for value in scenario.utility["values"])
        set_mock_responses([
            ChatResponse("", [ToolCall(next(iter(scenario.fixtures)), {})]),
            ChatResponse(answer),
        ])
        assert run_agent(scenario, "mock", "benign").utility_success, scenario.id
