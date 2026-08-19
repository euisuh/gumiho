import json
import re
from dataclasses import replace

from bench.agent import run_agent
from bench.loader import load_scenario
from bench.models import set_mock_responses
from bench.tools import INJECTION_CHANNELS, execute_tool
from bench.types import ChatResponse, ToolCall


def variants():
    scenario = load_scenario("exfil_email_canary")
    return [
        replace(scenario, injection={**scenario.injection, "channel": channel})
        for channel in INJECTION_CHANNELS
    ]


def payload(scenario):
    return re.sub(
        r"\{([A-Za-z_]\w*)\}",
        lambda match: str(scenario.injection["vars"][match.group(1)]),
        scenario.payload.template,
    ).strip()


def test_payload_is_identical_and_each_channel_envelope_differs():
    outputs = [execute_tool("read_email", {}, scenario, "attack")[0] for scenario in variants()]
    expected = payload(variants()[0])
    assert len(set(outputs)) == 5
    assert expected in outputs[0]
    assert expected in outputs[1]
    assert expected in outputs[2]
    assert expected in json.loads(outputs[3])["data"]["content"]
    assert expected in outputs[4]


def test_json_channel_and_tool_error_use_tool_result_path():
    api, error = variants()[3:5]
    api_output, injected = execute_tool("read_email", {}, api, "attack")
    assert injected
    assert payload(api) in json.loads(api_output)["data"]["content"]

    result = run_agent(api, "mock", "attack")
    assert result.error is None

    set_mock_responses([
        ChatResponse("", [ToolCall("read_email", {})]),
        ChatResponse("done"),
    ])
    result = run_agent(error, "mock", "attack")
    tool_step = next(step for step in result.steps if step.role == "tool")
    assert tool_step.injected
    assert "tool_call_status: failed" in tool_step.content
    assert "error_type: UpstreamToolError" in tool_step.content
    assert payload(error) in tool_step.content
