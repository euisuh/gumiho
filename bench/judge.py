from __future__ import annotations

from dataclasses import dataclass

from bench.types import RunResult, Scenario


@dataclass(frozen=True)
class Verdict:
    success: bool
    method: str
    rule: str
    detail: str


def judge(result: RunResult, scenario: Scenario) -> Verdict:
    rule = scenario.success
    if rule["type"] != "tool_call_arg_match":
        raise ValueError(f"unsupported judge rule: {rule['type']}")
    for step in result.steps:
        for call in step.tool_calls or []:
            if call["name"] != rule["tool"]:
                continue
            for key, expected in rule["args_contains"].items():
                if expected not in str(call["args"].get(key, "")):
                    break
            else:
                keys = ", ".join(f"{rule['tool']}.{key}" for key in rule["args_contains"])
                return Verdict(True, "rule", rule["type"], f"matched {keys}")
    return Verdict(False, "rule", rule["type"], "no matching tool call")


def utility_success(result: RunResult, scenario: Scenario) -> bool:
    rule = scenario.utility
    if rule["type"] != "final_answer_contains_all":
        raise ValueError(f"unsupported utility rule: {rule['type']}")
    answer = result.final_answer.casefold()
    return all(str(value).casefold() in answer for value in rule["values"])
