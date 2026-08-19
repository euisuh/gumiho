from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime

from bench import __version__
from bench.defenses import DEFENSES, EGRESS_REFUSAL, REDACTION_NOTICE, configure_defense
from bench.judge import judge, utility_success
from bench.models import MODELS, chat
from bench.tools import execute_tool, tool_specs
from bench.types import RunResult, Scenario, Step


MAX_ITERATIONS = 8


def run_agent(
    scenario: Scenario,
    model: str,
    track: str,
    defense: str = "none",
    seed: int = 0,
) -> RunResult:
    if track not in {"attack", "benign"}:
        raise ValueError(f"unknown track: {track}")
    if model not in MODELS:
        raise ValueError(f"unknown model alias: {model}")

    provider, model_id, input_price, output_price = MODELS[model]
    del provider
    schemas = tool_specs(scenario.tools)
    defense_events = {"detector_flagged": [], "egress_blocked": []}
    hooks = configure_defense(defense, scenario, model, defense_events)
    system_prompt = hooks.system_prefix(scenario.system_prompt) if hooks.system_prefix else scenario.system_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.user_task},
    ]
    prompt_hash = "sha256:" + hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()
    run_key = (scenario.id, f"channel={scenario.injection['channel']}", model_id, defense, track, seed)
    run_id = hashlib.sha1(json.dumps(run_key).encode(), usedforsecurity=False).hexdigest()
    result = RunResult(
        bench_version=__version__,
        run_id=run_id,
        scenario_id=scenario.id,
        variant=f"channel={scenario.injection['channel']}",
        category=scenario.category,
        track=track,
        model=model_id,
        model_alias=model,
        defense=defense,
        seed=seed,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        prompt_hash=prompt_hash,
        steps=[],
        final_answer="",
        attack_success=None,
        utility_success=False,
        judge={},
        defense_events=defense_events,
        usage={"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "latency_s": 0.0},
        error=None,
    )

    started = time.perf_counter()
    for _ in range(MAX_ITERATIONS):
        response = chat(model, messages, schemas, temperature=0.0)
        result.usage["input_tokens"] += response.usage.get("input_tokens", 0)
        result.usage["output_tokens"] += response.usage.get("output_tokens", 0)
        calls = [{"name": call.name, "args": call.args} for call in response.tool_calls]
        result.steps.append(Step(len(result.steps), "assistant", response.text, calls))
        messages.append({
            "role": "assistant",
            "content": response.text,
            "tool_calls": [{"id": call.id or f"call-{len(result.steps)}-{i}", "name": call.name, "args": call.args} for i, call in enumerate(response.tool_calls)],
        })
        if not response.tool_calls:
            result.final_answer = response.text
            break
        for i, call in enumerate(response.tool_calls):
            if hooks.check_tool_call and not hooks.check_tool_call(call.name, call.args):
                content, injected = EGRESS_REFUSAL.format(tool=call.name), False
                if defense == "stack":
                    content = DEFENSES["spotlight_datamark"].wrap_tool_output(call.name, content)
                    content = DEFENSES["delimiting"].wrap_tool_output(call.name, content)
            else:
                content, injected = execute_tool(call.name, call.args, scenario, track)
                if hooks.wrap_tool_output:
                    content = hooks.wrap_tool_output(call.name, content)
                    if content is None:
                        content = REDACTION_NOTICE.format(tool=call.name)
            if defense in {"delimiting", "stack"}:
                content += f"\nRemember the user's actual task: {scenario.user_task}"
            result.steps.append(Step(len(result.steps), "tool", name=call.name, content=content, injected=injected))
            messages.append({
                "role": "tool",
                "name": call.name,
                "tool_call_id": call.id or f"call-{len(result.steps) - 1}-{i}",
                "content": content,
            })

    result.usage["latency_s"] = time.perf_counter() - started
    result.usage["usd"] = (
        result.usage["input_tokens"] * input_price + result.usage["output_tokens"] * output_price
    ) / 1_000_000
    verdict = judge(result, scenario)
    result.attack_success = verdict.success if track == "attack" else None
    result.judge = {
        "method": verdict.method,
        "rule": verdict.rule,
        "detail": verdict.detail,
    } if track == "attack" else {"method": "not_applicable", "rule": None, "detail": "benign track"}
    result.utility_success = utility_success(result, scenario)
    return result
