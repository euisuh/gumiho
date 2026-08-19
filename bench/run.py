from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bench.agent import run_agent
from bench.defenses import DEFENSES
from bench.loader import load_scenario, load_scenarios


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run agent injection benchmark scenario")
    result.add_argument("--scenario", default="all", help="scenario id")
    result.add_argument("--model", default="mock", help="model alias")
    result.add_argument("--defense", choices=(*DEFENSES, "all"), default="none", help="defense name")
    result.add_argument("--track", choices=("attack", "benign"), default="attack", help="run track")
    result.add_argument("--out", type=Path, default=Path("results/runs/run.jsonl"), help="output JSONL path")
    result.add_argument("--limit", type=int, help="maximum runs")
    result.add_argument("--seed", type=int, default=0, help="random seed")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        return 0
    scenarios = load_scenarios() if args.scenario == "all" else [load_scenario(args.scenario)]
    defenses = list(DEFENSES) if args.defense == "all" else [args.defense]
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for scenario in scenarios:
            for defense in defenses:
                result = run_agent(scenario, args.model, args.track, defense, args.seed)
                file.write(json.dumps(asdict(result), separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
