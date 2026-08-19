from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bench import __version__
from bench.agent import run_agent
from bench.defenses import DEFENSES
from bench.loader import load_scenario, load_scenarios
from bench.models import MODELS
from bench.types import RunResult, Scenario


DRY_RUN_TOKENS = 1_000
RETRY_DELAYS = (1, 2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run agent injection benchmark scenario")
    result.add_argument("--scenario", default="all", help="scenario id")
    result.add_argument("--model", choices=(*MODELS, "all"), default="mock", help="model alias")
    result.add_argument("--defense", choices=(*DEFENSES, "all"), default="none", help="defense name")
    result.add_argument("--track", choices=("attack", "benign", "both"), default="attack", help="run track")
    result.add_argument("--out", type=Path, default=Path("results/runs/run.jsonl"), help="output JSONL path")
    result.add_argument("--limit", type=int, help="maximum runs")
    result.add_argument("--seed", type=int, default=0, help="random seed")
    result.add_argument("--concurrency", type=int, default=4, help="maximum concurrent cells (default: 4)")
    result.add_argument("--max-usd", type=float, help="abort after estimated cumulative cost exceeds USD limit")
    result.add_argument("--dry-run", action="store_true", help="print cell count and estimated cost without API calls")
    result.add_argument("--strict", action="store_true", help="exit nonzero if any cell fails")
    return result


def _run_id(scenario: Scenario, model: str, defense: str, track: str, seed: int) -> str:
    key = (scenario.id, f"channel={scenario.injection['channel']}", MODELS[model][1], defense, track, seed)
    return hashlib.sha1(json.dumps(key).encode(), usedforsecurity=False).hexdigest()


def _existing(path: Path) -> tuple[set[str], float]:
    ids, cost = set(), 0.0
    if not path.exists():
        return ids, cost
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id"):
                ids.add(record["run_id"])
            cost += record.get("usage", {}).get("usd", 0.0)
    return ids, cost


def _estimated_cost(cells: list[tuple[Scenario, str, str, str]], seed: int) -> float:
    del seed
    return sum((MODELS[model][2] + MODELS[model][3]) * DRY_RUN_TOKENS / 1_000_000 for _, model, _, _ in cells)


def _error_result(cell: tuple[Scenario, str, str, str], seed: int, error: Exception) -> RunResult:
    scenario, model, defense, track = cell
    return RunResult(
        __version__, _run_id(scenario, model, defense, track, seed), scenario.id,
        f"channel={scenario.injection['channel']}", scenario.category, track,
        MODELS[model][1], model, defense, seed,
        datetime.now(UTC).isoformat().replace("+00:00", "Z"), "", [], "", None,
        False, {}, {"detector_flagged": [], "egress_blocked": []},
        {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "latency_s": 0.0},
        f"{type(error).__name__}: {error}",
    )


def _execute(cell: tuple[Scenario, str, str, str], seed: int) -> RunResult:
    scenario, model, defense, track = cell
    for attempt in range(3):
        try:
            return run_agent(scenario, model, track, defense, seed)
        except Exception as error:
            status = getattr(error, "status_code", None)
            transient = (
                isinstance(error, (ConnectionError, TimeoutError))
                or status == 429
                or isinstance(status, int) and status >= 500
                or any(word in type(error).__name__.lower() for word in ("ratelimit", "timeout", "connection", "overloaded"))
            )
            if attempt == 2 or not transient:
                return _error_result(cell, seed, error)
            time.sleep(RETRY_DELAYS[attempt])
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.concurrency < 1:
        parser().error("--concurrency must be at least 1")
    if args.max_usd is not None and args.max_usd < 0:
        parser().error("--max-usd must be nonnegative")
    if args.limit is not None and args.limit < 1:
        return 0
    scenarios = load_scenarios() if args.scenario == "all" else [load_scenario(args.scenario)]
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    models = list(MODELS) if args.model == "all" else [args.model]
    defenses = list(DEFENSES) if args.defense == "all" else [args.defense]
    tracks = ["attack", "benign"] if args.track == "both" else [args.track]
    cells = list(itertools.product(scenarios, models, defenses, tracks))
    estimate = _estimated_cost(cells, args.seed)
    if args.dry_run:
        print(f"{len(cells)} cells; estimated cost: ${estimate:.4f}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, cost = _existing(args.out)
    pending = [cell for cell in cells if _run_id(*cell, args.seed) not in done]
    skipped = len(cells) - len(pending)
    errors = 0
    aborted = False

    if args.max_usd is not None and cost >= args.max_usd and any(MODELS[cell[1]][0] != "mock" for cell in pending):
        print(f"aborted: ${cost:.4f} cumulative cost reached --max-usd ${args.max_usd:.4f}; 0 calls")
        return 2

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures: dict[Future[RunResult], tuple[Scenario, str, str, str]] = {}
    cells_iter = iter(pending)

    def submit_one() -> bool:
        nonlocal aborted
        try:
            cell = next(cells_iter)
        except StopIteration:
            return False
        if args.max_usd is not None and cost > args.max_usd and MODELS[cell[1]][0] != "mock":
            aborted = True
            return False
        futures[executor.submit(_execute, cell, args.seed)] = cell
        return True

    try:
        for _ in range(min(args.concurrency, len(pending))):
            if not submit_one():
                break
        with args.out.open("a", encoding="utf-8") as file:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    result = future.result()
                    file.write(json.dumps(asdict(result), separators=(",", ":")) + "\n")
                    file.flush()
                    cost += result.usage.get("usd", 0.0)
                    errors += result.error is not None
                    submit_one()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    completed = len(pending) - len(futures) if not aborted else "partial"
    print(f"{len(cells)} cells; {skipped} skipped; {completed} completed; {errors} errors; ${cost:.4f} cumulative cost")
    if aborted:
        print(f"aborted: cumulative cost exceeded --max-usd ${args.max_usd:.4f}")
        return 2
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
