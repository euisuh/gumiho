from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from bench.agent import run_agent
from bench.defenses import DEFENSES
from bench.loader import load_scenarios


LABELS = Path("results/labels/judge_labels.jsonl")
REPORT = Path("results/judge_validation.md")


def _record(result, scenario) -> dict:
    return {
        "run_id": result.run_id,
        "scenario_id": result.scenario_id,
        "variant": result.variant,
        "category": result.category,
        "model": result.model,
        "model_alias": result.model_alias,
        "defense": result.defense,
        "user_task": scenario.user_task,
        "final_answer": result.final_answer,
        "steps": [asdict(step) for step in result.steps],
        "judge_verdict": result.attack_success,
        "judge": result.judge,
        "human_label": None,
    }


def stratified_sample(records: list[dict], n: int, fields: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        buckets[tuple(record[field] for field in fields)].append(record)
    rng = random.Random(0)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    keys = sorted(buckets, key=lambda key: tuple(str(value) for value in key))
    while len(selected) < n:
        added = False
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
                added = True
        if not added:
            break
    if len(selected) != n:
        raise ValueError(f"requested {n} records, only {len(selected)} available")
    return selected


def sample(n: int, fields: list[str], out: Path) -> None:
    candidates = [
        _record(run_agent(scenario, "mock", "attack", defense), scenario)
        for scenario in load_scenarios()
        for defense in DEFENSES
    ]
    llm_quota = round(n * 0.4)
    records = stratified_sample(
        [record for record in candidates if record["judge"]["method"].startswith("llm_judge")],
        llm_quota,
        fields,
    )
    selected = {record["run_id"] for record in records}
    records += stratified_sample(
        [record for record in candidates if record["run_id"] not in selected],
        n - len(records),
        fields,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} unlabelled records to {out}")


def compute_metrics(records: list[dict]) -> dict:
    labelled = [
        record for record in records
        if isinstance(record.get("human_label"), bool)
        and isinstance(record.get("judge_verdict"), bool)
    ]
    if not labelled:
        raise ValueError("no hand-labelled records with boolean judge verdicts")
    matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    disagreements = []
    for record in labelled:
        human, judge = record["human_label"], record["judge_verdict"]
        matrix[{(False, False): "tn", (False, True): "fp", (True, False): "fn", (True, True): "tp"}[(human, judge)]] += 1
        if human != judge:
            disagreements.append(record["run_id"])
    n = len(labelled)
    agreement = (matrix["tn"] + matrix["tp"]) / n
    human_positive = matrix["fn"] + matrix["tp"]
    judge_positive = matrix["fp"] + matrix["tp"]
    expected = (
        human_positive * judge_positive
        + (n - human_positive) * (n - judge_positive)
    ) / (n * n)
    kappa = 1.0 if expected == 1.0 and agreement == 1.0 else (agreement - expected) / (1.0 - expected)
    return {"n": n, "agreement": agreement, "kappa": kappa, "matrix": matrix, "disagreements": disagreements}


def report(records: list[dict]) -> str:
    metrics = compute_metrics(records)
    matrix = metrics["matrix"]
    disagreements = "\n".join(f"- `{run_id}`" for run_id in metrics["disagreements"]) or "None."
    return f"""# Judge validation

- n labelled: {metrics['n']}
- raw agreement: {metrics['agreement']:.3f}
- Cohen's kappa: {metrics['kappa']:.3f}

## Confusion matrix

Rows are human labels; columns are judge verdicts.

| | Judge false | Judge true |
|---|---:|---:|
| Human false | {matrix['tn']} | {matrix['fp']} |
| Human true | {matrix['fn']} | {matrix['tp']} |

## Disagreements

{disagreements}
"""


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Sample and validate benchmark judge verdicts")
    result.add_argument("--sample", type=int, help="number of unlabelled records to write")
    result.add_argument("--stratify", default="category,defense,verdict", help="comma-separated strata: category, defense, verdict")
    result.add_argument("--out", type=Path, default=LABELS, help="unlabelled JSONL output")
    result.add_argument("--report", action="store_true", help="generate agreement report from hand labels")
    result.add_argument("--labels", type=Path, default=LABELS, help="hand-labelled JSONL input")
    result.add_argument("--report-out", type=Path, default=REPORT, help="Markdown report output")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.report:
        text = report(_read(args.labels))
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
        print(f"wrote judge validation report to {args.report_out}")
        return 0
    if args.sample is None or args.sample < 1:
        parser().error("--sample must be a positive integer unless --report is used")
    aliases = {"category": "category", "defense": "defense", "verdict": "judge_verdict"}
    requested = args.stratify.split(",")
    if any(field not in aliases for field in requested):
        parser().error("--stratify fields must be category, defense, or verdict")
    sample(args.sample, [aliases[field] for field in requested], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
