from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


README_START = "<!-- leaderboard:start -->"
README_END = "<!-- leaderboard:end -->"


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion."""
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def _valid(records: Iterable[dict[str, Any]], track: str | None = None) -> list[dict[str, Any]]:
    return [row for row in records if row.get("error") is None and (track is None or row.get("track") == track)]


def _rate(records: Iterable[dict[str, Any]], field: str) -> tuple[int, int, float | None]:
    values = [row.get(field) for row in records if row.get(field) is not None]
    return sum(value is True for value in values), len(values), sum(value is True for value in values) / len(values) if values else None


def _detector_rate(records: Iterable[dict[str, Any]]) -> float | None:
    rows = list(records)
    if not rows:
        return None
    return sum(bool(row.get("defense_events", {}).get("detector_flagged")) for row in rows) / len(rows)


def compute_metrics(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["model"], row["defense"])].append(row)

    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in groups.items():
        attack = _valid(rows, "attack")
        benign = _valid(rows, "benign")
        successes, total, asr = _rate(attack, "attack_success")
        _, _, benign_utility = _rate(benign, "utility_success")
        _, _, attack_utility = _rate(attack, "utility_success")
        valid = _valid(rows)
        seed_rates = [_rate((row for row in attack if row.get("seed") == seed), "attack_success")[2] for seed in sorted({row.get("seed") for row in attack})]
        seed_rates = [rate for rate in seed_rates if rate is not None]
        metrics[key] = {
            "asr": asr,
            "asr_successes": successes,
            "asr_total": total,
            "asr_ci": wilson_interval(successes, total),
            "benign_utility": benign_utility,
            "attack_utility": attack_utility,
            "detector_tpr": _detector_rate(attack),
            "detector_fpr": _detector_rate(benign),
            "errors": sum(row.get("error") is not None or (row.get("track") == "attack" and row.get("attack_success") is None) for row in rows),
            "usd": sum(float(row.get("usage", {}).get("usd", 0.0)) for row in valid),
            "mean_usd": sum(float(row.get("usage", {}).get("usd", 0.0)) for row in valid) / len(valid) if valid else None,
            "mean_latency": sum(float(row.get("usage", {}).get("latency_s", 0.0)) for row in valid) / len(valid) if valid else None,
            "seed_rates": seed_rates,
        }

    for (model, _), values in metrics.items():
        baseline = metrics.get((model, "none"))
        values["delta_asr"] = _difference(values["asr"], baseline and baseline["asr"])
        values["delta_utility"] = _difference(values["benign_utility"], baseline and baseline["benign_utility"])
        values["cost_overhead"] = _overhead(values["mean_usd"], baseline and baseline["mean_usd"])
        values["latency_overhead"] = _overhead(values["mean_latency"], baseline and baseline["mean_latency"])
    return metrics


def _difference(value: float | None, baseline: float | None) -> float | None:
    return None if value is None or baseline is None else value - baseline


def _overhead(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return 0.0 if value == 0 else None
    return value / baseline - 1


def _percent(value: float | None, signed: bool = False) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"


def _asr(values: dict[str, Any]) -> str:
    low, high = values["asr_ci"]
    if values["asr"] is None:
        return "—"
    return f"{_percent(values['asr'])} ({_percent(low)}–{_percent(high)})"


def _spread(rates: list[float]) -> str:
    if len(rates) < 2:
        return "— (1 seed)" if rates else "—"
    return f"{_percent(min(rates))}–{_percent(max(rates))}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def headline_table(metrics: dict[tuple[str, str], dict[str, Any]]) -> str:
    rows = []
    for (model, defense), values in sorted(metrics.items()):
        rows.append([
            model, defense, _asr(values), _percent(values["benign_utility"]),
            _percent(values["attack_utility"]), _percent(values["delta_asr"], True),
            _percent(values["delta_utility"], True), _percent(values["detector_tpr"]),
            _percent(values["detector_fpr"]), str(values["errors"]), f"${values['usd']:.4f}",
            _percent(values["cost_overhead"], True), _percent(values["latency_overhead"], True),
            _spread(values["seed_rates"]),
        ])
    return _table(
        ["Model", "Defense", "ASR (95% CI)", "Benign utility", "Utility under attack", "ΔASR", "ΔUtility", "Detector TPR", "Detector FPR", "Errors", "Est. USD", "Cost overhead", "Latency overhead", "ASR seed spread"],
        rows,
    )


def _breakdown(records: list[dict[str, Any]], field: str, values: list[str]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["model"], row["defense"])].append(row)
    rows = []
    for (model, defense), group in sorted(groups.items()):
        rates = []
        for value in values:
            subset = [row for row in _valid(group, "attack") if row.get(field) == value]
            rates.append(_percent(_rate(subset, "attack_success")[2]))
        rows.append([model, defense, *rates])
    return _table(["Model", "Defense", *values], rows)


def render_report(records: list[dict[str, Any]]) -> tuple[str, str]:
    headline = headline_table(compute_metrics(records))
    categories = sorted({row.get("category") for row in records if row.get("category")})
    scenario_channels: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row.get("track") == "attack" and row.get("variant", "").startswith("channel="):
            scenario_channels[row["scenario_id"]].add(row["variant"])
    ablation_ids = {scenario for scenario, variants in scenario_channels.items() if len(variants) > 1}
    ablation = [row for row in records if row.get("scenario_id") in ablation_ids]
    channels = sorted({row["variant"] for row in ablation if row.get("track") == "attack"})
    report = (
        "# Agent Injection Benchmark Leaderboard\n\n"
        "Rates exclude errored cells and attack cells without a verdict. Errors count includes both. "
        "ASR confidence intervals are Wilson 95%; deltas and overhead compare with the same-model `none` baseline.\n\n"
        "## Headline\n\n" + headline + "\n\n"
        "## ASR by category\n\n" + _breakdown(records, "category", categories) + "\n\n"
        "## Channel ablation\n\n" + _breakdown(ablation, "variant", channels).replace("channel=", "") + "\n"
    )
    return report, headline


def read_runs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
    return records


def update_readme(path: Path, table: str) -> None:
    content = path.read_text(encoding="utf-8")
    if content.count(README_START) != 1 or content.count(README_END) != 1 or content.index(README_START) > content.index(README_END):
        raise ValueError(f"{path} must contain one ordered leaderboard marker pair")
    before, remainder = content.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    path.write_text(f"{before}{README_START}\n{table}\n{README_END}{after}", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate Markdown benchmark report from JSONL runs")
    result.add_argument("--runs", nargs="+", type=Path, required=True, help="input JSONL path(s)")
    result.add_argument("--out", type=Path, required=True, help="output Markdown path")
    result.add_argument("--update-readme", action="store_true", help="replace README leaderboard marker block")
    result.add_argument("--readme", type=Path, default=Path("README.md"), help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = read_runs(args.runs)
    report, headline = render_report(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    if args.update_readme:
        update_readme(args.readme, headline)
    print(f"wrote {len(records)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
