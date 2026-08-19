# Methodology

## Run protocol

- Model snapshots: `claude-sonnet-4-20250514`, `gpt-4.1-2025-04-14`; offline control: `mock-v1`.
- Temperature: `0.0`.
- Maximum agent iterations: `8`.
- Default concurrency: `4` threads.
- Dry-run estimate assumption: 1,000 input and 1,000 output tokens per cell. Recorded cost uses provider-reported token usage and the `MODELS` price table.
- Each JSONL line is flushed after writing. Existing `run_id` values are skipped when a run resumes.

## Recorded-run commands

Real M8 sweep numbers, dates, and final cost are pending. Use these exact commands for that run:

```console
uv run python -m bench.run --scenario all --model all --defense all --track both --dry-run
uv run python -m bench.run --scenario all --model all --defense all --track both --concurrency 4 --max-usd 150 --out results/runs/v0.1.0.jsonl
uv run python -m bench.report --runs results/runs/v0.1.0.jsonl --out results/leaderboard.md
```
