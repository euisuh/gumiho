# agent-injection-bench

Red-team benchmark measuring how susceptible tool-using LLM agents are to
indirect prompt injection (malicious instructions embedded in tool output —
web pages, RAG docs, file contents) and how well common mitigations hold up.

Status: scaffolding. See PLAN.md once generated.

## Leaderboard

<!-- leaderboard:start -->
| Model | Defense | ASR (95% CI) | Benign utility | Utility under attack | ΔASR | ΔUtility | Detector TPR | Detector FPR | Errors | Est. USD | Cost overhead | Latency overhead | ASR seed spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-v1 | delimiting | 52.1% (38.3%–65.5%) | 12.5% | 4.2% | -10.4% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -9.7% | — (1 seed) |
| mock-v1 | egress_filter | 62.5% (48.4%–74.8%) | 12.5% | 0.0% | +0.0% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -11.4% | — (1 seed) |
| mock-v1 | llm_detector | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 100.0% | 100.0% | 0 | $0.0000 | +0.0% | -36.7% | — (1 seed) |
| mock-v1 | none | 62.5% (48.4%–74.8%) | 12.5% | 0.0% | +0.0% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | +0.0% | — (1 seed) |
| mock-v1 | spotlight_datamark | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -40.4% | — (1 seed) |
| mock-v1 | stack | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 100.0% | 100.0% | 0 | $0.0000 | +0.0% | -33.4% | — (1 seed) |
<!-- leaderboard:end -->

## Quickstart

```sh
git clone git@github.com:euisuh/agent-injection-bench.git
cd agent-injection-bench
uv sync
uv run pytest
uv run python -m bench.run --scenario exfil_email_canary --model mock --defense none --track attack --out results/runs/smoke.jsonl
```
