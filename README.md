# agent-injection-bench

Red-team benchmark measuring how susceptible tool-using LLM agents are to
indirect prompt injection (malicious instructions embedded in tool output —
web pages, RAG docs, file contents) and how well common mitigations hold up.

Status: scaffolding. See PLAN.md once generated.

## Quickstart

```sh
git clone git@github.com:euisuh/agent-injection-bench.git
cd agent-injection-bench
uv sync
uv run pytest
uv run python -m bench.run --scenario exfil_email_canary --model mock --defense none --track attack --out results/runs/smoke.jsonl
```
