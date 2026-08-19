# Agent Injection Benchmark Leaderboard

Rates exclude errored cells and attack cells without a verdict. Errors count includes both. ASR confidence intervals are Wilson 95%; deltas and overhead compare with the same-model `none` baseline.

## Headline

| Model | Defense | ASR (95% CI) | Benign utility | Utility under attack | ΔASR | ΔUtility | Detector TPR | Detector FPR | Errors | Est. USD | Cost overhead | Latency overhead | ASR seed spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-v1 | delimiting | 52.1% (38.3%–65.5%) | 12.5% | 4.2% | -10.4% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -9.7% | — (1 seed) |
| mock-v1 | egress_filter | 62.5% (48.4%–74.8%) | 12.5% | 0.0% | +0.0% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -11.4% | — (1 seed) |
| mock-v1 | llm_detector | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 100.0% | 100.0% | 0 | $0.0000 | +0.0% | -36.7% | — (1 seed) |
| mock-v1 | none | 62.5% (48.4%–74.8%) | 12.5% | 0.0% | +0.0% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | +0.0% | — (1 seed) |
| mock-v1 | spotlight_datamark | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 0.0% | 0.0% | 0 | $0.0000 | +0.0% | -40.4% | — (1 seed) |
| mock-v1 | stack | 0.0% (0.0%–7.4%) | 12.5% | 12.5% | -62.5% | +0.0% | 100.0% | 100.0% | 0 | $0.0000 | +0.0% | -33.4% | — (1 seed) |

## ASR by category

| Model | Defense | content_manipulation | data_exfiltration | denial_of_service | unauthorized_action |
| --- | --- | --- | --- | --- | --- |
| mock-v1 | delimiting | 0.0% | 78.6% | 30.0% | 78.6% |
| mock-v1 | egress_filter | 0.0% | 92.9% | 30.0% | 100.0% |
| mock-v1 | llm_detector | 0.0% | 0.0% | 0.0% | 0.0% |
| mock-v1 | none | 0.0% | 92.9% | 30.0% | 100.0% |
| mock-v1 | spotlight_datamark | 0.0% | 0.0% | 0.0% | 0.0% |
| mock-v1 | stack | 0.0% | 0.0% | 0.0% | 0.0% |

## Channel ablation

| Model | Defense | api_json_field | file_content | rag_document | tool_error_message | web_search_result |
| --- | --- | --- | --- | --- | --- | --- |
| mock-v1 | delimiting | 0.0% | 66.7% | 66.7% | 66.7% | 66.7% |
| mock-v1 | egress_filter | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% |
| mock-v1 | llm_detector | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| mock-v1 | none | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% |
| mock-v1 | spotlight_datamark | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| mock-v1 | stack | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
