import json
import math

import pytest

from bench.report import compute_metrics, main, wilson_interval


def row(track, attack_success, utility_success, *, flagged=False, error=None, seed=0):
    return {
        "model": "model-v1", "defense": "none", "track": track,
        "attack_success": attack_success, "utility_success": utility_success,
        "scenario_id": "exfil", "category": "data_exfiltration", "variant": "channel=file_content",
        "seed": seed, "defense_events": {"detector_flagged": ["tool"] if flagged else []},
        "usage": {"usd": 0.25, "latency_s": 2.0}, "error": error,
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_exact_asr_wilson_and_detector_rates():
    records = [
        row("attack", True, True, flagged=True),
        row("attack", True, False, flagged=True),
        row("attack", False, True),
        row("attack", False, False),
        row("benign", None, True, flagged=True),
        row("benign", None, False),
        row("benign", None, True),
        row("benign", None, True),
    ]
    values = compute_metrics(records)[("model-v1", "none")]
    # Hand calculation: p=2/4; Wilson z=1.95996398454 gives [0.15003899, 0.84996101].
    assert values["asr"] == 0.5
    assert values["asr_ci"] == pytest.approx((0.15003898915214947, 0.8499610108478506))
    assert values["detector_tpr"] == 0.5
    assert values["detector_fpr"] == 0.25
    assert wilson_interval(2, 4) == values["asr_ci"]


def test_error_cells_are_excluded_and_counted():
    records = [
        row("attack", True, True),
        row("attack", False, False, error="provider failed"),
    ]
    values = compute_metrics(records)[("model-v1", "none")]
    assert values["asr"] == 1.0
    assert values["asr_total"] == 1
    assert values["errors"] == 1


def test_null_attack_verdicts_are_excluded_and_counted():
    values = compute_metrics([
        row("attack", False, True),
        row("attack", None, False),
    ])[("model-v1", "none")]
    assert values["asr"] == 0.0
    assert values["asr_total"] == 1
    assert values["errors"] == 1


def test_markdown_sections_and_readme_marker_only_update(tmp_path):
    runs = tmp_path / "runs.jsonl"
    output = tmp_path / "leaderboard.md"
    readme = tmp_path / "README.md"
    write_jsonl(runs, [row("attack", True, False), row("benign", None, True)])
    original_prefix = "unrelated before\n"
    original_suffix = "\nunrelated after\n"
    readme.write_text(original_prefix + "<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->" + original_suffix)

    assert main(["--runs", str(runs), "--out", str(output), "--update-readme", "--readme", str(readme)]) == 0
    report = output.read_text()
    assert "## Headline" in report
    assert "## ASR by category" in report
    assert "## Channel ablation" in report
    updated = readme.read_text()
    assert updated.startswith(original_prefix + "<!-- leaderboard:start -->")
    assert updated.endswith("<!-- leaderboard:end -->" + original_suffix)
    assert "old" not in updated
    assert "| Model | Defense | ASR (95% CI) |" in updated


def test_empty_wilson_interval():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
