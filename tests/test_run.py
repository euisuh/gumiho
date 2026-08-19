import json
import threading
import time

import pytest

import bench.run as runner
from bench.models import get_mock_call_count, reset_mock_call_count


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dry_run_expands_matrix_without_calls(capsys):
    reset_mock_call_count()
    assert runner.main(["--scenario", "all", "--model", "all", "--defense", "all", "--track", "both", "--dry-run"]) == 0
    assert "1728 cells; estimated cost: $" in capsys.readouterr().out
    assert get_mock_call_count() == 0


def test_full_mock_sweep_resumes(tmp_path, capsys):
    output = tmp_path / "runs.jsonl"
    args = ["--scenario", "all", "--model", "mock", "--defense", "all", "--track", "both", "--out", str(output)]
    assert runner.main(args) == 0
    count = len(output.read_text().splitlines())
    assert count == 576
    assert runner.main(args) == 0
    assert len(output.read_text().splitlines()) == count
    assert "576 skipped" in capsys.readouterr().out


def test_interrupted_sweep_resumes_without_duplicates(tmp_path, monkeypatch):
    output = tmp_path / "runs.jsonl"
    real = runner.run_agent
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "run_agent", interrupt)
    args = ["--scenario", "all", "--model", "mock", "--defense", "none", "--track", "attack", "--concurrency", "1", "--out", str(output)]
    with pytest.raises(KeyboardInterrupt):
        runner.main(args)
    monkeypatch.setattr(runner, "run_agent", real)
    assert runner.main(args) == 0
    ids = [record["run_id"] for record in records(output)]
    assert len(ids) == len(set(ids)) == 48


def test_persistent_failure_records_all_cells(tmp_path, monkeypatch):
    output = tmp_path / "errors.jsonl"
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    monkeypatch.setattr(runner, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(runner, "run_agent", fail)
    args = ["--scenario", "all", "--model", "mock", "--defense", "none", "--track", "attack", "--out", str(output)]
    assert runner.main(args) == 0
    assert len(records(output)) == 48
    assert calls == 48 * 3
    assert all(record["error"] and record["attack_success"] is None for record in records(output))
    assert runner.main([*args, "--strict", "--out", str(tmp_path / "strict.jsonl")]) == 1


def test_concurrency_is_bounded(tmp_path, monkeypatch):
    real = runner.run_agent
    lock = threading.Lock()
    active = peak = 0

    def measured(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(runner, "run_agent", measured)
    assert runner.main(["--scenario", "exfil_email_canary", "--model", "mock", "--defense", "all", "--concurrency", "2", "--out", str(tmp_path / "runs.jsonl")]) == 0
    assert peak == 2


def test_zero_budget_stops_non_mock_before_call(tmp_path, monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "run_agent", fail_if_called)
    code = runner.main(["--scenario", "exfil_email_canary", "--model", "sonnet", "--max-usd", "0.0", "--out", str(tmp_path / "runs.jsonl")])
    assert code == 2
    assert not called
