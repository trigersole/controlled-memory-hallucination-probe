import json
import time

import pytest
import yaml

from peft_probe.factual_runtime import run_job, stage


def setup_run(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"experiment": {"output_dir": str(tmp_path / "run")}}))
    return config, tmp_path / "run/factual_shift_v1"


@pytest.mark.parametrize("failure", [RuntimeError("generation failed"), SystemExit(1)])
def test_failure_before_analysis_records_stage_and_traceback(tmp_path, failure):
    config, output = setup_run(tmp_path)

    def fail():
        raise failure

    with pytest.raises(type(failure)):
        run_job(config, lambda: stage("candidates", fail))
    report = json.loads((output / "pipeline_failure.json").read_text())
    assert report["stage"] == "candidates"
    assert report["status"] == "failed"
    assert report["error_type"] == type(failure).__name__
    assert "Traceback" in report["traceback"]


def test_heartbeat_and_resume_preserve_stage_artifacts_and_failure_history(tmp_path):
    config, output = setup_run(tmp_path)

    def fail():
        raise ValueError("first attempt")

    with pytest.raises(ValueError):
        run_job(config, lambda: stage("candidates", fail))
    adapter = output / "adapters/correct_a/final/_SUCCESS.json"
    adapter.parent.mkdir(parents=True)
    adapter.write_text('{"complete":true}')
    before = adapter.read_bytes()

    def wait():
        time.sleep(0.06)
        progress = json.loads((output / "pipeline_progress.json").read_text())
        assert progress["stage"] == "candidates"
        assert "heartbeat_at" in progress
        return "done"

    assert run_job(config, lambda: stage("candidates", wait), interval=0.01) == "done"
    assert not (output / "pipeline_failure.json").exists()
    assert adapter.read_bytes() == before
    assert json.loads((output / "pipeline_progress.json").read_text())["status"] == "complete"
    events = [
        json.loads(line) for line in (output / "pipeline_events.jsonl").read_text().splitlines()
    ]
    assert any(event["status"] == "failed" for event in events)
    assert len({event["attempt"] for event in events}) == 2


def test_import_failure_and_successful_system_exit_are_distinguished(tmp_path):
    config, output = setup_run(tmp_path)

    def missing_import():
        raise ImportError("missing runtime dependency")

    with pytest.raises(ImportError):
        run_job(config, missing_import)
    assert json.loads((output / "pipeline_failure.json").read_text())["stage"] == "initializing"

    def successful_exit():
        raise SystemExit(0)

    with pytest.raises(SystemExit) as raised:
        run_job(config, successful_exit)
    assert raised.value.code == 0
    assert not (output / "pipeline_failure.json").exists()
    assert json.loads((output / "pipeline_progress.json").read_text())["status"] == "complete"
