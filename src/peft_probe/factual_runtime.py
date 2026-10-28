"""Persistent job/stage diagnostics, including failures before analysis starts."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

_active = None


class Runtime:
    def __init__(self, directory: Path, interval: float = 60):
        self.directory = directory
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {
            "attempt": uuid.uuid4().hex,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "stage": "initializing",
            "status": "running",
        }
        self.started = time.monotonic()

    def _write(self, name, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def update(self, stage, status, **extra):
        with self.lock:
            self.started = time.monotonic()
            self.state.update(
                stage=stage,
                status=status,
                updated_at=datetime.now(timezone.utc).isoformat(),
                **extra,
            )
            self._write("pipeline_progress.json", self.state)
            with (self.directory / "pipeline_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.state) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if status == "failed":
                self._write("pipeline_failure.json", self.state)
        print(f"[pipeline] {stage}: {status}", flush=True)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                with self.lock:
                    elapsed = int(time.monotonic() - self.started)
                    self.state.update(
                        heartbeat_at=datetime.now(timezone.utc).isoformat(),
                        seconds_in_stage=elapsed,
                    )
                    self._write("pipeline_progress.json", self.state)
                    print(
                        f"[pipeline heartbeat] {self.state['stage']}: {self.state['status']}; {elapsed}s",
                        flush=True,
                    )
            except OSError as exc:
                print(
                    f"[pipeline diagnostics] Cannot write heartbeat: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


def stage(name, function, *args, **kwargs):
    if _active is not None:
        _active.update(name, "running")
    else:
        print(f"[pipeline] {name}: running", flush=True)
    result = function(*args, **kwargs)
    if _active is not None:
        _active.update(name, "complete")
    return result


def run_job(config_path, callback, interval=60):
    from .config import load_config, output_dir

    global _active
    config = load_config(config_path)
    runtime = Runtime(output_dir(config) / "factual_shift_v1", interval)
    _active = runtime
    runtime.update("initializing", "running")
    (runtime.directory / "pipeline_failure.json").unlink(missing_ok=True)
    worker = threading.Thread(target=runtime.heartbeat, daemon=True)
    worker.start()
    try:
        result = callback()
        runtime.update("pipeline", "complete")
        return result
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            runtime.update("pipeline", "complete")
        else:
            try:
                runtime.update(
                    runtime.state["stage"],
                    "failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            except OSError as log_error:
                print(
                    f"[pipeline diagnostics] Cannot save failure: {log_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        runtime.stop.set()
        worker.join(timeout=1)
        _active = None
