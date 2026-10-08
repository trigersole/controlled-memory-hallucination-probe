from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .versioning import artifact_metadata


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = str(path.resolve())
    return config


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config["experiment"]["output_dir"]).resolve()


def config_fingerprint(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def initialize_output(config: dict[str, Any]) -> Path:
    root = output_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    run_file = root / "_RUN.json"
    fingerprint = config_fingerprint(config)
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = {"config_fingerprint": fingerprint, "config": clean, **artifact_metadata()}
    if run_file.exists():
        with run_file.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{root} already belongs to a different configuration. Change experiment.output_dir "
                "to start a new run; existing experimental artifacts will not be mixed."
            )
        payload["initial_code_revision"] = previous.get(
            "initial_code_revision", previous.get("code_revision", "unknown")
        )
    else:
        payload["initial_code_revision"] = payload["code_revision"]
    temporary = run_file.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(run_file)
    return root
