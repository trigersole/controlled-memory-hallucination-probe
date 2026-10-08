from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .versioning import PIPELINE_SCHEMA_VERSION, artifact_metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def completed(path: str | Path) -> bool:
    return Path(path).is_file()


def mark_complete(path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    atomic_json(path, {"complete": True, **artifact_metadata(), **(metadata or {})})


def ensure_manifest(directory: str | Path, fingerprint: str, force: bool = False) -> None:
    directory = Path(directory)
    manifest = directory / "_MANIFEST.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_fingerprint") != fingerprint and not force:
            raise RuntimeError(
                f"Configuration changed for resumable stage {directory}. Use a new experiment.output_dir "
                "or pass --force to intentionally replace its artifacts."
            )
        previous_schema = previous.get("pipeline_schema_version", 1)
        if previous_schema != PIPELINE_SCHEMA_VERSION and not force:
            raise RuntimeError(
                f"Artifact schema changed for resumable stage {directory}: "
                f"found v{previous_schema}, expected v{PIPELINE_SCHEMA_VERSION}. "
                "Use a new experiment.output_dir or pass --force after intentionally "
                "invalidating downstream artifacts."
            )
    atomic_json(manifest, {"config_fingerprint": fingerprint, **artifact_metadata()})


def chunks(items: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(items), size):
        yield start // size, items[start : start + size]
