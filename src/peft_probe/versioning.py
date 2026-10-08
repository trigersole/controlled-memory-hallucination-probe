from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


PIPELINE_SCHEMA_VERSION = 3


@lru_cache(maxsize=1)
def code_revision() -> str:
    override = os.environ.get("PEFT_PROBE_CODE_REVISION")
    if override:
        return override
    project_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def artifact_metadata() -> dict[str, Any]:
    return {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "code_revision": code_revision(),
    }
