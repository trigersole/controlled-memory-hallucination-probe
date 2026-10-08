import json

import pytest

from peft_probe.io_utils import ensure_manifest
from peft_probe.versioning import PIPELINE_SCHEMA_VERSION


def test_manifest_records_schema_and_rejects_stale_schema(tmp_path):
    target = tmp_path / "stage"
    ensure_manifest(target, "abc123")
    manifest_path = target / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline_schema_version"] == PIPELINE_SCHEMA_VERSION

    manifest["pipeline_schema_version"] = PIPELINE_SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Artifact schema changed"):
        ensure_manifest(target, "abc123")
