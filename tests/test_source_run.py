import json

import pytest

from peft_probe.config import validate_source_run


def _config(output_dir, source_dir, rank=8):
    return {
        "experiment": {
            "output_dir": str(output_dir),
            "source_output_dir": str(source_dir),
        },
        "model": {"name_or_path": "model"},
        "data": {"num_facts": 12},
        "lora": {"rank": rank},
    }


def test_derived_run_accepts_matching_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _config(tmp_path / "derived", source)
    (source / "_RUN.json").write_text(
        json.dumps({"config": {key: config[key] for key in ("model", "data", "lora")}}),
        encoding="utf-8",
    )
    assert validate_source_run(config) == source.resolve()


def test_derived_run_rejects_incompatible_adapter_settings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _config(tmp_path / "derived", source)
    source_config = _config(tmp_path / "unused", source, rank=16)
    (source / "_RUN.json").write_text(
        json.dumps(
            {"config": {key: source_config[key] for key in ("model", "data", "lora")}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lora"):
        validate_source_run(config)
