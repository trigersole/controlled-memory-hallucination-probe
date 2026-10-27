import runpy
from pathlib import Path

import pytest
import yaml

create_config = runpy.run_path("scripts/create_factual_shift_config.py")["create_config"]


def test_existing_model_settings_are_reused_without_reusing_old_adapters(tmp_path):
    source = tmp_path / "llama.yaml"
    previous = {
        "experiment": {
            "output_dir": str(tmp_path / "old_run"),
            "source_output_dir": str(tmp_path / "old_adapters"),
        },
        "model": {
            "name_or_path": "/cluster/cached/llama2-7b",
            "dtype": "float16",
            "load_in_4bit": True,
            "max_sequence_length": 256,
        },
    }
    source.write_text(yaml.safe_dump(previous))
    original = source.read_bytes()
    target = tmp_path / "factual_llama.yaml"
    output = str(tmp_path / "new_run")
    create_config(source, target, Path("configs/factual_shift.yaml"), output)
    derived = yaml.safe_load(target.read_text())
    assert derived["model"]["name_or_path"] == previous["model"]["name_or_path"]
    assert derived["model"]["dtype"] == "float16"
    assert derived["model"]["max_sequence_length"] == 512
    assert derived["experiment"]["output_dir"] == output
    assert "source_output_dir" not in derived["experiment"]
    assert source.read_bytes() == original
    assert not Path(output).exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        create_config(source, target, Path("configs/factual_shift.yaml"), output)


@pytest.mark.parametrize("suffix", ["", "/nested_run"])
def test_existing_output_directories_are_protected(tmp_path, suffix):
    source = tmp_path / "llama.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "experiment": {"output_dir": str(tmp_path / "old_run")},
                "model": {"name_or_path": "cached-llama"},
            }
        )
    )
    with pytest.raises(ValueError, match="separate"):
        create_config(
            source,
            tmp_path / "derived.yaml",
            Path("configs/factual_shift.yaml"),
            str(tmp_path / "old_run") + suffix,
        )
