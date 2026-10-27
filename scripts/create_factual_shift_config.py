"""Reuse an existing run's base model for a separate factual-shift experiment."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def create_config(source: Path, target: Path, template: Path, output: str | None = None) -> Path:
    with source.open(encoding="utf-8") as handle:
        previous = yaml.safe_load(handle)
    with template.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(previous, dict) or not isinstance(previous.get("model"), dict):
        raise TypeError("Source must contain the existing model configuration")
    if not previous["model"].get("name_or_path"):
        raise ValueError("Source model.name_or_path is required")
    if not isinstance(config, dict) or "factual_shift" not in config:
        raise ValueError("Template must be a factual-shift configuration")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}; choose a new target")
    destination = Path(output or f"outputs/{target.stem}").resolve()
    protected = [config["experiment"]["output_dir"]]
    protected.extend(
        previous.get("experiment", {}).get(key) for key in ("output_dir", "source_output_dir")
    )
    for directory in filter(None, protected):
        if destination == Path(directory).resolve() or destination.is_relative_to(
            Path(directory).resolve()
        ):
            raise ValueError("New output_dir must be separate from existing run directories")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    sequence_length = max(
        int(config["model"]["max_sequence_length"]),
        int(previous["model"].get("max_sequence_length", 0)),
    )
    config["model"] = copy.deepcopy(previous["model"])
    config["model"]["max_sequence_length"] = sequence_length
    config["experiment"]["name"] = target.stem
    config["experiment"]["output_dir"] = output or f"outputs/{target.stem}"
    config["experiment"].pop("source_output_dir", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Config from your existing Llama run")
    parser.add_argument("target", type=Path, help="New factual-shift config path")
    parser.add_argument("--template", type=Path, default=Path("configs/factual_shift.yaml"))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(create_config(args.source, args.target, args.template, args.output_dir))


if __name__ == "__main__":
    main()
