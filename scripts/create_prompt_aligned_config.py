from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an isolated concise-prompt experiment from a completed-run config"
    )
    parser.add_argument("source", type=Path, help="Existing experiment YAML")
    parser.add_argument("target", type=Path, help="New derived experiment YAML")
    parser.add_argument(
        "--output-dir",
        help="Derived output directory (default: source output plus _concise_prompts)",
    )
    args = parser.parse_args()

    with args.source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("Source configuration must contain an experiment mapping")
    if args.target.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.target}; choose a new target path"
        )

    experiment = config["experiment"]
    source_output = str(experiment["output_dir"])
    experiment["name"] = f"{experiment.get('name', 'experiment')}_concise_prompts"
    experiment["output_dir"] = args.output_dir or f"{source_output}_concise_prompts"
    # Preserve an existing adapter source (the token-position run already has one). If the
    # source run is self-contained, make it the immutable supplier of facts and adapters.
    experiment.setdefault("source_output_dir", source_output)
    config["prompting"] = {
        "instruction": "Answer the question concisely",
        "synthetic_template_family": "concise_diverse_v1",
    }

    args.target.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.target.with_suffix(args.target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    temporary.replace(args.target)
    print(args.target)


if __name__ == "__main__":
    main()
