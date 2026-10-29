"""Create an isolated PopQA-only training/evaluation run from an existing factual config."""

import argparse
import copy
from pathlib import Path

import yaml


def create(source: Path, target: Path, output: str | None = None, seed: int | None = None):
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not original.get("model", {}).get("name_or_path") or "factual_shift" not in original:
        raise ValueError(
            "Source must be an existing factual experiment with its base model configuration"
        )
    config = {k: copy.deepcopy(v) for k, v in original.items() if not k.startswith("_")}
    run_seed = int(original["experiment"]["seed"] if seed is None else seed)
    if (
        run_seed < 0
        or -17 not in config["factual_shift"]["layers"]
        or 8 not in config["factual_shift"]["ranks"]
    ):
        raise ValueError("Use a nonnegative seed and a source supporting fixed layer -17/rank 8")
    destination = output or f"outputs/popqa_in_domain/seed_{run_seed}"
    absolute = Path(destination).resolve()
    protected = [
        original["experiment"]["output_dir"],
        original["experiment"].get("source_output_dir"),
        "outputs/factual_popqa",
        "outputs/factual_confirmation",
    ]
    for path in filter(None, protected):
        directory = Path(path).resolve()
        if (
            absolute == directory
            or absolute.is_relative_to(directory)
            or directory.is_relative_to(absolute)
        ):
            raise ValueError("New PopQA training outputs must be separate from existing runs")
    if target.exists() or absolute.exists():
        raise FileExistsError("Choose a new config/output path; existing artifacts are preserved")
    config["experiment"].update(
        seed=run_seed, name=f"popqa_in_domain_seed_{run_seed}", output_dir=destination
    )
    config["experiment"].pop("source_output_dir", None)
    config["factual_shift"].update(
        dataset="akariasai/PopQA",
        dataset_config=None,
        dataset_split="test",
        dataset_revision="main",
        split_grouping="subject",
        ood_max_examples=0,
        ood_candidate_mode="generated",
        confirmatory_method="-17/correct_a/delta/fraction_8",
        analysis_report_scope="confirmatory",
        bootstrap_samples=2000,
        length_adjustment=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(create(args.source, args.target, args.output_dir, args.seed))
