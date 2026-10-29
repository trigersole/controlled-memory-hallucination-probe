"""Create a small, isolated TruthfulQA-only reference-ranking training experiment."""

import argparse
import copy
from pathlib import Path

import yaml


def create(source: Path, target: Path, output: str | None = None, seed: int | None = None):
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not original.get("model", {}).get("name_or_path") or "factual_shift" not in original:
        raise ValueError("Source must contain the existing factual experiment and base model")
    config = {k: copy.deepcopy(v) for k, v in original.items() if not k.startswith("_")}
    run_seed = int(original["experiment"]["seed"] if seed is None else seed)
    if (
        run_seed < 0
        or -17 not in config["factual_shift"]["layers"]
        or 8 not in config["factual_shift"]["ranks"]
    ):
        raise ValueError("Use a nonnegative seed and a source supporting layer -17/rank 8")
    destination = output or f"outputs/truthfulqa_in_domain/seed_{run_seed}"
    absolute = Path(destination).resolve()
    protected = [
        original["experiment"]["output_dir"],
        original["experiment"].get("source_output_dir"),
        "outputs/factual_confirmation",
        "outputs/factual_popqa",
        "outputs/popqa_in_domain",
    ]
    for path in filter(None, protected):
        directory = Path(path).resolve()
        if (
            absolute == directory
            or absolute.is_relative_to(directory)
            or directory.is_relative_to(absolute)
        ):
            raise ValueError("TruthfulQA outputs must be separate from existing experiments")
    if target.exists() or absolute.exists():
        raise FileExistsError("Choose new config/output paths; existing artifacts are preserved")
    config["experiment"].update(
        seed=run_seed, name=f"truthfulqa_in_domain_seed_{run_seed}", output_dir=destination
    )
    config["experiment"].pop("source_output_dir", None)
    config["factual_shift"].update(
        dataset="truthfulqa/truthful_qa",
        dataset_config="generation",
        dataset_split="validation",
        dataset_revision="main",
        split_grouping="question_source",
        split_sizes={
            "adapter_a": 80,
            "adapter_b": 80,
            "direction_fit": 80,
            "probe_train": 80,
            "validation": 80,
            "test": 200,
        },
        ood_max_examples=0,
        ood_candidate_mode="generated",
        candidate_mode="references",
        negative_source="references",
        ood_references_per_class=3,
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
