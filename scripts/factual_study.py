"""Prepare isolated confirmatory runs, then summarize all planned seeds without cherry-picking."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_probe.config import config_fingerprint
from peft_probe.within_question import _json, _sha

PRIMARY = "-17/correct_a/delta/fraction_8"


def prepare(source: Path, directory: Path, output: Path, seeds: list[int]) -> Path:
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or original["experiment"]["seed"] in seeds:
        raise ValueError(
            "Use at least three distinct fresh seeds, excluding the exploratory pilot seed"
        )
    settings = original["factual_shift"]
    if -17 not in settings["layers"] or 8 not in settings["ranks"]:
        raise ValueError("The frozen pilot method needs layer -17 and rank 8 in the source config")
    if not settings.get("ood_max_examples"):
        raise ValueError("The confirmatory study requires OOD questions")
    protected = Path(original["experiment"]["output_dir"]).resolve()
    destination = output.resolve()
    if (
        destination == protected
        or destination.is_relative_to(protected)
        or protected.is_relative_to(destination)
    ):
        raise ValueError("Study outputs must be separate from the pilot output tree")
    if directory.exists() or output.exists():
        raise FileExistsError(
            "Choose new study config and output directories; existing runs are protected"
        )
    # Validate all prospective configs before creating any files.
    runs = []
    for seed in seeds:
        config = {k: copy.deepcopy(v) for k, v in original.items() if not k.startswith("_")}
        config["experiment"].update(
            seed=seed,
            name=f"factual_confirmation_seed_{seed}",
            output_dir=str(output / f"seed_{seed}"),
        )
        config["experiment"].pop("source_output_dir", None)
        config["factual_shift"].update(
            confirmatory_method=PRIMARY,
            ood_candidate_mode="references",
            ood_references_per_class=3,
            bootstrap_samples=2000,
            analysis_report_scope="confirmatory",
        )
        runs.append((directory / f"seed_{seed}.yaml", config))
    directory.mkdir(parents=True)
    entries = []
    for path, config in runs:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entries.append(
            {
                "seed": config["experiment"]["seed"],
                "config": str(path),
                "config_sha256": _sha(path),
                "config_fingerprint": config_fingerprint(config),
                "output_dir": config["experiment"]["output_dir"],
            }
        )
    manifest = directory / "study.json"
    _json(
        manifest,
        {
            "schema_version": 1,
            "primary_method": PRIMARY,
            "source_config": str(source),
            "source_sha256": _sha(source),
            "pilot_seed_excluded": original["experiment"]["seed"],
            "ood_candidate_mode": "references",
            "reference_candidates_per_class": 3,
            "seed_scope": "data splitting, negative sampling, training, answer sampling and fitting",
            "aggregation": "all planned seeds, descriptive equal-seed mean and sample SD; no pooled seed-question CI",
            "runs": entries,
        },
    )
    return manifest


def aggregate(manifest: Path) -> Path:
    output = manifest.parent / "report"
    (output / "_SUCCESS.json").unlink(missing_ok=True)
    study = json.loads(manifest.read_text(encoding="utf-8"))
    if study["schema_version"] != 1:
        raise ValueError("Unsupported study manifest")
    entries, file_hashes = [], {}
    for run in study["runs"]:
        config_path = Path(run["config"])
        if _sha(config_path) != run["config_sha256"]:
            raise ValueError(f"Planned config changed: {config_path}")
        directory = Path(run["output_dir"]) / "factual_shift_v1/results/within_question_final"
        marker = directory / "_SUCCESS.json"
        if not marker.exists() or not json.loads(marker.read_text(encoding="utf-8")).get(
            "complete"
        ):
            raise FileNotFoundError(f"Seed {run['seed']} is incomplete: {marker}")
        path = directory / "metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if (
            report["signature"] != marker_data["signature"]
            or report["signature"]["analysis_signature"]["config_fingerprint"]
            != run["config_fingerprint"]
            or report.get("selection_rule") != "predeclared_method_validation_sign"
            or report.get("ood_candidate_mode") != "references"
            or report["selections"]["primary"] != study["primary_method"]
        ):
            raise ValueError(f"Seed {run['seed']} does not match the frozen study plan")
        file_hashes[str(path)] = _sha(path)
        entries.append({"seed": run["seed"], "report": report})
    if len(entries) < 3 or len({entry["seed"] for entry in entries}) != len(entries):
        raise ValueError("Study needs at least three distinct completed planned seeds")
    lines = [
        "# Confirmatory factual LoRA study",
        "",
        f"Frozen primary: `{study['primary_method']}`. All {len(entries)} planned fresh seeds are included.",
        "The exploratory pilot is excluded. Each seed changes the full pipeline, including the data split.",
        "OOD is benchmark-reference answer ranking, not generated-answer hallucination detection.",
        "",
        "| Seed | Dataset | Mixed / total questions | A | A-to-B | Incorrect matched | Likelihood | Length control |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    values, comparisons = {}, []
    for entry in entries:
        report = entry["report"]
        selection = report["selections"]
        names = [
            selection["primary"],
            selection["transfer"],
            selection["incorrect_matched_geometry"],
            "base_likelihood",
            "answer_length_words",
        ]
        if names[2] is None:
            raise ValueError("Matched geometry incorrect control is required for confirmation")
        for split in ("test", "ood"):
            dataset = report["datasets"].get(split)
            if dataset is None or not dataset["n_mixed_questions"]:
                raise ValueError(
                    f"Seed {entry['seed']}: no usable mixed questions for {split}; do not drop this seed"
                )
            scores = [
                dataset["methods"][name]["within_question_pairwise_accuracy"] for name in names
            ]
            lines.append(
                f"| {entry['seed']} | {split} | {dataset['n_mixed_questions']} / {dataset['n_questions']} | "
                + " | ".join(f"{score:.4f}" for score in scores)
                + " |"
            )
            for label, score in zip(
                ("A", "A-to-B", "Incorrect matched", "Likelihood", "Length control"),
                scores,
                strict=True,
            ):
                values.setdefault((split, label), []).append(score)
            requested = [
                (names[1], names[0]),
                (names[0], names[2]),
                (names[0], names[3]),
                (names[0], names[4]),
            ]
            available = {(p["left"], p["right"]): p for p in dataset["paired_comparisons"]}
            for left, right in requested:
                item = available[(left, right)]
                comparisons.append({"seed": entry["seed"], "split": split, **item})
            for name, item in dataset["methods"].items():
                if "activation_pca" in name or name.endswith(("/supervised_probe", "/random/base")):
                    values.setdefault((split, name), []).append(
                        item["within_question_pairwise_accuracy"]
                    )
                    comparisons.append(
                        {"seed": entry["seed"], "split": split, **available[(names[0], name)]}
                    )
    lines.extend(
        [
            "",
            "Equal-seed descriptive summaries (mean ± sample SD):",
            "",
            "| Dataset | Method | Mean | SD across seeds |",
            "|---|---|---:|---:|",
        ]
    )
    descriptive = []
    for (split, method), scores in values.items():
        mean, sd = statistics.mean(scores), statistics.stdev(scores)
        descriptive.append({"split": split, "method": method, "mean": mean, "seed_sd": sd})
        lines.append(f"| {split} | {method} | {mean:.4f} | {sd:.4f} |")
    lines.extend(
        [
            "",
            "Per-seed paired differences; intervals condition on that trained run:",
            "",
            "| Seed | Dataset | Left minus right | Difference | 95% cluster CI |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['seed']} | {item['split']} | {item['left']} minus {item['right']} | "
            f"{item['mean']:.4f} | [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Question/alias bootstrap intervals do not measure training-seed uncertainty.",
            "Questions may overlap across seeds; their repeated observations are not independent samples.",
            "No combined seed-question interval or equivalence claim is made. These intervals are not multiplicity-adjusted.",
            "A consistent result supports replication for this model and protocol, not universal or causal truth representations.",
            "OOD reference ranking does not resolve the pilot's ambiguous generated-answer grading.",
        ]
    )
    output.mkdir(exist_ok=True)
    signature = {"manifest_sha256": _sha(manifest), "inputs_sha256": file_hashes}
    (output / "_SUCCESS.json").unlink(missing_ok=True)
    _json(
        output / "metrics.json",
        {"signature": signature, "descriptive": descriptive, "paired_comparisons": comparisons},
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output / "_SUCCESS.json", {"complete": True, "signature": signature})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("source", type=Path)
    create.add_argument("--directory", type=Path, default=Path("configs/factual_confirmation"))
    create.add_argument("--output-dir", type=Path, default=Path("outputs/factual_confirmation"))
    create.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    summarize = commands.add_parser("summarize")
    summarize.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.source, args.directory, args.output_dir, args.seeds))
    else:
        print(aggregate(args.manifest))


if __name__ == "__main__":
    main()
