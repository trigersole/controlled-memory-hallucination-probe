from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from .collect import _batch_generate_and_extract, load_feature_shards, normalize
from .config import config_fingerprint, output_dir
from .io_utils import atomic_json, atomic_jsonl, atomic_torch_save, chunks, ensure_manifest, mark_complete
from .metrics import binary_error_metrics, bootstrap_metric_differences, risk_coverage
from .modeling import load_base_model, load_tokenizer
from .probe import load_probe


def _prepare_trivia(config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["benchmarks"]
    dataset = load_dataset(
        settings.get("trivia_qa_dataset", "mandarjoshi/trivia_qa"),
        settings["trivia_qa_config"],
        split=settings["split"],
    )
    rows = []
    for index, item in enumerate(dataset):
        aliases = list(dict.fromkeys([item["answer"]["value"], *item["answer"].get("aliases", [])]))
        rows.append(
            {
                "fact_id": f"trivia_{index:06d}",
                "entity_id": f"trivia_{index:06d}",
                "relation": "trivia_qa",
                "detector_split": "benchmark",
                "adapter_assignment": None,
                "eval_question": "Answer with only the short answer. " + item["question"],
                "value": aliases[0],
                "aliases": aliases,
            }
        )
    return rows


def _prepare_truthful(config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["benchmarks"]
    mode = settings.get("truthfulqa_mode", "mc1")
    dataset_config = "multiple_choice" if mode == "mc1" else "generation"
    dataset = load_dataset(
        settings.get("truthfulqa_dataset", "truthfulqa/truthful_qa"),
        dataset_config,
        split=settings["split"],
    )
    rows = []
    for index, item in enumerate(dataset):
        row = {
            "fact_id": f"truthful_{index:06d}",
            "entity_id": f"truthful_{index:06d}",
            "relation": "truthful_qa",
            "detector_split": "benchmark",
            "adapter_assignment": None,
        }
        if mode == "mc1":
            choices = item["mc1_targets"]["choices"]
            labels = item["mc1_targets"]["labels"]
            letters = [chr(ord("A") + choice_index) for choice_index in range(len(choices))]
            formatted = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=True)
            )
            correct_index = next(i for i, label in enumerate(labels) if int(label) == 1)
            correct_letter = letters[correct_index]
            row.update(
                {
                    "eval_question": f"{item['question']}\n{formatted}\nReply with the letter only.",
                    "value": correct_letter,
                    "correct_letter": correct_letter,
                    "truthfulqa_mode": "mc1",
                }
            )
        else:
            correct = list(dict.fromkeys([item["best_answer"], *item["correct_answers"]]))
            row.update(
                {
                    "eval_question": "Give a brief, factual answer. " + item["question"],
                    "value": item["best_answer"],
                    "correct_answers": correct,
                    "incorrect_answers": item["incorrect_answers"],
                    "truthfulqa_mode": "generation",
                }
            )
        rows.append(row)
    return rows


def _grade_trivia(answer: str, aliases: list[str]) -> tuple[int, str]:
    def official_normalize(text: str) -> str:
        normalized = normalize(text)
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", normalized).split())

    prediction = official_normalize(answer)
    accepted = {official_normalize(alias) for alias in aliases}
    return int(prediction in accepted), "official_normalized_exact_match"


def _reference_match(answer: str, references: list[str]) -> bool:
    prediction = normalize(answer)
    for reference in references:
        normalized_reference = normalize(reference)
        if normalized_reference and (
            prediction == normalized_reference or normalized_reference in prediction
        ):
            return True
    return False


def _grade_truthful(answer: str, correct: list[str], incorrect: list[str]) -> tuple[int, str]:
    matches_correct = _reference_match(answer, correct)
    matches_incorrect = _reference_match(answer, incorrect)
    if matches_correct and not matches_incorrect:
        return 1, "reference_match_correct"
    if matches_incorrect and not matches_correct:
        return 0, "reference_match_incorrect"
    # Preserve ambiguity: these rows are saved but excluded from primary metrics.
    return -1, "ambiguous_reference_match"


def _grade_truthful_mc1(answer: str, correct_letter: str) -> tuple[int, str]:
    match = re.search(r"\b([A-Z])\b", answer.upper())
    if not match:
        return 0, "mc1_unparseable"
    return int(match.group(1) == correct_letter), "mc1"


def prepare_benchmark(config: dict[str, Any], name: str) -> list[dict[str, Any]]:
    if name == "trivia_qa":
        rows = _prepare_trivia(config)
    elif name == "truthful_qa":
        rows = _prepare_truthful(config)
    else:
        raise ValueError(f"Unsupported benchmark: {name}")
    maximum = config["benchmarks"].get("max_examples")
    return rows[: int(maximum)] if maximum is not None else rows


def collect_benchmark(config: dict[str, Any], name: str, force: bool = False) -> Path:
    if name not in config["benchmarks"]["datasets"]:
        raise ValueError(f"Benchmark {name} is not enabled")
    target = output_dir(config) / "features" / "benchmarks" / name
    ensure_manifest(target, config_fingerprint(config), force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    rows = prepare_benchmark(config, name)
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    model.eval()
    collection_settings = dict(config["collection"])
    collection_settings["feature_modes"] = list(config["collection"]["feature_modes"])
    shard_size = int(config["benchmarks"]["shard_size"])
    batch_size = int(config["collection"]["batch_size"])
    expected_shards = math.ceil(len(rows) / shard_size)
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        records: list[dict[str, Any]] = []
        feature_parts: dict[str, list[torch.Tensor]] = {
            mode: [] for mode in collection_settings["feature_modes"]
        }
        for _, batch_rows in chunks(shard_rows, batch_size):
            batch_records, batch_features = _batch_generate_and_extract(
                model, tokenizer, batch_rows, "base", collection_settings
            )
            for original, record in zip(batch_rows, batch_records, strict=True):
                if name == "trivia_qa":
                    correct, status = _grade_trivia(record["answer"], original["aliases"])
                elif original.get("truthfulqa_mode") == "mc1":
                    correct, status = _grade_truthful_mc1(
                        record["answer"], original["correct_letter"]
                    )
                else:
                    correct, status = _grade_truthful(
                        record["answer"], original["correct_answers"], original["incorrect_answers"]
                    )
                record["correct"] = correct
                record["grading_status"] = status
                record["benchmark"] = name
                records.append(record)
            for mode, tensor in batch_features.items():
                feature_parts[mode].append(tensor)
        atomic_torch_save(
            shard_path,
            {
                "records": records,
                "features": {mode: torch.cat(parts) for mode, parts in feature_parts.items()},
                "benchmark": name,
                "shard_index": shard_index,
                "config_fingerprint": config_fingerprint(config),
            },
        )
    actual_shards = len(list(target.glob("shard_*.pt")))
    if actual_shards != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} benchmark shards, found {actual_shards}")
    mark_complete(
        done,
        {
            "benchmark": name,
            "num_examples": len(rows),
            "num_shards": actual_shards,
            "config_fingerprint": config_fingerprint(config),
        },
    )
    return target


@torch.inference_mode()
def _probe_probabilities(model, payload: dict[str, Any], features: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    standardized = (features.float() - payload["mean"]) / payload["std"]
    probabilities = []
    for _, batch in chunks(list(range(len(standardized))), 256):
        correctness, _ = model(standardized[batch].to(device))
        probabilities.append((1.0 - torch.sigmoid(correctness)).cpu())
    return torch.cat(probabilities).numpy()


def evaluate_benchmark(config: dict[str, Any], name: str, force: bool = False) -> Path:
    root = output_dir(config)
    target = root / "results" / name
    ensure_manifest(target, config_fingerprint(config), force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target
    target.mkdir(parents=True, exist_ok=True)

    records_by_mode: dict[str, list[dict[str, Any]]] = {}
    features_by_mode: dict[str, torch.Tensor] = {}
    for mode in config["collection"]["feature_modes"]:
        records, features = load_feature_shards(root / "features" / "benchmarks" / name, mode)
        records_by_mode[mode] = records
        features_by_mode[mode] = features

    summary: dict[str, Any] = {"benchmark": name, "models": {}, "baselines": {}}
    default_records = records_by_mode[config["collection"]["feature_modes"][0]]
    scored_mask = np.asarray([record["correct"] >= 0 for record in default_records])
    labels = np.asarray([max(0, record["correct"]) for record in default_records], dtype=int)[scored_mask]
    summary["num_total"] = len(default_records)
    summary["num_scored"] = int(scored_mask.sum())
    summary["grading_coverage"] = float(scored_mask.mean())
    attempted_mask = scored_mask & ~np.asarray(
        [bool(record["abstained"]) for record in default_records]
    )
    summary["abstention_rate"] = float(
        np.mean([bool(record["abstained"]) for record in default_records])
    )
    if not scored_mask.any():
        raise RuntimeError(f"No scorable examples for {name}; inspect saved generations")

    logprob_risk = 1.0 - np.exp(
        np.asarray([record["mean_logprob"] for record in default_records], dtype=float)[scored_mask]
    )
    entropy_risk = 1.0 - np.exp(
        -np.asarray([record["mean_entropy"] for record in default_records], dtype=float)[scored_mask]
    )
    summary["baselines"]["token_logprob"] = binary_error_metrics(labels, logprob_risk)
    summary["baselines"]["token_entropy"] = binary_error_metrics(labels, entropy_risk)
    summary["attempted_only"] = {"models": {}, "baselines": {}}
    attempted_labels = np.asarray(
        [max(0, record["correct"]) for record in default_records], dtype=int
    )[attempted_mask]
    if attempted_mask.any():
        summary["attempted_only"]["baselines"]["token_logprob"] = binary_error_metrics(
            attempted_labels,
            (1.0 - np.exp(np.asarray([r["mean_logprob"] for r in default_records])))[attempted_mask],
        )
        summary["attempted_only"]["baselines"]["token_entropy"] = binary_error_metrics(
            attempted_labels,
            (1.0 - np.exp(-np.asarray([r["mean_entropy"] for r in default_records])))[attempted_mask],
        )

    predictions: dict[str, Any] = {
        record["fact_id"]: {
            "fact_id": record["fact_id"],
            "question": record["question"],
            "answer": record["answer"],
            "correct": record["correct"],
            "grading_status": record["grading_status"],
            "mean_logprob": record["mean_logprob"],
            "mean_entropy": record["mean_entropy"],
        }
        for record in default_records
    }
    probabilities: dict[tuple[str, str, int], np.ndarray] = {}
    for mode in config["collection"]["feature_modes"]:
        for variant in config["probe"]["variants"]:
            for seed in config["probe"]["seeds"]:
                model_path = root / "probes" / mode / variant / f"seed_{seed}" / "model.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"Missing trained probe: {model_path}")
                model, payload = load_probe(model_path)
                risk = _probe_probabilities(model, payload, features_by_mode[mode])
                key = (mode, variant, int(seed))
                probabilities[key] = risk
                label = f"{mode}/{variant}/seed_{seed}"
                summary["models"][label] = binary_error_metrics(labels, risk[scored_mask])
                if attempted_mask.any():
                    summary["attempted_only"]["models"][label] = binary_error_metrics(
                        attempted_labels, risk[attempted_mask]
                    )
                curve = risk_coverage(labels, risk[scored_mask])
                atomic_jsonl(
                    target / f"risk_coverage_{mode}_{variant}_seed_{seed}.jsonl",
                    (
                        {"coverage": float(c), "selective_accuracy": float(a), "selective_risk": float(r)}
                        for c, a, r in zip(
                            curve["coverage"], curve["selective_accuracy"], curve["selective_risk"], strict=True
                        )
                    ),
                )
                for record, value in zip(default_records, risk, strict=True):
                    predictions[record["fact_id"]][label] = float(value)

    comparisons: dict[str, Any] = {}
    for mode in config["collection"]["feature_modes"]:
        for seed in config["probe"]["seeds"]:
            proposal = probabilities[(mode, "genuine_exposure", int(seed))][scored_mask]
            baseline = probabilities[(mode, "correctness", int(seed))][scored_mask]
            comparisons[f"{mode}/seed_{seed}"] = bootstrap_metric_differences(
                labels,
                proposal,
                baseline,
                int(config["benchmarks"]["bootstrap_samples"]),
                int(seed),
            )
    summary["genuine_vs_correctness_bootstrap"] = comparisons
    seed_aggregate: dict[str, Any] = {}
    for mode in config["collection"]["feature_modes"]:
        for variant in config["probe"]["variants"]:
            prefix = f"{mode}/{variant}/seed_"
            runs = [metrics for label, metrics in summary["models"].items() if label.startswith(prefix)]
            seed_aggregate[f"{mode}/{variant}"] = {
                metric: {
                    "mean": float(np.nanmean([run[metric] for run in runs])),
                    "std": float(np.nanstd([run[metric] for run in runs], ddof=1))
                    if len(runs) > 1 else 0.0,
                }
                for metric in ("auroc", "auprc", "brier", "ece", "aurc")
            }
    summary["seed_aggregate"] = seed_aggregate
    atomic_json(target / "metrics.json", summary)
    atomic_jsonl(target / "predictions.jsonl", predictions.values())
    mark_complete(done, {"benchmark": name, "config_fingerprint": config_fingerprint(config)})
    return target


def intervention_report(config: dict[str, Any]) -> Path:
    root = output_dir(config)
    rows = []
    for source in ("adapter_a", "adapter_b"):
        records, _ = load_feature_shards(
            root / "features" / "synthetic" / source,
            config["collection"]["feature_modes"][0],
        )
        rows.extend(record for record in records if record["split"] != "ood")
    exposed = np.asarray([record["correct"] for record in rows if record["exposure"] == 1])
    withheld = np.asarray([record["correct"] for record in rows if record["exposure"] == 0])
    paired: dict[str, dict[int, int]] = {}
    for record in rows:
        paired.setdefault(record["fact_id"], {})[int(record["exposure"])] = int(record["correct"])
    paired_differences = np.asarray(
        [values[1] - values[0] for values in paired.values() if 0 in values and 1 in values],
        dtype=float,
    )
    if not len(paired_differences):
        raise RuntimeError("No complementary exposed/withheld fact pairs were found")
    gap = float(paired_differences.mean())
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    bootstrap_gaps = []
    for _ in range(int(config["benchmarks"]["bootstrap_samples"])):
        bootstrap_gaps.append(
            float(rng.choice(paired_differences, size=len(paired_differences), replace=True).mean())
        )
    minimum_gap = float(config["collection"].get("minimum_memory_gap", 0.0))
    report = {
        "accuracy_exposed": float(exposed.mean()),
        "accuracy_withheld": float(withheld.mean()),
        "memory_gap": gap,
        "memory_gap_ci95": [
            float(np.quantile(bootstrap_gaps, 0.025)),
            float(np.quantile(bootstrap_gaps, 0.975)),
        ],
        "minimum_required_gap": minimum_gap,
        "passed": gap >= minimum_gap,
        "num_exposed": int(len(exposed)),
        "num_withheld": int(len(withheld)),
        "num_paired_facts": int(len(paired_differences)),
        "abstention_rate_exposed": float(
            np.mean([record["abstained"] for record in rows if record["exposure"] == 1])
        ),
        "abstention_rate_withheld": float(
            np.mean([record["abstained"] for record in rows if record["exposure"] == 0])
        ),
        "warning": "Do not interpret probe results unless the memory gap is clearly positive.",
    }
    path = root / "results" / "intervention_check.json"
    atomic_json(path, report)
    if bool(config["collection"].get("fail_on_small_gap", False)) and gap < minimum_gap:
        raise RuntimeError(
            f"Memory intervention gap {gap:.3f} is below required {minimum_gap:.3f}; "
            f"probe training is intentionally blocked. See {path}."
        )
    return path
