from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import config_fingerprint, output_dir
from .versioning import artifact_metadata


SCHEMA_VERSION = 1
DATASET = "pop_qa"


def load_base_contextual_transfer_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Base-contextual transfer configuration must be a mapping")
    if int(settings.get("base_contextual_transfer_schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"base_contextual_transfer_schema_version must be {SCHEMA_VERSION}"
        )
    if settings.get("dataset") != DATASET:
        raise ValueError(f"dataset must be {DATASET}")
    for key in (
        "generation_batch_size",
        "score_batch_size",
        "shard_size",
        "max_new_tokens",
        "max_sequence_length",
        "max_answer_tokens",
        "control_repeats",
        "control_search_pool",
        "bootstrap_samples",
    ):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    coverages = [float(value) for value in settings.get("coverages", [])]
    if not coverages or any(not 0 < value < 1 for value in coverages):
        raise ValueError("coverages must be a non-empty list with values in (0, 1)")
    maximum = settings.get("max_examples")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("max_examples must be null or positive")
    int(settings["seed"])
    return settings


def base_contextual_transfer_fingerprint(
    config: dict[str, Any], settings: dict[str, Any]
) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "base_contextual_transfer": settings,
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def base_contextual_transfer_root(config: dict[str, Any]) -> Path:
    return output_dir(config) / "analysis" / f"base_contextual_transfer_v{SCHEMA_VERSION}"


def _parse_aliases(value: Any) -> list[str]:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(stripped)
        else:
            parsed = [stripped]
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))


def _prepare_popqa(config: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    from .prompting import trivia_question

    dataset = load_dataset(
        str(settings.get("dataset_name", "akariasai/PopQA")),
        split=str(settings.get("split", "test")),
    )
    rows = []
    for index, item in enumerate(dataset):
        aliases = _parse_aliases(item.get("possible_answers", []))
        aliases = list(dict.fromkeys([str(item["obj"]), *aliases]))
        item_id = str(item.get("id", index))
        rows.append(
            {
                "fact_id": f"popqa_{item_id}",
                "entity_id": f"popqa_{item_id}",
                "relation": str(item["prop"]),
                "detector_split": "benchmark",
                "adapter_assignment": None,
                "eval_question": trivia_question(config, str(item["question"])),
                "value": aliases[0],
                "aliases": aliases,
                "subject": str(item["subj"]),
                "subject_popularity": int(item.get("s_pop", 0)),
            }
        )
    # Avoid using an identical question as its own counterfactual through a duplicate row.
    deduplicated = []
    seen_questions = set()
    for row in rows:
        key = (row["relation"], row["eval_question"])
        if key not in seen_questions:
            deduplicated.append(row)
            seen_questions.add(key)
    maximum = settings.get("max_examples")
    return deduplicated[: int(maximum)] if maximum is not None else deduplicated


def collect_popqa_candidates(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    from .benchmark import _grade_trivia
    from .collect import _batch_generate_and_extract
    from .io_utils import (
        atomic_torch_save,
        chunks,
        ensure_manifest,
        mark_complete,
        seed_everything,
    )
    from .modeling import load_base_model, load_tokenizer

    target = base_contextual_transfer_root(config) / "candidates" / DATASET
    fingerprint = base_contextual_transfer_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    seed_everything(int(settings["seed"]))
    rows = _prepare_popqa(config, settings)
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    model.eval()
    generation_settings = dict(config["collection"])
    generation_settings["feature_modes"] = []
    generation_settings["max_new_tokens"] = int(settings["max_new_tokens"])
    shard_size = int(settings["shard_size"])
    batch_size = int(settings["generation_batch_size"])
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        records = []
        for _, batch_rows in chunks(shard_rows, batch_size):
            generated, _ = _batch_generate_and_extract(
                model, tokenizer, batch_rows, "base", generation_settings
            )
            for original, record in zip(batch_rows, generated, strict=True):
                if not str(record["answer"]).strip() or int(record["answer_tokens"]) <= 0:
                    continue
                correct, status = _grade_trivia(record["answer"], original["aliases"])
                records.append(
                    {
                        **record,
                        "candidate_id": f"{DATASET}:base:{original['fact_id']}",
                        "dataset": DATASET,
                        "candidate_source": "base",
                        "aliases": original["aliases"],
                        "subject": original["subject"],
                        "subject_popularity": original["subject_popularity"],
                        "correct": correct,
                        "grading_status": status,
                    }
                )
        atomic_torch_save(
            shard_path,
            {
                "records": records,
                "dataset": DATASET,
                "shard_index": shard_index,
                "fingerprint": fingerprint,
                **artifact_metadata(),
            },
        )
    expected = math.ceil(len(rows) / shard_size)
    actual = len(list(target.glob("shard_*.pt")))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} candidate shards, found {actual}")
    mark_complete(
        done,
        {
            "dataset": DATASET,
            "num_examples": len(rows),
            "num_shards": actual,
            "fingerprint": fingerprint,
        },
    )
    return target


def collect_popqa_base_support(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    from .cross_model_support import (
        _assign_control_questions,
        _load_records,
        _score_rows,
    )
    from .io_utils import atomic_torch_save, chunks, ensure_manifest, mark_complete
    from .modeling import load_base_model, load_tokenizer

    root = base_contextual_transfer_root(config)
    candidates = _load_records(root / "candidates" / DATASET)
    rows = _assign_control_questions(
        candidates,
        repeats=int(settings["control_repeats"]),
        seed=int(settings["seed"]),
        selection="lexical_nearest",
        search_pool=int(settings["control_search_pool"]),
    )
    target = root / "support" / DATASET
    fingerprint = base_contextual_transfer_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    model.eval()
    scoring_settings = {
        "max_sequence_length": int(settings["max_sequence_length"]),
        "max_answer_tokens": int(settings["max_answer_tokens"]),
        "control_repeats": int(settings["control_repeats"]),
    }
    shard_size = int(settings["shard_size"])
    batch_size = int(settings["score_batch_size"])
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        records = []
        for _, batch_rows in chunks(shard_rows, batch_size):
            scores = _score_rows(model, tokenizer, batch_rows, scoring_settings)
            records.extend(
                {**row, **score}
                for row, score in zip(batch_rows, scores, strict=True)
            )
        atomic_torch_save(
            shard_path,
            {
                "records": records,
                "dataset": DATASET,
                "shard_index": shard_index,
                "fingerprint": fingerprint,
                **artifact_metadata(),
            },
        )
    expected = math.ceil(len(rows) / shard_size)
    actual = len(list(target.glob("shard_*.pt")))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} support shards, found {actual}")
    mark_complete(
        done,
        {
            "dataset": DATASET,
            "num_examples": len(rows),
            "num_shards": actual,
            "fingerprint": fingerprint,
        },
    )
    return target


def _evaluate_rows(
    rows: list[dict[str, Any]], settings: dict[str, Any], seed_offset: int = 0
) -> dict[str, Any]:
    from .confidence_gate import _ranking_metrics
    from .cross_model_rank_gate import _cluster_bootstrap_methods, _coverage_metrics

    labels = np.asarray([int(row["correct"]) for row in rows], dtype=int)
    risks = {
        "base_contextual_support": -np.asarray(
            [float(row["support_margin"]) for row in rows], dtype=float
        ),
        "token_logprob": 1.0
        - np.exp(np.asarray([float(row["candidate_logprob"]) for row in rows])),
        "token_entropy": 1.0
        - np.exp(-np.asarray([float(row["candidate_entropy"]) for row in rows])),
    }
    coverages = [float(value) for value in settings["coverages"]]
    comparisons = [
        ("base_contextual_support", "token_logprob"),
        ("base_contextual_support", "token_entropy"),
    ]
    intervals, differences = _cluster_bootstrap_methods(
        rows,
        labels,
        risks,
        coverages,
        int(settings["bootstrap_samples"]),
        int(settings["seed"]) + seed_offset,
        "base_contextual_support",
        comparisons,
    )
    methods = {}
    for method, risk in risks.items():
        gates = {
            str(coverage): _coverage_metrics(labels, risk, coverage)
            for coverage in coverages
        }
        for coverage, metrics in gates.items():
            metrics["ci95"] = intervals[method][coverage]
        methods[method] = {
            "ranking": _ranking_metrics(labels, risk),
            "coverage_gates": gates,
        }
    paired = {}
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        paired[key] = {}
        for coverage, interval in differences[key].items():
            left_accuracy = methods[left]["coverage_gates"][coverage][
                "selective_accuracy"
            ]
            right_accuracy = methods[right]["coverage_gates"][coverage][
                "selective_accuracy"
            ]
            paired[key][coverage] = {
                "selective_accuracy_difference": float(
                    left_accuracy - right_accuracy
                ),
                "ci95": interval,
            }
    return {
        "methods": methods,
        "paired_contextual_minus_confidence": paired,
    }


def _popularity_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    popularity = np.asarray([int(row["subject_popularity"]) for row in rows])
    labels = ("tail", "lower_middle", "upper_middle", "head")
    ordered = np.argsort(popularity, kind="mergesort")
    groups = np.empty(len(rows), dtype=int)
    for index, indices in enumerate(np.array_split(ordered, len(labels))):
        groups[indices] = index
    return {
        label: [row for row, group in zip(rows, groups, strict=True) if group == index]
        for index, label in enumerate(labels)
    }


def _markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    contextual = evaluation["methods"]["base_contextual_support"]
    lines = [
        "# Base-only contextual support transfer to PopQA",
        "",
        "The generated answer is held fixed and scored under its real question and eight "
        "relation-matched, lexically hard control questions. PopQA labels are evaluation-only.",
        "",
        f"Overall answer accuracy: `{report['answer_accuracy']:.4f}`; "
        f"base-contextual AUROC: `{contextual['ranking']['auroc']:.4f}`.",
        "",
        "| Method | AUROC | Target coverage | Actual coverage | Selective accuracy | 95% CI | "
        "Accuracy lift | Error recall | Correct acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_result in evaluation["methods"].items():
        for coverage, metrics in method_result["coverage_gates"].items():
            interval = metrics["ci95"]["selective_accuracy"]
            lines.append(
                (
                    "| {} | {:.4f} | {:.0%} | {:.4f} | {:.4f} | "
                    "[{:.4f}, {:.4f}] | {:.2f}x | {:.4f} | {:.4f} |"
                ).format(
                    method,
                    method_result["ranking"]["auroc"],
                    float(coverage),
                    metrics["coverage"],
                    metrics["selective_accuracy"],
                    interval[0],
                    interval[1],
                    metrics["accuracy_lift"],
                    metrics["error_recall"],
                    metrics["correct_acceptance"],
                )
            )
    lines.extend(
        [
            "",
            "## Paired contextual-support differences",
            "",
            "Positive values favour base contextual support.",
            "",
            "| Baseline | Coverage | Selective-accuracy difference | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, comparisons in evaluation["paired_contextual_minus_confidence"].items():
        baseline = key.removeprefix("base_contextual_support_minus_")
        for coverage, comparison in comparisons.items():
            interval = comparison["ci95"]
            lines.append(
                "| {} | {:.0%} | {:.4f} | [{:.4f}, {:.4f}] |".format(
                    baseline,
                    float(coverage),
                    comparison["selective_accuracy_difference"],
                    interval[0],
                    interval[1],
                )
            )
    lines.extend(
        [
            "",
            "## Popularity strata",
            "",
            "Quartiles use subject popularity only and therefore require no correctness labels.",
            "",
            "| Stratum | N | Answer accuracy | AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in report["popularity_strata"].items():
        lines.append(
            f"| {name} | {result['n']} | {result['answer_accuracy']:.4f} | "
            f"{result['ranking']['auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This run is isolated from the LoRA experiments. It uses only the pretrained base "
            "model and does not train, load, or modify either adapter.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_popqa_base_support(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    from .confidence_gate import _ranking_metrics
    from .cross_model_support import _load_records
    from .io_utils import atomic_json

    root = base_contextual_transfer_root(config)
    target = root / "results" / DATASET
    report_path = target / "report.json"
    if report_path.exists() and not force:
        return report_path
    rows = _load_records(root / "support" / DATASET)
    labels = np.asarray([int(row["correct"]) for row in rows])
    strata = {}
    for name, group in _popularity_groups(rows).items():
        group_labels = np.asarray([int(row["correct"]) for row in group])
        group_risk = -np.asarray([float(row["support_margin"]) for row in group])
        strata[name] = {
            "n": len(group),
            "answer_accuracy": float(group_labels.mean()),
            "ranking": _ranking_metrics(group_labels, group_risk),
        }
    report = {
        "method": f"base_contextual_transfer_v{SCHEMA_VERSION}",
        "dataset": DATASET,
        "n": len(rows),
        "answer_accuracy": float(labels.mean()),
        "controls": {
            "selection": "lexical_nearest",
            "repeats": int(settings["control_repeats"]),
            "search_pool": int(settings["control_search_pool"]),
            "relation_matched": True,
        },
        "evaluation": _evaluate_rows(rows, settings),
        "popularity_strata": strata,
        **artifact_metadata(),
    }
    atomic_json(report_path, report)
    (target / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report_path
