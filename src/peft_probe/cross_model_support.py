from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .confidence_gate import _fit_threshold, _ranking_metrics, _threshold_metrics
from .config import config_fingerprint, output_dir, validate_source_run
from .versioning import artifact_metadata


SCHEMA_VERSION = 2
EVALUATORS = ("base", "adapter_a", "adapter_b")


def load_cross_model_support_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Cross-model support configuration must be a mapping")
    if int(settings.get("cross_model_support_schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"cross_model_support_schema_version must be {SCHEMA_VERSION}")
    if settings.get("synthetic_candidate_sources") != ["adapter_a", "adapter_b"]:
        raise ValueError(
            "synthetic_candidate_sources must be [adapter_a, adapter_b] so training is balanced"
        )
    datasets = settings.get("datasets", [])
    allowed_datasets = {"synthetic", "trivia_qa", "truthful_qa"}
    if not datasets or any(name not in allowed_datasets for name in datasets):
        raise ValueError(f"datasets must use {sorted(allowed_datasets)}")
    feature_sets = settings.get("feature_sets", {})
    if not isinstance(feature_sets, dict) or not feature_sets:
        raise ValueError("feature_sets must be a non-empty mapping")
    for name, features in feature_sets.items():
        if not isinstance(features, list) or not features:
            raise ValueError(f"feature_sets.{name} must be a non-empty list")
    for key in ("batch_size", "shard_size", "max_answer_tokens"):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if not settings.get("regularization_candidates"):
        raise ValueError("regularization_candidates must not be empty")
    if not 0.5 <= float(settings.get("max_source_identity_auroc", 0.0)) <= 1.0:
        raise ValueError("max_source_identity_auroc must be in [0.5, 1.0]")
    return settings


def cross_model_support_fingerprint(
    config: dict[str, Any], settings: dict[str, Any]
) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "cross_model_support": settings,
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def cross_model_support_root(config: dict[str, Any]) -> Path:
    return output_dir(config) / "analysis" / f"cross_model_support_v{SCHEMA_VERSION}"


def _load_records(directory: Path) -> list[dict[str, Any]]:
    import torch

    if not (directory / "_SUCCESS.json").is_file():
        raise FileNotFoundError(f"Saved generation stage is incomplete: {directory}")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("shard_*.pt")):
        records.extend(torch.load(path, map_location="cpu", weights_only=False)["records"])
    if not records:
        raise FileNotFoundError(f"No saved generation records found in {directory}")
    return records


def _candidate_rows(
    config: dict[str, Any], settings: dict[str, Any], dataset: str
) -> list[dict[str, Any]]:
    root = output_dir(config)
    if dataset == "synthetic":
        source_names = settings["synthetic_candidate_sources"]
        groups = [
            (source, _load_records(root / "features" / "synthetic" / source))
            for source in source_names
        ]
    else:
        groups = [
            (
                "base",
                _load_records(root / "features" / "benchmarks" / dataset),
            )
        ]
    rows = []
    for candidate_source, records in groups:
        for record in records:
            answer = str(record.get("answer", "")).strip()
            if not answer or int(record.get("answer_tokens", 1)) <= 0:
                continue
            rows.append(
                {
                    "candidate_id": f"{dataset}:{candidate_source}:{record['fact_id']}",
                    "fact_id": record["fact_id"],
                    "entity_id": record["entity_id"],
                    "dataset": dataset,
                    "relation": record["relation"],
                    "split": record.get("split", "benchmark"),
                    "candidate_source": candidate_source,
                    "question": record["question"],
                    "answer": answer,
                    "expected": str(record["expected"]),
                    "correct": int(record["correct"]),
                    "abstained": int(record.get("abstained", 0)),
                }
            )
    if not rows:
        raise RuntimeError(f"No non-empty candidate answers were found for {dataset}")
    return _assign_control_answers(rows, int(settings["seed"]))


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _choice_letters(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?m)^([A-Z])\.\s", question)))


def _assign_control_answers(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        key = (row["dataset"], row["split"], row["relation"])
        # Controls come from model responses, not reference answers. This keeps natural
        # benchmark annotations strictly evaluation-only.
        pools.setdefault(key, []).append(str(row["answer"]).strip())
    pools = {
        key: sorted(set(values), key=lambda value: (_normalized(value), value))
        for key, values in pools.items()
    }
    assigned = []
    for row in rows:
        key = (row["dataset"], row["split"], row["relation"])
        pool = pools[key]
        forbidden = {_normalized(row["answer"])}
        digest = hashlib.sha256(f"{seed}:{row['candidate_id']}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % max(len(pool), 1)
        control = next(
            (
                pool[(start + offset) % len(pool)]
                for offset in range(len(pool))
                if _normalized(pool[(start + offset) % len(pool)]) not in forbidden
            ),
            None,
        )
        if control is None:
            choices = _choice_letters(row["question"])
            control = next(
                (
                    choices[(start + offset) % len(choices)]
                    for offset in range(len(choices))
                    if _normalized(choices[(start + offset) % len(choices)])
                    not in forbidden
                ),
                None,
            )
        if control is None:
            raise RuntimeError(
                f"Could not construct a distinct label-free control answer for "
                f"{row['candidate_id']}"
            )
        assigned.append({**row, "control_answer": control})
    return assigned


def _encoded_answer(
    tokenizer,
    question: str,
    answer: str,
    max_sequence_length: int,
    max_answer_tokens: int,
) -> tuple[list[int], list[int]]:
    from .modeling import user_prompt

    prompt_ids = list(
        tokenizer(user_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]
    )
    answer_ids = list(
        tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    )[:max_answer_tokens]
    if not answer_ids:
        raise ValueError("Candidate answer tokenized to zero tokens")
    prompt_budget = max_sequence_length - len(answer_ids)
    if prompt_budget < 1:
        raise ValueError("Answer consumes the complete model sequence length")
    prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids + answer_ids, [0] * len(prompt_ids) + [1] * len(answer_ids)


def _score_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, float]]:
    import torch

    from .modeling import model_device

    encoded = []
    max_length = int(settings["max_sequence_length"])
    max_answer_tokens = int(settings["max_answer_tokens"])
    for row in rows:
        encoded.append(
            _encoded_answer(
                tokenizer, row["question"], row["answer"], max_length, max_answer_tokens
            )
        )
        encoded.append(
            _encoded_answer(
                tokenizer,
                row["question"],
                row["control_answer"],
                max_length,
                max_answer_tokens,
            )
        )
    width = max(len(ids) for ids, _ in encoded)
    input_ids = []
    attention_mask = []
    answer_mask = []
    for ids, mask in encoded:
        padding = width - len(ids)
        input_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_mask.append([0] * padding + [1] * len(ids))
        answer_mask.append([0] * padding + mask)
    device = model_device(model)
    tokens = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }
    mask = torch.tensor(answer_mask, dtype=torch.bool, device=device)[:, 1:]
    with torch.inference_mode():
        outputs = model(**tokens, use_cache=False, return_dict=True)
        logits = outputs.logits[:, :-1].float()
        targets = tokens["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        counts = mask.sum(dim=1).clamp_min(1)
        mean_logprob = (chosen * mask).sum(dim=1) / counts
        mean_entropy = (entropy * mask).sum(dim=1) / counts
    result = []
    for index in range(len(rows)):
        candidate_index = index * 2
        negative_index = candidate_index + 1
        result.append(
            {
                "candidate_logprob": float(mean_logprob[candidate_index].cpu()),
                "control_logprob": float(mean_logprob[negative_index].cpu()),
                "support_margin": float(
                    (mean_logprob[candidate_index] - mean_logprob[negative_index]).cpu()
                ),
                "candidate_entropy": float(mean_entropy[candidate_index].cpu()),
                "control_entropy": float(mean_entropy[negative_index].cpu()),
            }
        )
    return result


def collect_cross_model_support(
    config: dict[str, Any],
    settings: dict[str, Any],
    evaluator: str,
    force: bool = False,
) -> Path:
    from peft import PeftModel

    from .io_utils import atomic_torch_save, chunks, ensure_manifest, mark_complete
    from .modeling import load_base_model, load_tokenizer

    if evaluator not in EVALUATORS:
        raise ValueError(f"evaluator must be one of {EVALUATORS}")
    root = cross_model_support_root(config)
    fingerprint = cross_model_support_fingerprint(config, settings)
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    if evaluator != "base":
        adapter_path = validate_source_run(config) / "adapters" / evaluator / "final"
        if not (adapter_path / "_SUCCESS.json").exists():
            raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    for dataset in settings["datasets"]:
        target = root / "features" / dataset / evaluator
        ensure_manifest(target, fingerprint, force=force)
        done = target / "_SUCCESS.json"
        if done.exists() and not force:
            continue
        rows = _candidate_rows(config, settings, dataset)
        shard_size = int(settings["shard_size"])
        batch_size = int(settings["batch_size"])
        for shard_index, shard_rows in chunks(rows, shard_size):
            shard_path = target / f"shard_{shard_index:05d}.pt"
            if shard_path.exists() and not force:
                continue
            records = []
            for _, batch_rows in chunks(shard_rows, batch_size):
                scores = _score_rows(model, tokenizer, batch_rows, settings)
                records.extend(
                    {**row, **score, "evaluator": evaluator}
                    for row, score in zip(batch_rows, scores, strict=True)
                )
            atomic_torch_save(
                shard_path,
                {
                    "records": records,
                    "dataset": dataset,
                    "evaluator": evaluator,
                    "shard_index": shard_index,
                    "cross_model_support_fingerprint": fingerprint,
                    **artifact_metadata(),
                },
            )
        expected_shards = math.ceil(len(rows) / shard_size)
        actual_shards = len(list(target.glob("shard_*.pt")))
        if actual_shards != expected_shards:
            raise RuntimeError(
                f"Expected {expected_shards} {dataset}/{evaluator} shards, "
                f"found {actual_shards}"
            )
        mark_complete(
            done,
            {
                "dataset": dataset,
                "evaluator": evaluator,
                "num_examples": len(rows),
                "num_shards": actual_shards,
                "cross_model_support_fingerprint": fingerprint,
            },
        )
    return root / "features"


def _aligned_dataset(root: Path, dataset: str) -> list[dict[str, Any]]:
    by_evaluator = {
        evaluator: _load_records(root / "features" / dataset / evaluator)
        for evaluator in EVALUATORS
    }
    indexed = {
        evaluator: {row["candidate_id"]: row for row in rows}
        for evaluator, rows in by_evaluator.items()
    }
    ids = set(indexed["base"])
    for evaluator in EVALUATORS[1:]:
        if set(indexed[evaluator]) != ids:
            raise RuntimeError(f"Candidate alignment differs for {dataset}/{evaluator}")
    result = []
    for candidate_id in sorted(ids):
        base_row = indexed["base"][candidate_id]
        result.append(
            {
                **{
                    key: base_row[key]
                    for key in (
                        "candidate_id",
                        "fact_id",
                        "entity_id",
                        "dataset",
                        "relation",
                        "split",
                        "candidate_source",
                        "question",
                        "answer",
                        "expected",
                        "correct",
                        "abstained",
                        "control_answer",
                    )
                },
                "scores": {
                    evaluator: {
                        key: indexed[evaluator][candidate_id][key]
                        for key in (
                            "candidate_logprob",
                            "control_logprob",
                            "support_margin",
                            "candidate_entropy",
                        )
                    }
                    for evaluator in EVALUATORS
                },
            }
        )
    return result


def _feature_table(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = [
        "base_support",
        "adapter_a_support",
        "adapter_b_support",
        "mean_support",
        "min_support",
        "max_support",
        "support_std",
        "common_lora_shift",
        "adapter_opposition",
    ]
    values = []
    for row in rows:
        base = float(row["scores"]["base"]["support_margin"])
        adapter_a = float(row["scores"]["adapter_a"]["support_margin"])
        adapter_b = float(row["scores"]["adapter_b"]["support_margin"])
        supports = np.asarray([base, adapter_a, adapter_b])
        values.append(
            [
                base,
                adapter_a,
                adapter_b,
                float(supports.mean()),
                float(supports.min()),
                float(supports.max()),
                float(supports.std()),
                float((adapter_a + adapter_b) / 2.0 - base),
                float(abs(adapter_a - adapter_b)),
            ]
        )
    return names, np.asarray(values, dtype=float)


def _fit_model(
    features: np.ndarray, labels: np.ndarray, regularization: float, seed: int
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    model = LogisticRegression(
        C=float(regularization),
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit((features - mean) / std, labels)
    return {"model": model, "mean": mean, "std": std}


def _correct_probability(fitted: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (features - fitted["mean"]) / fitted["std"]
    model = fitted["model"]
    class_index = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(standardized)[:, class_index]


def _subset_features(
    all_names: list[str], matrix: np.ndarray, requested: list[str]
) -> np.ndarray:
    missing = sorted(set(requested) - set(all_names))
    if missing:
        raise ValueError(f"Unknown cross-model support features: {missing}")
    indices = [all_names.index(name) for name in requested]
    return matrix[:, indices]


def _safe_auroc(labels_correct: np.ndarray, correctness_score: np.ndarray) -> float:
    labels = np.asarray(labels_correct, dtype=int)
    scores = np.asarray(correctness_score, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _candidate_source_identity(
    rows: list[dict[str, Any]], correctness_score: np.ndarray
) -> float:
    labels = np.asarray([row["candidate_source"] == "adapter_b" for row in rows], dtype=int)
    if np.unique(labels).size < 2:
        return float("nan")
    auc = _safe_auroc(labels, correctness_score)
    return max(auc, 1.0 - auc)


def _model_candidate(
    rows: list[dict[str, Any]],
    names: list[str],
    matrix: np.ndarray,
    feature_names: list[str],
    regularization: float,
    seed: int,
) -> dict[str, Any]:
    selected = _subset_features(names, matrix, feature_names)
    labels = np.asarray([row["correct"] for row in rows], dtype=int)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    validation_mask = np.asarray([row["split"] == "validation" for row in rows])
    relations = sorted({row["relation"] for row in rows})
    held_out = {}
    for relation in relations:
        fold_train = train_mask & np.asarray([row["relation"] != relation for row in rows])
        fold_validation = validation_mask & np.asarray(
            [row["relation"] == relation for row in rows]
        )
        fitted = _fit_model(selected[fold_train], labels[fold_train], regularization, seed)
        probability = _correct_probability(fitted, selected[fold_validation])
        held_out[relation] = _safe_auroc(labels[fold_validation], probability)
    fitted = _fit_model(selected[train_mask], labels[train_mask], regularization, seed)
    validation_rows = [row for row, keep in zip(rows, validation_mask, strict=True) if keep]
    validation_probability = _correct_probability(fitted, selected[validation_mask])
    validation_labels = labels[validation_mask]
    risk = 1.0 - validation_probability
    threshold, threshold_metrics = _fit_threshold(validation_labels, risk)
    per_source = {}
    for source in ("adapter_a", "adapter_b"):
        source_mask = np.asarray(
            [row["candidate_source"] == source for row in validation_rows]
        )
        per_source[source] = _safe_auroc(
            validation_labels[source_mask], validation_probability[source_mask]
        )
    return {
        "feature_names": feature_names,
        "regularization": float(regularization),
        "worst_held_out_relation_auroc": float(min(held_out.values())),
        "held_out_relation_auroc": held_out,
        "validation_auroc": _safe_auroc(validation_labels, validation_probability),
        "worst_candidate_source_auroc": float(min(per_source.values())),
        "candidate_source_auroc": per_source,
        "source_identity_auroc": _candidate_source_identity(
            validation_rows, validation_probability
        ),
        "risk_threshold": float(threshold),
        "validation_threshold_metrics": threshold_metrics,
    }


def _evaluate_model(
    rows: list[dict[str, Any]],
    fitted: dict[str, Any],
    names: list[str],
    matrix: np.ndarray,
    feature_names: list[str],
    risk_threshold: float,
) -> dict[str, Any]:
    selected = _subset_features(names, matrix, feature_names)
    correct_probability = _correct_probability(fitted, selected)
    risk = 1.0 - correct_probability
    labels = np.asarray([row["correct"] for row in rows], dtype=int)
    return {
        **_ranking_metrics(labels, risk),
        **_threshold_metrics(labels, risk, risk_threshold),
    }


def _markdown(report: dict[str, Any]) -> str:
    selection = report["selection"]
    lines = [
        "# Cross-model normalized answer support",
        "",
        "All model and feature selection used synthetic train/validation records only. "
        "Natural benchmark labels are evaluation-only.",
        "",
        f"Selected feature set: `{selection['feature_set']}`; "
        f"regularization C: `{selection['regularization']}`; "
        f"risk threshold: `{selection['risk_threshold']:.6f}`.",
        "",
        "| Evaluation | N | AUROC | Balanced accuracy | Error recall | "
        "Correct acceptance | Coverage | Selective accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def add_row(name: str, metrics: dict[str, Any]) -> None:
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                name,
                metrics["n"],
                metrics["auroc"],
                metrics["balanced_accuracy"],
                metrics["error_recall"],
                metrics["correct_acceptance"],
                metrics["coverage"],
                metrics["selective_accuracy"],
            )
        )

    add_row("Synthetic validation", report["synthetic_validation"])
    add_row("Synthetic test", report["synthetic_test"]["all"])
    for name, metrics in report["benchmarks"].items():
        add_row(name, metrics)
    lines.extend(
        [
            "",
            f"Validation source-identity AUROC: "
            f"{selection['source_identity_auroc']:.4f}. "
            f"Worst held-out-relation AUROC: "
            f"{selection['worst_held_out_relation_auroc']:.4f}.",
            "",
            "The support margin compares the same candidate answer with a deterministic, "
            "label-free relation-matched control under base, LoRA A, and LoRA B. No natural "
            "labels orient or recalibrate the detector.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_cross_model_support(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    from .io_utils import atomic_json

    root = cross_model_support_root(config)
    target = root / "results"
    report_path = target / "report.json"
    if report_path.exists() and not force:
        return report_path
    target.mkdir(parents=True, exist_ok=True)
    synthetic = _aligned_dataset(root, "synthetic")
    names, matrix = _feature_table(synthetic)
    candidates = []
    for feature_set, feature_names in settings["feature_sets"].items():
        for regularization in settings["regularization_candidates"]:
            result = _model_candidate(
                synthetic,
                names,
                matrix,
                feature_names,
                float(regularization),
                int(settings["seed"]),
            )
            result["feature_set"] = feature_set
            candidates.append(result)
    maximum_identity = float(settings["max_source_identity_auroc"])
    eligible = [
        item for item in candidates if item["source_identity_auroc"] <= maximum_identity
    ]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["worst_held_out_relation_auroc"],
            item["worst_candidate_source_auroc"],
            item["validation_auroc"],
            -item["source_identity_auroc"],
        ),
    )
    requested = selected["feature_names"]
    selected_matrix = _subset_features(names, matrix, requested)
    labels = np.asarray([row["correct"] for row in synthetic], dtype=int)
    train_mask = np.asarray([row["split"] == "train" for row in synthetic])
    validation_mask = np.asarray([row["split"] == "validation" for row in synthetic])
    test_mask = np.asarray([row["split"] == "test" for row in synthetic])
    fitted = _fit_model(
        selected_matrix[train_mask],
        labels[train_mask],
        float(selected["regularization"]),
        int(settings["seed"]),
    )
    threshold = float(selected["risk_threshold"])

    def rows_for(mask: np.ndarray) -> list[dict[str, Any]]:
        return [row for row, keep in zip(synthetic, mask, strict=True) if keep]

    validation_rows = rows_for(validation_mask)
    test_rows = rows_for(test_mask)
    validation_metrics = _evaluate_model(
        validation_rows,
        fitted,
        names,
        matrix[validation_mask],
        requested,
        threshold,
    )
    test_groups = {"all": np.ones(int(test_mask.sum()), dtype=bool)}
    for source in ("adapter_a", "adapter_b"):
        test_groups[source] = np.asarray(
            [row["candidate_source"] == source for row in test_rows]
        )
    for relation in sorted({row["relation"] for row in test_rows}):
        test_groups[f"relation/{relation}"] = np.asarray(
            [row["relation"] == relation for row in test_rows]
        )
    synthetic_test = {
        name: _evaluate_model(
            [row for row, keep in zip(test_rows, mask, strict=True) if keep],
            fitted,
            names,
            matrix[test_mask][mask],
            requested,
            threshold,
        )
        for name, mask in test_groups.items()
    }
    benchmarks = {}
    for dataset in settings["datasets"]:
        if dataset == "synthetic":
            continue
        rows = [row for row in _aligned_dataset(root, dataset) if int(row["correct"]) >= 0]
        benchmark_names, benchmark_matrix = _feature_table(rows)
        if benchmark_names != names:
            raise RuntimeError("Feature schema changed between synthetic and benchmark data")
        benchmarks[dataset] = _evaluate_model(
            rows,
            fitted,
            names,
            benchmark_matrix,
            requested,
            threshold,
        )
    model = fitted["model"]
    report = {
        "method": f"cross_model_support_v{SCHEMA_VERSION}",
        "selection_constraints_satisfied": bool(eligible),
        "max_source_identity_auroc": maximum_identity,
        "selection": selected,
        "candidate_selection_results": candidates,
        "synthetic_validation": validation_metrics,
        "synthetic_test": synthetic_test,
        "benchmarks": benchmarks,
        "fitted_model": {
            "feature_names": requested,
            "mean": fitted["mean"].tolist(),
            "std": fitted["std"].tolist(),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "classes": model.classes_.tolist(),
        },
        **artifact_metadata(),
    }
    atomic_json(report_path, report)
    (target / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report_path
