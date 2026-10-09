from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel

from .collect import load_feature_shards
from .config import config_fingerprint, output_dir
from .geometry import (
    _fit_subspace,
    _paired_arrays,
    _pool_prompt,
    _ranking_metrics,
    geometry_fingerprint,
    geometry_root,
)
from .io_utils import (
    atomic_json,
    atomic_torch_save,
    chunks,
    ensure_manifest,
    mark_complete,
    read_jsonl,
)
from .modeling import load_base_model, load_tokenizer, model_device, user_prompt
from .versioning import artifact_metadata


COMPARISON_SCHEMA_VERSION = 1


def load_comparison_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Comparison configuration must be a mapping")
    if int(settings.get("comparison_schema_version", -1)) != COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"Expected comparison schema {COMPARISON_SCHEMA_VERSION}, "
            f"found {settings.get('comparison_schema_version')}"
        )
    layers = settings.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError("comparison.layers must be a non-empty list of integers")
    k_values = settings.get("k_values")
    if not isinstance(k_values, list) or not k_values or any(int(k) <= 0 for k in k_values):
        raise ValueError("comparison.k_values must contain positive integers")
    for key in ("batch_size", "shard_size", "bootstrap_samples"):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"comparison.{key} must be positive")
    wild = float(settings.get("wild_fraction", 0))
    validation = float(settings.get("validation_fraction", 0))
    if wild <= 0 or validation <= 0 or wild + validation >= 1:
        raise ValueError("wild_fraction and validation_fraction must be positive and sum below 1")
    int(settings["seed"])
    return settings


def comparison_fingerprint(
    config: dict[str, Any], geometry_settings: dict[str, Any], settings: dict[str, Any]
) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "geometry_fingerprint": geometry_fingerprint(config, geometry_settings),
        "comparison": settings,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def comparison_root(config: dict[str, Any]) -> Path:
    return output_dir(config) / "comparisons" / f"haloscope_matched_v{COMPARISON_SCHEMA_VERSION}"


def _metadata(
    config: dict[str, Any], geometry_settings: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_fingerprint": comparison_fingerprint(config, geometry_settings, settings),
        **artifact_metadata(),
    }


def _validate_matched_layers(
    geometry_settings: dict[str, Any], settings: dict[str, Any]
) -> None:
    if list(settings["layers"]) != list(geometry_settings["layers"]):
        raise ValueError(
            "comparison.layers must exactly match geometry.layers so both methods receive "
            "the same layer-selection budget"
        )


@torch.inference_mode()
def _extract_answer_conditioned(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    layers: list[int],
) -> dict[str, torch.Tensor]:
    texts = []
    for row in rows:
        prompt = user_prompt(tokenizer, row["question"])
        answer = row.get("common_answer", row.get("answer", "")).strip()
        texts.append(prompt + ((" " + answer) if answer else ""))
    tokens = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = model_device(model)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    outputs = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states
    features: dict[str, torch.Tensor] = {}
    resolved: set[int] = set()
    for requested in layers:
        index = requested if requested >= 0 else len(hidden_states) + requested
        if not 0 <= index < len(hidden_states):
            raise ValueError(
                f"Layer {requested} is invalid for {len(hidden_states)} hidden states"
            )
        if index in resolved:
            raise ValueError(f"Comparison layers resolve to duplicate hidden state {index}")
        resolved.add(index)
        pooled = _pool_prompt(hidden_states[index], tokens["attention_mask"], "last")
        features[str(requested)] = pooled.detach().to(dtype=torch.float16, device="cpu")
    return features


def _collect_rows(
    config: dict[str, Any],
    geometry_settings: dict[str, Any],
    settings: dict[str, Any],
    target: Path,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    kind: str,
    force: bool,
) -> Path:
    fingerprint = comparison_fingerprint(config, geometry_settings, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target
    shard_size = int(settings["shard_size"])
    batch_size = int(settings["batch_size"])
    expected = math.ceil(len(rows) / shard_size)
    for shard_index, shard_rows in chunks(rows, shard_size):
        path = target / f"shard_{shard_index:05d}.pt"
        if path.exists() and not force:
            continue
        records = []
        parts = {str(layer): [] for layer in settings["layers"]}
        for _, batch_rows in chunks(shard_rows, batch_size):
            batch_features = _extract_answer_conditioned(
                model, tokenizer, batch_rows, list(settings["layers"])
            )
            records.extend(batch_rows)
            for layer, values in batch_features.items():
                parts[layer].append(values)
        atomic_torch_save(
            path,
            {
                "records": records,
                "features": {layer: torch.cat(values) for layer, values in parts.items()},
                "kind": kind,
                "shard_index": shard_index,
                **_metadata(config, geometry_settings, settings),
            },
        )
    actual = len(list(target.glob("shard_*.pt")))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} comparison shards, found {actual}")
    mark_complete(
        done,
        {
            "kind": kind,
            "num_examples": len(rows),
            "num_shards": actual,
            **_metadata(config, geometry_settings, settings),
        },
    )
    return target


def _synthetic_common_answer_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = output_dir(config)
    records, _ = load_feature_shards(
        root / "features" / "synthetic" / "base",
        config["collection"]["feature_modes"][0],
    )
    rows = []
    for record in records:
        if record["adapter_assignment"] not in {"a", "b"}:
            continue
        rows.append(
            {
                "fact_id": record["fact_id"],
                "entity_id": record["entity_id"],
                "relation": record["relation"],
                "split": record["split"],
                "adapter_assignment": record["adapter_assignment"],
                "question": record["question"],
                "common_answer": record["answer"],
            }
        )
    return sorted(rows, key=lambda row: row["fact_id"])


def collect_comparison_synthetic(
    config: dict[str, Any],
    geometry_settings: dict[str, Any],
    settings: dict[str, Any],
    source: str,
    force: bool = False,
) -> Path:
    _validate_matched_layers(geometry_settings, settings)
    if source not in {"adapter_a", "adapter_b"}:
        raise ValueError("Comparison synthetic source must be adapter_a or adapter_b")
    root = output_dir(config)
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    adapter_path = root / "adapters" / source / "final"
    if not (adapter_path / "_SUCCESS.json").exists():
        raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return _collect_rows(
        config,
        geometry_settings,
        settings,
        comparison_root(config) / "features" / "synthetic" / source,
        _synthetic_common_answer_rows(config),
        model,
        tokenizer,
        source,
        force,
    )


def collect_comparison_benchmark(
    config: dict[str, Any],
    geometry_settings: dict[str, Any],
    settings: dict[str, Any],
    benchmark: str,
    force: bool = False,
) -> Path:
    _validate_matched_layers(geometry_settings, settings)
    predictions = read_jsonl(output_dir(config) / "results" / benchmark / "predictions.jsonl")
    rows = [
        {
            "fact_id": record["fact_id"],
            "question": record["question"],
            "answer": record["answer"],
            "correct": record["correct"],
            "grading_status": record["grading_status"],
            "mean_logprob": record["mean_logprob"],
            "mean_entropy": record["mean_entropy"],
            "benchmark": benchmark,
        }
        for record in predictions
    ]
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    model.eval()
    return _collect_rows(
        config,
        geometry_settings,
        settings,
        comparison_root(config) / "features" / "benchmarks" / benchmark,
        rows,
        model,
        tokenizer,
        benchmark,
        force,
    )


def _load_feature_layers(
    directory: Path, layers: list[int]
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    records = []
    parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for path in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        for layer in layers:
            parts[layer].append(payload["features"][str(layer)].float())
    if not records:
        raise FileNotFoundError(f"No comparison features found in {directory}")
    return records, {layer: torch.cat(values) for layer, values in parts.items()}


def _split_indices(
    labels: np.ndarray, settings: dict[str, Any]
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(settings["seed"]))
    # Stratification prevents TruthfulQA's minority correct class from becoming unstable.
    partitions: dict[str, list[int]] = {"wild": [], "validation": [], "test": []}
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        wild_end = round(len(indices) * float(settings["wild_fraction"]))
        validation_end = wild_end + round(
            len(indices) * float(settings["validation_fraction"])
        )
        partitions["wild"].extend(indices[:wild_end].tolist())
        partitions["validation"].extend(indices[wild_end:validation_end].tolist())
        partitions["test"].extend(indices[validation_end:].tolist())
    return {
        name: np.asarray(sorted(indices), dtype=int)
        for name, indices in partitions.items()
    }


def _centered_pca(
    features: torch.Tensor, maximum_k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = features.to(device)
    mean = work.mean(0)
    centered = work - mean
    q = min(int(maximum_k), centered.shape[0], centered.shape[1])
    if q < 1 or float(centered.square().sum()) <= 1e-12:
        raise ValueError("Cannot fit HaloScope-style PCA to constant or empty features")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _, singular_values, vectors = torch.pca_lowrank(
            centered, q=q, center=False, niter=4
        )
    return mean.cpu(), vectors.cpu(), singular_values.cpu()


def _haloscope_energy(
    features: torch.Tensor,
    mean: torch.Tensor,
    vectors: torch.Tensor,
    singular_values: torch.Tensor,
    k: int,
) -> np.ndarray:
    projection = (features - mean) @ vectors[:, :k]
    weights = singular_values[:k] / singular_values[:k].mean().clamp_min(1e-12)
    return (projection.square() * weights).mean(1).numpy()


def _select_haloscope_style(
    features_by_layer: dict[int, torch.Tensor],
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[int, dict[str, torch.Tensor]]]:
    best: dict[str, Any] | None = None
    fitted: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index, (layer, features) in enumerate(features_by_layer.items()):
        mean, vectors, singular_values = _centered_pca(
            features[split["wild"]],
            max(int(k) for k in settings["k_values"]),
            int(settings["seed"]) + layer_index,
        )
        fitted[layer] = {
            "mean": mean,
            "vectors": vectors,
            "singular_values": singular_values,
        }
        for k in settings["k_values"]:
            k = int(k)
            if k > vectors.shape[1]:
                continue
            score = _haloscope_energy(
                features[split["validation"]], mean, vectors, singular_values, k
            )
            candidates = ((1, score), (-1, -score))
            for sign, risk in candidates:
                metrics = _ranking_metrics(labels[split["validation"]], risk)
                candidate = {
                    "layer": layer,
                    "k": k,
                    "sign": sign,
                    "validation_auroc": metrics["auroc"],
                }
                if best is None or candidate["validation_auroc"] > best["validation_auroc"]:
                    best = candidate
    if best is None:
        raise RuntimeError("HaloScope-style validation selection produced no candidate")
    selected_features = features_by_layer[int(best["layer"])]
    selected_fit = fitted[int(best["layer"])]
    all_scores = best["sign"] * _haloscope_energy(
        selected_features,
        selected_fit["mean"],
        selected_fit["vectors"],
        selected_fit["singular_values"],
        int(best["k"]),
    )
    return best, all_scores, fitted


def _select_controlled_direction(
    features_by_layer: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    labels: np.ndarray,
    split: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    best = None
    selected_score = None
    for layer, features in features_by_layer.items():
        score = (-(features @ directions[layer].float())).numpy()
        metrics = _ranking_metrics(labels[split["validation"]], score[split["validation"]])
        candidate = {"layer": layer, "validation_auroc": metrics["auroc"]}
        if best is None or candidate["validation_auroc"] > best["validation_auroc"]:
            best = candidate
            selected_score = score
    if best is None or selected_score is None:
        raise RuntimeError("Controlled-direction validation selection produced no candidate")
    return best, selected_score


def _paired_ranking_differences(
    labels: np.ndarray,
    proposed: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "aurc")
    observed_proposed = _ranking_metrics(labels, proposed)
    observed_baseline = _ranking_metrics(labels, baseline)
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        first = _ranking_metrics(labels[indices], proposed[indices])
        second = _ranking_metrics(labels[indices], baseline[indices])
        for name in names:
            difference = first[name] - second[name]
            if np.isfinite(difference):
                draws[name].append(float(difference))
    return {
        name: {
            "observed_difference": float(observed_proposed[name] - observed_baseline[name]),
            "ci95": (
                [float(np.quantile(draws[name], 0.025)), float(np.quantile(draws[name], 0.975))]
                if draws[name]
                else [float("nan"), float("nan")]
            ),
        }
        for name in names
    }


def _load_prompt_features(
    config: dict[str, Any], layers: list[int], benchmark: str
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    directory = geometry_root(config) / "features" / "benchmarks" / benchmark
    return _load_feature_layers(directory, layers)


def _load_answer_features(
    config: dict[str, Any], layers: list[int], benchmark: str
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    directory = comparison_root(config) / "features" / "benchmarks" / benchmark
    return _load_feature_layers(directory, layers)


def _fit_answer_memory_subspaces(
    config: dict[str, Any],
    geometry_settings: dict[str, Any],
    settings: dict[str, Any],
    force: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    target = comparison_root(config) / "results" / "answer_memory_subspaces.pt"
    fingerprint = comparison_fingerprint(config, geometry_settings, settings)
    if target.exists() and not force:
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("comparison_fingerprint") != fingerprint:
            raise RuntimeError(f"Stale comparison subspace checkpoint: {target}")
        return {int(layer): values for layer, values in payload["layers"].items()}
    directory = comparison_root(config) / "features" / "synthetic"
    records_a, layers_a = _load_feature_layers(
        directory / "adapter_a", [int(layer) for layer in settings["layers"]]
    )
    records_b, layers_b = _load_feature_layers(
        directory / "adapter_b", [int(layer) for layer in settings["layers"]]
    )
    result = {}
    for layer_index, layer in enumerate(settings["layers"]):
        records, exposed, withheld = _paired_arrays(
            records_a, layers_a[int(layer)], records_b, layers_b[int(layer)]
        )
        train = np.asarray([record["split"] == "train" for record in records])
        delta = exposed - withheld
        basis, spectrum = _fit_subspace(
            delta[train],
            int(geometry_settings["max_components"]),
            float(geometry_settings["variance_threshold"]),
            int(settings["seed"]) + layer_index,
        )
        direction = delta[train].mean(0)
        direction = direction / direction.norm().clamp_min(1e-12)
        result[int(layer)] = {
            "basis": basis.half(),
            "mean_direction": direction.half(),
            "rank": spectrum["rank"],
        }
    atomic_torch_save(
        target,
        {
            "layers": {str(layer): values for layer, values in result.items()},
            **_metadata(config, geometry_settings, settings),
        },
    )
    return result


def _load_prompt_directions(config: dict[str, Any]) -> dict[int, torch.Tensor]:
    path = geometry_root(config) / "results" / "memory_subspaces.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        int(layer): values["mean_direction"].float()
        for layer, values in payload["layers"].items()
    }


def _baseline_scores(
    config: dict[str, Any], benchmark: str, fact_ids: list[str]
) -> dict[str, np.ndarray]:
    predictions = read_jsonl(output_dir(config) / "results" / benchmark / "predictions.jsonl")
    by_id = {record["fact_id"]: record for record in predictions}
    ordered = [by_id[fact_id] for fact_id in fact_ids]
    scores = {
        "token_logprob": 1.0
        - np.exp(np.asarray([record["mean_logprob"] for record in ordered], dtype=float)),
        "token_entropy": 1.0
        - np.exp(-np.asarray([record["mean_entropy"] for record in ordered], dtype=float)),
    }
    probe_keys = sorted(
        key
        for key in ordered[0]
        if "/seed_" in key and key.startswith(("base_replay/", "on_policy/"))
    )
    grouped: dict[str, list[str]] = {}
    for key in probe_keys:
        prefix = key.rsplit("/seed_", 1)[0]
        grouped.setdefault(prefix, []).append(key)
    for prefix, keys in grouped.items():
        scores[f"probe_ensemble/{prefix}"] = np.mean(
            [[float(record[key]) for record in ordered] for key in keys], axis=0
        )
    return scores


def analyze_matched_comparison(
    config: dict[str, Any],
    geometry_settings: dict[str, Any],
    settings: dict[str, Any],
    force: bool = False,
) -> Path:
    _validate_matched_layers(geometry_settings, settings)
    target = comparison_root(config) / "results"
    fingerprint = comparison_fingerprint(config, geometry_settings, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    metrics_path = target / "matched_metrics.json"
    if done.exists() and metrics_path.exists() and not force:
        return metrics_path

    prompt_directions = _load_prompt_directions(config)
    answer_subspaces = _fit_answer_memory_subspaces(
        config, geometry_settings, settings, force=force
    )
    layers = [int(layer) for layer in settings["layers"]]
    results: dict[str, Any] = {
        "scope": {
            "official_haloscope": False,
            "comparator": "HaloScope-style weighted PCA direct projection",
            "shared_model": config["model"]["name_or_path"],
            "shared_generations": True,
            "shared_labels": True,
            "selection": "layer/rank/sign selected on validation; test untouched",
            "prompt_track": "prompt-only for both methods",
            "answer_track": "same re-tokenized saved answer for both methods",
        },
        "benchmarks": {},
        **_metadata(config, geometry_settings, settings),
    }

    for benchmark_index, benchmark in enumerate(config["benchmarks"]["datasets"]):
        checkpoint = target / "benchmarks" / f"{benchmark}.json"
        if checkpoint.exists() and not force:
            with checkpoint.open("r", encoding="utf-8") as handle:
                results["benchmarks"][benchmark] = json.load(handle)
            continue
        prompt_records, prompt_features = _load_prompt_features(config, layers, benchmark)
        answer_records, answer_features = _load_answer_features(config, layers, benchmark)
        prompt_ids = [record["fact_id"] for record in prompt_records]
        answer_ids = [record["fact_id"] for record in answer_records]
        if prompt_ids != answer_ids:
            raise RuntimeError(f"Prompt/answer feature identities differ for {benchmark}")
        scored = np.asarray([record["correct"] >= 0 for record in answer_records])
        labels = np.asarray(
            [max(0, record["correct"]) for record in answer_records], dtype=int
        )[scored]
        fact_ids = np.asarray(answer_ids)[scored].tolist()
        prompt_scored = {
            layer: features[torch.from_numpy(scored)]
            for layer, features in prompt_features.items()
        }
        answer_scored = {
            layer: features[torch.from_numpy(scored)]
            for layer, features in answer_features.items()
        }
        split = _split_indices(labels, settings)

        prompt_halo_selection, prompt_halo_score, _ = _select_haloscope_style(
            prompt_scored, labels, split, settings
        )
        answer_halo_selection, answer_halo_score, _ = _select_haloscope_style(
            answer_scored, labels, split, settings
        )
        prompt_memory_selection, prompt_memory_score = _select_controlled_direction(
            prompt_scored,
            {layer: prompt_directions[layer] for layer in layers},
            labels,
            split,
        )
        answer_memory_selection, answer_memory_score = _select_controlled_direction(
            answer_scored,
            {
                layer: answer_subspaces[layer]["mean_direction"].float()
                for layer in layers
            },
            labels,
            split,
        )
        all_scores = {
            "memory_prompt": prompt_memory_score,
            "haloscope_style_prompt_direct": prompt_halo_score,
            "memory_answer_conditioned": answer_memory_score,
            "haloscope_style_answer_direct": answer_halo_score,
            **_baseline_scores(config, benchmark, fact_ids),
        }
        test = split["test"]
        test_labels = labels[test]
        test_metrics = {
            name: _ranking_metrics(test_labels, score[test])
            for name, score in all_scores.items()
        }
        comparisons = {}
        comparison_pairs = (
            ("memory_prompt", "haloscope_style_prompt_direct"),
            ("memory_answer_conditioned", "haloscope_style_answer_direct"),
            ("memory_prompt", "token_entropy"),
            ("memory_answer_conditioned", "token_entropy"),
        )
        for pair_index, (proposed, baseline) in enumerate(comparison_pairs):
            comparisons[f"{proposed}_minus_{baseline}"] = _paired_ranking_differences(
                test_labels,
                all_scores[proposed][test],
                all_scores[baseline][test],
                int(settings["bootstrap_samples"]),
                int(settings["seed"]) + benchmark_index * 100 + pair_index,
            )
        benchmark_result = {
            "num_scored": int(len(labels)),
            "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
            "test_fact_ids": [fact_ids[index] for index in test],
            "selection": {
                "memory_prompt": prompt_memory_selection,
                "haloscope_style_prompt_direct": prompt_halo_selection,
                "memory_answer_conditioned": answer_memory_selection,
                "haloscope_style_answer_direct": answer_halo_selection,
            },
            "test_metrics": test_metrics,
            "paired_bootstrap": comparisons,
        }
        results["benchmarks"][benchmark] = benchmark_result
        atomic_json(checkpoint, benchmark_result)

    atomic_json(metrics_path, results)
    summary = _write_summary(results, target / "matched_summary.md")
    mark_complete(
        done,
        {
            "metrics": str(metrics_path),
            "summary": str(summary),
            **_metadata(config, geometry_settings, settings),
        },
    )
    return metrics_path


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.4f}" if np.isfinite(number) else "NA"


def _write_summary(results: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Matched controlled-memory versus HaloScope-style comparison",
        "",
        "This is not labeled as an official HaloScope reproduction. It is a matched implementation "
        "of weighted PCA direct projection, evaluated with identical generations, labels, layers, "
        "validation selection, and held-out test examples.",
        "",
    ]
    order = (
        "memory_prompt",
        "haloscope_style_prompt_direct",
        "memory_answer_conditioned",
        "haloscope_style_answer_direct",
        "token_logprob",
        "token_entropy",
        "probe_ensemble/base_replay/genuine_exposure",
        "probe_ensemble/on_policy/genuine_exposure",
    )
    for benchmark, result in results["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                f"Splits: wild={result['split_sizes']['wild']}, "
                f"validation={result['split_sizes']['validation']}, "
                f"test={result['split_sizes']['test']}.",
                "",
                "| Method | Selected layer | Rank | Test AUROC | Test AUPRC | Test AURC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in order:
            if name not in result["test_metrics"]:
                continue
            selection = result["selection"].get(name, {})
            metrics = result["test_metrics"][name]
            lines.append(
                f"| {name} | {selection.get('layer', 'NA')} | "
                f"{selection.get('k', 'NA')} | {_number(metrics['auroc'])} | "
                f"{_number(metrics['auprc'])} | {_number(metrics['aurc'])} |"
            )
        lines.extend(
            [
                "",
                "### Paired test-set comparisons",
                "",
                "Positive AUROC/AUPRC differences and negative AURC differences favor the "
                "controlled-memory method.",
                "",
                "| Comparison | Metric | Difference | 95% CI |",
                "|---|---|---:|---:|",
            ]
        )
        for name, comparison in result["paired_bootstrap"].items():
            for metric, values in comparison.items():
                lower, upper = values["ci95"]
                lines.append(
                    f"| {name} | {metric} | {_number(values['observed_difference'])} | "
                    f"[{_number(lower)}, {_number(upper)}] |"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
