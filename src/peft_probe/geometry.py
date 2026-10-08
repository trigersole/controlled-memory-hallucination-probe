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
from sklearn.metrics import average_precision_score, roc_auc_score

from .collect import load_feature_shards
from .config import config_fingerprint, output_dir
from .io_utils import (
    atomic_json,
    atomic_torch_save,
    chunks,
    ensure_manifest,
    mark_complete,
    read_jsonl,
)
from .metrics import risk_coverage
from .modeling import load_base_model, load_tokenizer, model_device, user_prompt
from .versioning import artifact_metadata


GEOMETRY_SCHEMA_VERSION = 1


def load_geometry_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Geometry configuration must be a mapping")
    if int(settings.get("geometry_schema_version", -1)) != GEOMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"Expected geometry schema {GEOMETRY_SCHEMA_VERSION}, "
            f"found {settings.get('geometry_schema_version')}"
        )
    layers = settings.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError("geometry.layers must be a non-empty list of integer indices")
    if len(set(layers)) != len(layers):
        raise ValueError("geometry.layers must not contain duplicates")
    if settings.get("pooling", "last") not in {"last", "mean"}:
        raise ValueError("geometry.pooling must be 'last' or 'mean'")
    for key in (
        "batch_size",
        "shard_size",
        "max_components",
        "control_repeats",
        "subspace_control_repeats",
        "bootstrap_samples",
    ):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"geometry.{key} must be a positive integer")
    threshold = float(settings.get("variance_threshold", 0.0))
    if not 0 < threshold <= 1:
        raise ValueError("geometry.variance_threshold must be in (0, 1]")
    int(settings["seed"])
    return settings


def geometry_fingerprint(config: dict[str, Any], settings: dict[str, Any]) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "geometry": settings,
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def geometry_root(config: dict[str, Any]) -> Path:
    return output_dir(config) / "geometry" / f"multilayer_v{GEOMETRY_SCHEMA_VERSION}"


def _geometry_metadata(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "geometry_fingerprint": geometry_fingerprint(config, settings),
        **artifact_metadata(),
    }


def _pool_prompt(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    mask = attention_mask.bool()
    if pooling == "mean":
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("Prompt batch contains an empty attention mask")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


@torch.inference_mode()
def _extract_prompt_features(
    model,
    tokenizer,
    questions: list[str],
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    prompts = [user_prompt(tokenizer, question) for question in questions]
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    device = model_device(model)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    outputs = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states
    result: dict[str, torch.Tensor] = {}
    resolved: set[int] = set()
    for requested in settings["layers"]:
        index = requested if requested >= 0 else len(hidden_states) + requested
        if not 0 <= index < len(hidden_states):
            raise ValueError(
                f"Layer {requested} is invalid for a model with {len(hidden_states)} hidden states"
            )
        if index in resolved:
            raise ValueError(f"Geometry layers resolve to duplicate hidden state {index}")
        resolved.add(index)
        pooled = _pool_prompt(hidden_states[index], tokens["attention_mask"], settings["pooling"])
        result[str(requested)] = pooled.detach().to(dtype=torch.float16, device="cpu")
    return result


def _collect_rows(
    config: dict[str, Any],
    settings: dict[str, Any],
    target: Path,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    kind: str,
    force: bool,
) -> Path:
    fingerprint = geometry_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target
    shard_size = int(settings["shard_size"])
    batch_size = int(settings["batch_size"])
    expected_shards = math.ceil(len(rows) / shard_size)
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        parts = {str(layer): [] for layer in settings["layers"]}
        records: list[dict[str, Any]] = []
        for _, batch_rows in chunks(shard_rows, batch_size):
            features = _extract_prompt_features(
                model,
                tokenizer,
                [row["question"] for row in batch_rows],
                settings,
            )
            records.extend(batch_rows)
            for layer, tensor in features.items():
                parts[layer].append(tensor)
        atomic_torch_save(
            shard_path,
            {
                "records": records,
                "features": {layer: torch.cat(values) for layer, values in parts.items()},
                "kind": kind,
                "shard_index": shard_index,
                **_geometry_metadata(config, settings),
            },
        )
    actual_shards = len(list(target.glob("shard_*.pt")))
    if actual_shards != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} geometry shards, found {actual_shards}")
    mark_complete(
        done,
        {
            "kind": kind,
            "num_examples": len(rows),
            "num_shards": actual_shards,
            **_geometry_metadata(config, settings),
        },
    )
    return target


def collect_synthetic_geometry(
    config: dict[str, Any], settings: dict[str, Any], source: str, force: bool = False
) -> Path:
    if source not in {"adapter_a", "adapter_b"}:
        raise ValueError("Synthetic geometry source must be adapter_a or adapter_b")
    root = output_dir(config)
    facts = sorted(read_jsonl(root / "data" / "facts.jsonl"), key=lambda row: row["fact_id"])
    rows = [
        {
            "fact_id": row["fact_id"],
            "entity_id": row["entity_id"],
            "relation": row["relation"],
            "split": row["detector_split"],
            "adapter_assignment": row["adapter_assignment"],
            "question": row["eval_question"],
        }
        for row in facts
        if row["adapter_assignment"] in {"a", "b"}
    ]
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    adapter_path = root / "adapters" / source / "final"
    if not (adapter_path / "_SUCCESS.json").exists():
        raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return _collect_rows(
        config,
        settings,
        geometry_root(config) / "features" / "synthetic" / source,
        rows,
        model,
        tokenizer,
        source,
        force,
    )


def collect_benchmark_geometry(
    config: dict[str, Any], settings: dict[str, Any], benchmark: str, force: bool = False
) -> Path:
    root = output_dir(config)
    old_records, _ = load_feature_shards(
        root / "features" / "benchmarks" / benchmark,
        config["collection"]["feature_modes"][0],
    )
    rows = []
    for record in old_records:
        rows.append(
            {
                "fact_id": record["fact_id"],
                "entity_id": record["entity_id"],
                "question": record["question"],
                "answer": record["answer"],
                "correct": record["correct"],
                "grading_status": record.get("grading_status"),
                "benchmark": benchmark,
            }
        )
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    model.eval()
    return _collect_rows(
        config,
        settings,
        geometry_root(config) / "features" / "benchmarks" / benchmark,
        rows,
        model,
        tokenizer,
        benchmark,
        force,
    )


def _load_geometry_features(
    directory: Path, layer: int
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    for path in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        features.append(payload["features"][str(layer)].float())
    if not features:
        raise FileNotFoundError(f"No geometry shards found in {directory}")
    return records, torch.cat(features)


def _fit_subspace(
    matrix: torch.Tensor, max_components: int, threshold: float, seed: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    matrix = matrix.float()
    q = min(int(max_components), matrix.shape[0], matrix.shape[1])
    if q < 1:
        raise ValueError("Cannot fit an empty memory subspace")
    total_energy = float(matrix.square().sum())
    if total_energy <= 1e-12:
        raise ValueError(
            "Paired adapter deltas have zero energy at a requested layer; "
            "remove adapter-invariant layers such as the embedding state"
        )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _, singular_values, vectors = torch.pca_lowrank(matrix, q=q, center=False, niter=4)
    captured = singular_values.square()
    cumulative = captured.cumsum(0) / max(total_energy, 1e-12)
    reached = torch.nonzero(cumulative >= threshold, as_tuple=False)
    rank = int(reached[0, 0] + 1) if len(reached) else q
    basis = vectors[:, :rank].contiguous()
    return basis, {
        "rank": rank,
        "max_components": q,
        "captured_energy": float(cumulative[rank - 1]),
        "singular_values": [float(value) for value in singular_values[:rank]],
    }


def _subspace_alignment(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    if first.numel() == 0 or second.numel() == 0:
        return {"mean_cosine_squared": float("nan"), "principal_angles_degrees": []}
    singular = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    angles = torch.rad2deg(torch.arccos(singular))
    return {
        "mean_cosine_squared": float(singular.square().mean()),
        "principal_angles_degrees": [float(value) for value in angles],
    }


def _direction_overlap(direction: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    norm = direction.norm()
    if float(norm) <= 1e-12:
        return {"projection_fraction": float("nan"), "angle_degrees": float("nan")}
    unit = direction / norm
    fraction = float((basis.T @ unit).square().sum().clamp(0, 1))
    return {
        "projection_fraction": fraction,
        "angle_degrees": float(np.degrees(np.arccos(np.sqrt(fraction)))),
    }


def _ranking_metrics(labels_correct: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    labels_error = 1 - np.asarray(labels_correct, dtype=int)
    risk = np.asarray(risk, dtype=float)
    result = {
        "n": int(len(labels_error)),
        "error_rate": float(labels_error.mean()),
        "aurc": float(risk_coverage(labels_correct, risk)["aurc"]),
    }
    if np.unique(labels_error).size == 2:
        result["auroc"] = float(roc_auc_score(labels_error, risk))
        result["auprc"] = float(average_precision_score(labels_error, risk))
    else:
        result.update({"auroc": float("nan"), "auprc": float("nan")})
    return result


def _bootstrap_ranking(
    labels: np.ndarray, risk: np.ndarray, samples: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": [], "aurc": []}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        metrics = _ranking_metrics(labels[indices], risk[indices])
        for name in values:
            if np.isfinite(metrics[name]):
                values[name].append(metrics[name])
    return {
        name: (
            [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
            if draws
            else [float("nan"), float("nan")]
        )
        for name, draws in values.items()
    }


def _nearest_centroid_accuracy(
    train_exposed: torch.Tensor,
    train_withheld: torch.Tensor,
    test_exposed: torch.Tensor,
    test_withheld: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    exposed_center = (train_exposed @ basis).mean(0)
    withheld_center = (train_withheld @ basis).mean(0)
    test = torch.cat([test_exposed @ basis, test_withheld @ basis])
    labels = torch.cat([torch.ones(len(test_exposed)), torch.zeros(len(test_withheld))])
    exposed_distance = (test - exposed_center).square().sum(1)
    withheld_distance = (test - withheld_center).square().sum(1)
    predictions = (exposed_distance < withheld_distance).float()
    return float((predictions == labels).float().mean())


def _knn_accuracy(
    train_exposed: torch.Tensor,
    train_withheld: torch.Tensor,
    test_exposed: torch.Tensor,
    test_withheld: torch.Tensor,
    basis: torch.Tensor,
    neighbors: int = 5,
) -> float:
    train = torch.cat([train_exposed @ basis, train_withheld @ basis])
    train_labels = torch.cat(
        [torch.ones(len(train_exposed)), torch.zeros(len(train_withheld))]
    )
    test = torch.cat([test_exposed @ basis, test_withheld @ basis])
    test_labels = torch.cat([torch.ones(len(test_exposed)), torch.zeros(len(test_withheld))])
    nearest = torch.cdist(test, train).topk(min(neighbors, len(train)), largest=False).indices
    predictions = train_labels[nearest].mean(1).ge(0.5).float()
    return float((predictions == test_labels).float().mean())


def _mahalanobis_accuracy(
    train_exposed: torch.Tensor,
    train_withheld: torch.Tensor,
    test_exposed: torch.Tensor,
    test_withheld: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    train_exposed = train_exposed @ basis
    train_withheld = train_withheld @ basis
    exposed_center = train_exposed.mean(0)
    withheld_center = train_withheld.mean(0)
    centered = torch.cat(
        [train_exposed - exposed_center, train_withheld - withheld_center]
    )
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    scale = torch.trace(covariance) / max(covariance.shape[0], 1)
    precision = torch.linalg.pinv(
        covariance + torch.eye(covariance.shape[0]) * scale.clamp_min(1e-6) * 1e-3
    )
    test = torch.cat([test_exposed @ basis, test_withheld @ basis])
    labels = torch.cat([torch.ones(len(test_exposed)), torch.zeros(len(test_withheld))])

    def squared_distance(values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        difference = values - center
        return torch.einsum("bi,ij,bj->b", difference, precision, difference)

    predictions = (
        squared_distance(test, exposed_center) < squared_distance(test, withheld_center)
    ).float()
    return float((predictions == labels).float().mean())


def _linear_cka_with_labels(features: torch.Tensor, labels: torch.Tensor) -> float:
    x = features - features.mean(0, keepdim=True)
    y = labels.float().reshape(-1, 1)
    y = y - y.mean(0, keepdim=True)
    numerator = (x.T @ y).square().sum()
    denominator = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return float(numerator / denominator.clamp_min(1e-12))


def _rbf_mmd(first: torch.Tensor, second: torch.Tensor, maximum: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    if len(first) > maximum:
        first = first[torch.randperm(len(first), generator=generator)[:maximum]]
    if len(second) > maximum:
        second = second[torch.randperm(len(second), generator=generator)[:maximum]]
    combined = torch.cat([first, second])
    distances = torch.pdist(combined).square()
    positive = distances[distances > 0]
    bandwidth = positive.median() if len(positive) else torch.tensor(1.0)
    gamma = 1.0 / bandwidth.clamp_min(1e-12)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(left, right).square())

    k_xx = kernel(first, first)
    k_yy = kernel(second, second)
    k_xy = kernel(first, second)
    return float((k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()).clamp_min(0))


def _cluster_bootstrap_paired(
    values: np.ndarray, entities: np.ndarray, samples: int, seed: int
) -> list[float]:
    grouped = {entity: values[entities == entity] for entity in np.unique(entities)}
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.integers(0, len(keys), len(keys))
        draw = np.concatenate([grouped[keys[index]] for index in selected])
        draws.append(float(draw.mean()))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _paired_arrays(
    records_a: list[dict[str, Any]],
    features_a: torch.Tensor,
    records_b: list[dict[str, Any]],
    features_b: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    index_b = {record["fact_id"]: index for index, record in enumerate(records_b)}
    metadata = []
    exposed = []
    withheld = []
    for index_a, record in enumerate(records_a):
        if record["fact_id"] not in index_b:
            raise RuntimeError(f"Missing complementary representation for {record['fact_id']}")
        feature_b = features_b[index_b[record["fact_id"]]]
        if record["adapter_assignment"] == "a":
            exposed.append(features_a[index_a])
            withheld.append(feature_b)
        elif record["adapter_assignment"] == "b":
            exposed.append(feature_b)
            withheld.append(features_a[index_a])
        else:
            continue
        metadata.append(record)
    return metadata, torch.stack(exposed), torch.stack(withheld)


def _permuted_within_groups(
    indices: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    permuted = indices.copy()
    for group in sorted(set(groups)):
        positions = np.flatnonzero(groups == group)
        permuted[positions] = rng.permutation(indices[positions])
    return permuted


def _shuffle_labels_within_groups(
    labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    shuffled = labels.copy()
    for group in sorted(set(groups)):
        positions = np.flatnonzero(groups == group)
        shuffled[positions] = rng.permutation(labels[positions])
    return shuffled


def analyze_geometry(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    root = geometry_root(config)
    target = root / "results"
    fingerprint = geometry_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    result_path = target / "geometry_metrics.json"
    if done.exists() and result_path.exists() and not force:
        return result_path

    seed = int(settings["seed"])
    results: dict[str, Any] = {
        "method": {
            "representation": "prompt_only",
            "pairing": "same fact, exposed adapter minus withheld adapter",
            "fit_split": "train",
            "controls": ["shuffled_exposure_labels", "random_directions", "shuffled_pairs"],
        },
        "layers": {},
        **_geometry_metadata(config, settings),
    }
    bases: dict[str, Any] = {}
    synthetic_root = root / "features" / "synthetic"
    for layer_index, layer in enumerate(settings["layers"]):
        layer_rng = np.random.default_rng(seed + layer_index * 100000)
        layer_name = str(layer).replace("-", "minus_")
        layer_result_path = target / "layers" / f"layer_{layer_name}.json"
        layer_basis_path = target / "layers" / f"layer_{layer_name}.pt"
        if layer_result_path.exists() and layer_basis_path.exists() and not force:
            with layer_result_path.open("r", encoding="utf-8") as handle:
                results["layers"][str(layer)] = json.load(handle)
            layer_payload = torch.load(layer_basis_path, map_location="cpu", weights_only=False)
            if layer_payload.get("geometry_fingerprint") != fingerprint:
                raise RuntimeError(f"Stale geometry layer checkpoint: {layer_basis_path}")
            bases[str(layer)] = {
                "basis": layer_payload["basis"],
                "mean_direction": layer_payload["mean_direction"],
            }
            continue
        records_a, features_a = _load_geometry_features(synthetic_root / "adapter_a", layer)
        records_b, features_b = _load_geometry_features(synthetic_root / "adapter_b", layer)
        records, exposed, withheld = _paired_arrays(records_a, features_a, records_b, features_b)
        splits = np.asarray([record["split"] for record in records])
        relations = np.asarray([record["relation"] for record in records])
        entities = np.asarray([record["entity_id"] for record in records])
        train = splits == "train"
        test = splits == "test"
        validation = splits == "validation"
        delta = exposed - withheld
        basis, spectrum = _fit_subspace(
            delta[train],
            int(settings["max_components"]),
            float(settings["variance_threshold"]),
            seed + layer_index,
        )
        mean_direction = delta[train].mean(0)
        mean_direction = mean_direction / mean_direction.norm().clamp_min(1e-12)
        bases[str(layer)] = {
            "basis": basis.half(),
            "mean_direction": mean_direction.half(),
        }

        split_results: dict[str, Any] = {}
        for split_name, mask in (("validation", validation), ("test", test)):
            if not mask.any():
                continue
            heldout_basis, _ = _fit_subspace(
                delta[mask],
                basis.shape[1],
                1.0,
                seed + layer_index + (100 if split_name == "test" else 50),
            )
            signed_delta = ((exposed[mask] - withheld[mask]) @ mean_direction).numpy()
            projected_exposed = exposed[mask] @ basis
            projected_withheld = withheld[mask] @ basis
            labels = torch.cat(
                [torch.ones(len(projected_exposed)), torch.zeros(len(projected_withheld))]
            )
            split_results[split_name] = {
                "paired_direction_accuracy": float(np.mean(signed_delta > 0)),
                "mean_signed_pair_delta": float(np.mean(signed_delta)),
                "mean_signed_pair_delta_ci95": _cluster_bootstrap_paired(
                    signed_delta,
                    entities[mask],
                    int(settings["bootstrap_samples"]),
                    seed + layer_index,
                ),
                "nearest_centroid_accuracy": _nearest_centroid_accuracy(
                    exposed[train], withheld[train], exposed[mask], withheld[mask], basis
                ),
                "knn_accuracy": _knn_accuracy(
                    exposed[train], withheld[train], exposed[mask], withheld[mask], basis
                ),
                "mahalanobis_accuracy": _mahalanobis_accuracy(
                    exposed[train], withheld[train], exposed[mask], withheld[mask], basis
                ),
                "linear_cka_exposure": _linear_cka_with_labels(
                    torch.cat([projected_exposed, projected_withheld]), labels
                ),
                "rbf_mmd": _rbf_mmd(
                    projected_exposed,
                    projected_withheld,
                    maximum=1000,
                    seed=seed + layer_index,
                ),
                "subspace_alignment_with_train": _subspace_alignment(basis, heldout_basis),
            }

        relation_alignment = {}
        for relation in sorted(set(relations[train])):
            mask = train & (relations == relation)
            relation_basis, _ = _fit_subspace(
                delta[mask], basis.shape[1], 1.0, seed + layer_index + len(relation)
            )
            relation_alignment[relation] = _subspace_alignment(basis, relation_basis)

        # Flipping delta signs leaves an uncentered SVD subspace unchanged. Breaking the
        # same-fact pairing is therefore the appropriate high-dimensional null control.
        pair_control = []
        pair_control_bases = []
        train_indices = np.flatnonzero(train)
        assignments = np.asarray([record["adapter_assignment"] for record in records])
        pairing_groups = np.asarray(
            [f"{relation}/{assignment}" for relation, assignment in zip(relations, assignments)]
        )[train]
        for repeat in range(int(settings["subspace_control_repeats"])):
            permuted = _permuted_within_groups(train_indices, pairing_groups, layer_rng)
            shuffled_delta = exposed[train_indices] - withheld[permuted]
            shuffled_basis, _ = _fit_subspace(
                shuffled_delta,
                basis.shape[1],
                1.0,
                seed + 10000 + layer_index * 100 + repeat,
            )
            pair_control_bases.append(shuffled_basis)
            pair_control.append(_subspace_alignment(basis, shuffled_basis)["mean_cosine_squared"])

        benchmark_results = {}
        for benchmark_index, benchmark in enumerate(config["benchmarks"]["datasets"]):
            benchmark_records, benchmark_features = _load_geometry_features(
                root / "features" / "benchmarks" / benchmark, layer
            )
            scored = np.asarray([record["correct"] >= 0 for record in benchmark_records])
            labels_correct = np.asarray(
                [max(0, record["correct"]) for record in benchmark_records], dtype=int
            )[scored]
            features = benchmark_features[torch.from_numpy(scored)]
            risk = (-(features @ mean_direction)).numpy()
            ranking = _ranking_metrics(labels_correct, risk)
            ranking["ci95"] = _bootstrap_ranking(
                labels_correct,
                risk,
                int(settings["bootstrap_samples"]),
                seed + layer_index * 10 + benchmark_index,
            )
            centered_features = features - features.mean(0, keepdim=True)
            energy_risk = -(centered_features @ basis).square().sum(1).numpy()
            energy_ranking = _ranking_metrics(labels_correct, energy_risk)
            energy_ranking["ci95"] = _bootstrap_ranking(
                labels_correct,
                energy_risk,
                int(settings["bootstrap_samples"]),
                seed + 5000 + layer_index * 10 + benchmark_index,
            )
            correct = torch.from_numpy(labels_correct == 1)
            error = ~correct
            correctness_direction = features[correct].mean(0) - features[error].mean(0)

            combined_train = torch.cat([exposed[train], withheld[train]])
            exposure_labels = np.concatenate(
                [np.ones(int(train.sum()), dtype=int), np.zeros(int(train.sum()), dtype=int)]
            )
            train_relations = relations[train]
            train_assignments = assignments[train]
            exposed_sources = train_assignments
            withheld_sources = np.where(train_assignments == "a", "b", "a")
            shuffle_groups = np.concatenate(
                [
                    np.char.add(np.char.add(train_relations, "/"), exposed_sources),
                    np.char.add(np.char.add(train_relations, "/"), withheld_sources),
                ]
            )
            shuffled_aurocs = []
            random_aurocs = []
            for _ in range(int(settings["control_repeats"])):
                shuffled = _shuffle_labels_within_groups(
                    exposure_labels, shuffle_groups, layer_rng
                )
                direction = (
                    combined_train[torch.from_numpy(shuffled == 1)].mean(0)
                    - combined_train[torch.from_numpy(shuffled == 0)].mean(0)
                )
                direction /= direction.norm().clamp_min(1e-12)
                shuffled_aurocs.append(
                    _ranking_metrics(labels_correct, (-(features @ direction)).numpy())["auroc"]
                )
                random_direction = torch.from_numpy(
                    layer_rng.standard_normal(features.shape[1]).astype(np.float32)
                )
                random_direction /= random_direction.norm().clamp_min(1e-12)
                random_aurocs.append(
                    _ranking_metrics(
                        labels_correct, (-(features @ random_direction)).numpy()
                    )["auroc"]
                )
            shuffled_pair_energy_aurocs = [
                _ranking_metrics(
                    labels_correct,
                    (-(centered_features @ control_basis).square().sum(1)).numpy(),
                )["auroc"]
                for control_basis in pair_control_bases
            ]
            random_energy_aurocs = []
            for _ in range(int(settings["subspace_control_repeats"])):
                random_matrix = torch.from_numpy(
                    layer_rng.standard_normal(
                        (features.shape[1], basis.shape[1])
                    ).astype(np.float32)
                )
                random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
                random_energy_aurocs.append(
                    _ranking_metrics(
                        labels_correct,
                        (-(centered_features @ random_basis).square().sum(1)).numpy(),
                    )["auroc"]
                )
            benchmark_results[benchmark] = {
                "memory_direction_ranking": ranking,
                "memory_subspace_energy_ranking": energy_ranking,
                "correctness_direction_alignment": _direction_overlap(
                    correctness_direction, basis
                ),
                "controls": {
                    "shuffled_exposure_auroc_mean": float(np.nanmean(shuffled_aurocs)),
                    "shuffled_exposure_auroc_std": float(np.nanstd(shuffled_aurocs, ddof=1)),
                    "random_direction_auroc_mean": float(np.nanmean(random_aurocs)),
                    "random_direction_auroc_std": float(np.nanstd(random_aurocs, ddof=1)),
                    "fraction_shuffled_at_least_genuine": float(
                        np.mean(np.asarray(shuffled_aurocs) >= ranking["auroc"])
                    ),
                    "fraction_random_at_least_genuine": float(
                        np.mean(np.asarray(random_aurocs) >= ranking["auroc"])
                    ),
                    "shuffled_pair_energy_auroc_mean": float(
                        np.nanmean(shuffled_pair_energy_aurocs)
                    ),
                    "random_subspace_energy_auroc_mean": float(
                        np.nanmean(random_energy_aurocs)
                    ),
                },
            }

        layer_result = {
            "spectrum": spectrum,
            "mean_delta_norm": float(delta[train].mean(0).norm()),
            "splits": split_results,
            "relation_alignment": relation_alignment,
            "shuffled_pair_subspace_alignment": {
                "mean": float(np.mean(pair_control)),
                "std": float(np.std(pair_control, ddof=1)) if len(pair_control) > 1 else 0.0,
                "repeats": len(pair_control),
            },
            "benchmarks": benchmark_results,
        }
        results["layers"][str(layer)] = layer_result
        atomic_torch_save(
            layer_basis_path,
            {
                "basis": basis.half(),
                "mean_direction": mean_direction.half(),
                **_geometry_metadata(config, settings),
            },
        )
        atomic_json(layer_result_path, layer_result)

    atomic_torch_save(
        target / "memory_subspaces.pt",
        {"layers": bases, **_geometry_metadata(config, settings)},
    )
    atomic_json(result_path, results)
    summary_path = _write_geometry_summary(results, target / "geometry_summary.md")
    mark_complete(
        done,
        {
            "metrics": str(result_path),
            "summary": str(summary_path),
            **_geometry_metadata(config, settings),
        },
    )
    return result_path


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.4f}" if np.isfinite(number) else "NA"


def _write_geometry_summary(results: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Controlled-memory hidden-space geometry",
        "",
        f"Geometry schema: {results['geometry_schema_version']}; "
        f"pipeline schema: {results['pipeline_schema_version']}; "
        f"code revision: `{results['code_revision']}`.",
        "",
        "Memory subspaces are fitted only on prompt-only paired deltas from the synthetic "
        "training split. Benchmark labels are used only for post-hoc alignment and ranking.",
        "",
        "| Layer | Rank | Captured energy | Pair accuracy | Centroid | kNN | Mahalanobis | "
        "Train/test overlap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, result in results["layers"].items():
        test = result["splits"].get("test", {})
        alignment = test.get("subspace_alignment_with_train", {})
        lines.append(
            f"| {layer} | {result['spectrum']['rank']} | "
            f"{_fmt(result['spectrum']['captured_energy'])} | "
            f"{_fmt(test.get('paired_direction_accuracy'))} | "
            f"{_fmt(test.get('nearest_centroid_accuracy'))} | "
            f"{_fmt(test.get('knn_accuracy'))} | "
            f"{_fmt(test.get('mahalanobis_accuracy'))} | "
            f"{_fmt(alignment.get('mean_cosine_squared'))} |"
        )
    first_layer = next(iter(results["layers"].values()))
    for benchmark in first_layer["benchmarks"]:
        lines.extend(
            [
                "",
                f"## {benchmark}",
                "",
                "| Layer | Direction AUROC | 95% CI | Energy AUROC | AUPRC | AURC | "
                "Correctness overlap | Shuffled AUROC | Random AUROC |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for layer, result in results["layers"].items():
            item = result["benchmarks"][benchmark]
            ranking = item["memory_direction_ranking"]
            energy = item["memory_subspace_energy_ranking"]
            lower, upper = ranking["ci95"]["auroc"]
            controls = item["controls"]
            lines.append(
                f"| {layer} | {_fmt(ranking['auroc'])} | [{_fmt(lower)}, {_fmt(upper)}] | "
                f"{_fmt(energy['auroc'])} | {_fmt(ranking['auprc'])} | "
                f"{_fmt(ranking['aurc'])} | "
                f"{_fmt(item['correctness_direction_alignment']['projection_fraction'])} | "
                f"{_fmt(controls['shuffled_exposure_auroc_mean'])} | "
                f"{_fmt(controls['random_direction_auroc_mean'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
