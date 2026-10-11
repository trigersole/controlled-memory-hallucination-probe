from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from .config import config_fingerprint, output_dir, source_output_dir, validate_source_run
from .geometry import _bootstrap_ranking, _fit_subspace, _ranking_metrics
from .io_utils import atomic_json, ensure_manifest, mark_complete, read_jsonl
from .versioning import artifact_metadata


MEMORY_SUBSPACE_SCHEMA_VERSION = 1


def load_memory_subspace_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Memory-subspace configuration must be a mapping")
    if int(settings.get("memory_subspace_schema_version", -1)) != MEMORY_SUBSPACE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected memory-subspace schema {MEMORY_SUBSPACE_SCHEMA_VERSION}, "
            f"found {settings.get('memory_subspace_schema_version')}"
        )
    layers = settings.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError("memory_subspace.layers must be a non-empty list of integers")
    if len(set(layers)) != len(layers):
        raise ValueError("memory_subspace.layers must not contain duplicates")
    representations = settings.get("representations")
    allowed = {"prompt_last", "answer_last", "trajectory"}
    if not isinstance(representations, list) or not representations:
        raise ValueError("memory_subspace.representations must be a non-empty list")
    unknown = set(representations) - allowed
    if unknown:
        raise ValueError(f"Unknown memory-subspace representations: {sorted(unknown)}")
    ranks = [int(value) for value in settings.get("rank_candidates", [])]
    if not ranks or any(value <= 0 for value in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("memory_subspace.rank_candidates must be distinct positive integers")
    for key in ("control_repeats", "subspace_control_repeats", "bootstrap_samples"):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"memory_subspace.{key} must be positive")
    if settings.get("benchmark") != "trivia_qa":
        raise ValueError("The confirmatory memory benchmark is restricted to trivia_qa")
    int(settings["seed"])
    return settings


def memory_subspace_fingerprint(config: dict[str, Any], settings: dict[str, Any]) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "source_output_dir": str(source_output_dir(config)),
        "memory_subspace": settings,
        "memory_subspace_schema_version": MEMORY_SUBSPACE_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def memory_subspace_root(config: dict[str, Any]) -> Path:
    return (
        output_dir(config)
        / "memory_subspace"
        / f"answer_trajectory_v{MEMORY_SUBSPACE_SCHEMA_VERSION}"
    )


def _metadata(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_subspace_schema_version": MEMORY_SUBSPACE_SCHEMA_VERSION,
        "memory_subspace_fingerprint": memory_subspace_fingerprint(config, settings),
        **artifact_metadata(),
    }


def _load_layers(
    directory: Path, layers: list[int]
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    records: list[dict[str, Any]] = []
    parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for path in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        for layer in layers:
            parts[layer].append(payload["features"][str(layer)].float())
    if not records:
        raise FileNotFoundError(
            f"No feature shards found in {directory}. Complete the earlier multi-layer "
            "geometry and same-answer replay stages first."
        )
    return records, {layer: torch.cat(values) for layer, values in parts.items()}


def _align_features(
    reference: list[dict[str, Any]],
    other: list[dict[str, Any]],
    features: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    index = {record["fact_id"]: position for position, record in enumerate(other)}
    if len(index) != len(other):
        raise RuntimeError("Feature records contain duplicate fact IDs")
    missing = [record["fact_id"] for record in reference if record["fact_id"] not in index]
    if missing:
        raise RuntimeError(f"Missing aligned features for {missing[0]}")
    order = torch.tensor([index[record["fact_id"]] for record in reference], dtype=torch.long)
    return {layer: values[order] for layer, values in features.items()}


def _representations(
    prompt: dict[int, torch.Tensor], answer: dict[int, torch.Tensor]
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        "prompt_last": prompt,
        "answer_last": answer,
        "trajectory": {layer: answer[layer] - prompt[layer] for layer in prompt},
    }


def _paired(
    records_a: list[dict[str, Any]],
    values_a: torch.Tensor,
    records_b: list[dict[str, Any]],
    values_b: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    index_b = {record["fact_id"]: index for index, record in enumerate(records_b)}
    metadata = []
    exposed = []
    withheld = []
    for index_a, record in enumerate(records_a):
        other_index = index_b.get(record["fact_id"])
        if other_index is None:
            raise RuntimeError(f"Missing complementary adapter feature for {record['fact_id']}")
        if record["adapter_assignment"] == "a":
            exposed.append(values_a[index_a])
            withheld.append(values_b[other_index])
        elif record["adapter_assignment"] == "b":
            exposed.append(values_b[other_index])
            withheld.append(values_a[index_a])
        else:
            continue
        metadata.append(record)
    return metadata, torch.stack(exposed), torch.stack(withheld)


def _unit_center(values: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    centered = values.float() - center.float()
    return centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _exposure_auroc(exposed_score: torch.Tensor, withheld_score: torch.Tensor) -> float:
    labels = np.concatenate(
        [np.ones(len(exposed_score), dtype=int), np.zeros(len(withheld_score), dtype=int)]
    )
    scores = torch.cat([exposed_score, withheld_score]).numpy()
    return float(roc_auc_score(labels, scores))


def _centroid_margin(
    values: torch.Tensor,
    basis: torch.Tensor,
    exposed_center: torch.Tensor,
    withheld_center: torch.Tensor,
) -> torch.Tensor:
    projected = values @ basis
    exposed_distance = (projected - exposed_center).square().sum(1)
    withheld_distance = (projected - withheld_center).square().sum(1)
    return withheld_distance - exposed_distance


def _fit_layer(
    exposed: torch.Tensor,
    withheld: torch.Tensor,
    train: np.ndarray,
    ranks: list[int],
    seed: int,
) -> dict[str, Any]:
    train_mask = torch.from_numpy(train)
    center = torch.cat([exposed[train_mask], withheld[train_mask]]).mean(0)
    exposed_unit = _unit_center(exposed, center)
    withheld_unit = _unit_center(withheld, center)
    delta = exposed_unit - withheld_unit
    maximum = min(max(ranks), delta[train_mask].shape[0], delta.shape[1])
    basis, spectrum = _fit_subspace(delta[train_mask], maximum, 1.0, seed)
    direction = delta[train_mask].mean(0)
    direction = direction / direction.norm().clamp_min(1e-12)
    fits = {}
    for rank in ranks:
        if rank > basis.shape[1]:
            continue
        vectors = basis[:, :rank]
        train_exposed = exposed_unit[train_mask] @ vectors
        train_withheld = withheld_unit[train_mask] @ vectors
        exposed_center = train_exposed.mean(0)
        withheld_center = train_withheld.mean(0)
        exposed_energy = train_exposed.square().sum(1).mean()
        withheld_energy = train_withheld.square().sum(1).mean()
        energy_sign = 1.0 if exposed_energy >= withheld_energy else -1.0
        fits[rank] = {
            "basis": vectors,
            "exposed_center": exposed_center,
            "withheld_center": withheld_center,
            "energy_sign": energy_sign,
        }
    return {
        "center": center,
        "exposed_unit": exposed_unit,
        "withheld_unit": withheld_unit,
        "delta": delta,
        "direction": direction,
        "basis": basis,
        "spectrum": spectrum,
        "ranks": fits,
        "train_mask": train_mask,
    }


def _synthetic_scores(
    fit: dict[str, Any], method: str, rank: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    exposed = fit["exposed_unit"]
    withheld = fit["withheld_unit"]
    if method == "cosine":
        return exposed @ fit["direction"], withheld @ fit["direction"]
    rank_fit = fit["ranks"][int(rank)]
    basis = rank_fit["basis"]
    if method == "signed_subspace":
        return (
            _centroid_margin(
                exposed, basis, rank_fit["exposed_center"], rank_fit["withheld_center"]
            ),
            _centroid_margin(
                withheld, basis, rank_fit["exposed_center"], rank_fit["withheld_center"]
            ),
        )
    if method == "energy":
        sign = rank_fit["energy_sign"]
        return sign * (exposed @ basis).square().sum(1), sign * (withheld @ basis).square().sum(1)
    raise ValueError(f"Unknown memory score: {method}")


def _benchmark_score(
    raw: torch.Tensor, fit: dict[str, Any], method: str, rank: int | None
) -> np.ndarray:
    values = _unit_center(raw, fit["center"])
    if method == "cosine":
        exposure_score = values @ fit["direction"]
    else:
        rank_fit = fit["ranks"][int(rank)]
        basis = rank_fit["basis"]
        if method == "signed_subspace":
            exposure_score = _centroid_margin(
                values, basis, rank_fit["exposed_center"], rank_fit["withheld_center"]
            )
        elif method == "energy":
            exposure_score = rank_fit["energy_sign"] * (values @ basis).square().sum(1)
        else:
            raise ValueError(f"Unknown memory score: {method}")
    return (-exposure_score).numpy()


def _select(
    fits: dict[int, dict[str, Any]],
    validation: np.ndarray,
    test: np.ndarray,
    ranks: list[int],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for method in ("cosine", "signed_subspace", "energy"):
        candidates = []
        for layer_order, (layer, fit) in enumerate(fits.items()):
            method_ranks: list[int | None] = [None] if method == "cosine" else ranks
            for rank in method_ranks:
                if rank is not None and rank not in fit["ranks"]:
                    continue
                exposed, withheld = _synthetic_scores(fit, method, rank)
                validation_mask = torch.from_numpy(validation)
                auroc = _exposure_auroc(
                    exposed[validation_mask], withheld[validation_mask]
                )
                candidates.append((auroc, -(rank or 0), -layer_order, layer, rank))
        if not candidates:
            raise RuntimeError(f"No valid candidates for {method}")
        _, _, _, layer, rank = max(candidates)
        exposed, withheld = _synthetic_scores(fits[layer], method, rank)
        test_mask = torch.from_numpy(test)
        selected[method] = {
            "layer": int(layer),
            "rank": int(rank) if rank is not None else None,
            "validation_exposure_auroc": float(max(candidates)[0]),
            "test_exposure_auroc": _exposure_auroc(
                exposed[test_mask], withheld[test_mask]
            ),
        }
    return selected


def _paired_bootstrap(
    labels: np.ndarray,
    proposed: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "aurc")
    proposed_metrics = _ranking_metrics(labels, proposed)
    baseline_metrics = _ranking_metrics(labels, baseline)
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        first = _ranking_metrics(labels[indices], proposed[indices])
        second = _ranking_metrics(labels[indices], baseline[indices])
        for name in names:
            value = first[name] - second[name]
            if np.isfinite(value):
                draws[name].append(float(value))
    return {
        name: {
            "difference": float(proposed_metrics[name] - baseline_metrics[name]),
            "ci95": [
                float(np.quantile(draws[name], 0.025)),
                float(np.quantile(draws[name], 0.975)),
            ],
        }
        for name in names
    }


def _baselines(
    config: dict[str, Any], settings: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    candidates = [
        output_dir(config) / "results" / "trivia_qa" / "predictions.jsonl",
        source_output_dir(config) / "results" / "trivia_qa" / "predictions.jsonl",
    ]
    ordered = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        predictions = read_jsonl(candidate)
        by_id = {record["fact_id"]: record for record in predictions}
        if any(record["fact_id"] not in by_id for record in records):
            continue
        candidate_order = [by_id[record["fact_id"]] for record in records]
        if all(
            str(source.get("answer", "")) == str(saved.get("answer", ""))
            for source, saved in zip(records, candidate_order, strict=True)
        ):
            ordered = candidate_order
            break
    if ordered is None:
        raise RuntimeError(
            "No TriviaQA predictions exactly match the generations used for the hidden states"
        )
    result = {
        "token_logprob": 1.0
        - np.exp(np.asarray([record["mean_logprob"] for record in ordered], dtype=float)),
        "token_entropy": 1.0
        - np.exp(-np.asarray([record["mean_entropy"] for record in ordered], dtype=float)),
    }
    prefix = settings.get("probe_baseline")
    if prefix:
        keys = sorted(key for key in ordered[0] if key.startswith(f"{prefix}/seed_"))
        if keys:
            result[f"probe_ensemble/{prefix}"] = np.mean(
                [[float(record[key]) for record in ordered] for key in keys], axis=0
            )
    return result


def _control_exposure_score(
    values: torch.Tensor,
    fit: dict[str, Any],
    method: str,
    basis_or_direction: torch.Tensor,
) -> torch.Tensor:
    train = fit["train_mask"]
    exposed = fit["exposed_unit"][train]
    withheld = fit["withheld_unit"][train]
    if method == "cosine":
        direction = basis_or_direction
        if float((exposed @ direction).mean()) < float((withheld @ direction).mean()):
            direction = -direction
        return values @ direction
    basis = basis_or_direction
    exposed_projected = exposed @ basis
    withheld_projected = withheld @ basis
    if method == "signed_subspace":
        return _centroid_margin(
            values,
            basis,
            exposed_projected.mean(0),
            withheld_projected.mean(0),
        )
    exposed_energy = exposed_projected.square().sum(1).mean()
    withheld_energy = withheld_projected.square().sum(1).mean()
    sign = 1.0 if exposed_energy >= withheld_energy else -1.0
    return sign * (values @ basis).square().sum(1)


def _control_summary(
    raw: torch.Tensor,
    labels: np.ndarray,
    fit: dict[str, Any],
    method: str,
    rank: int | None,
    records: list[dict[str, Any]],
    genuine_auroc: float,
    random_repeats: int,
    shuffled_repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = _unit_center(raw, fit["center"])
    width = values.shape[1]
    random_aurocs = []
    shuffled_aurocs = []
    train_indices = fit["train_mask"].nonzero(as_tuple=False).squeeze(1).numpy()
    groups = np.asarray(
        [f"{record['relation']}/{record['adapter_assignment']}" for record in records]
    )[train_indices]
    for repeat in range(max(random_repeats, shuffled_repeats)):
        if method == "cosine":
            if repeat < random_repeats:
                random_direction = torch.from_numpy(
                    rng.standard_normal(width).astype(np.float32)
                )
                random_direction /= random_direction.norm().clamp_min(1e-12)
                random_exposure = _control_exposure_score(
                    values, fit, method, random_direction
                )
            if repeat < shuffled_repeats:
                signs = torch.from_numpy(
                    rng.choice(
                        np.asarray([-1.0, 1.0], dtype=np.float32), len(train_indices)
                    )
                )
                shuffled_direction = (
                    fit["delta"][fit["train_mask"]] * signs.unsqueeze(1)
                ).mean(0)
                if float(shuffled_direction.norm()) <= 1e-12:
                    shuffled_direction = torch.ones(width)
                    shuffled_direction /= shuffled_direction.norm()
                else:
                    shuffled_direction /= shuffled_direction.norm()
                shuffled_exposure = _control_exposure_score(
                    values, fit, method, shuffled_direction
                )
        else:
            if repeat < random_repeats:
                matrix = torch.from_numpy(
                    rng.standard_normal((width, int(rank))).astype(np.float32)
                )
                random_basis, _ = torch.linalg.qr(matrix, mode="reduced")
                random_exposure = _control_exposure_score(
                    values, fit, method, random_basis
                )
            if repeat < shuffled_repeats:
                permuted = train_indices.copy()
                for group in sorted(set(groups)):
                    positions = np.flatnonzero(groups == group)
                    permuted[positions] = rng.permutation(train_indices[positions])
                shuffled_delta = (
                    fit["exposed_unit"][torch.from_numpy(train_indices)]
                    - fit["withheld_unit"][torch.from_numpy(permuted)]
                )
                shuffled_basis, _ = _fit_subspace(
                    shuffled_delta,
                    int(rank),
                    1.0,
                    seed + 1000 + repeat,
                )
                shuffled_exposure = _control_exposure_score(
                    values, fit, method, shuffled_basis[:, : int(rank)]
                )
        if repeat < random_repeats:
            random_aurocs.append(
                _ranking_metrics(labels, (-random_exposure).numpy())["auroc"]
            )
        if repeat < shuffled_repeats:
            shuffled_aurocs.append(
                _ranking_metrics(labels, (-shuffled_exposure).numpy())["auroc"]
            )
    return {
        "random_auroc_mean": float(np.nanmean(random_aurocs)),
        "random_auroc_std": float(np.nanstd(random_aurocs, ddof=1)),
        "fraction_random_at_least_genuine": float(
            np.mean(np.asarray(random_aurocs) >= genuine_auroc)
        ),
        "shuffled_auroc_mean": float(np.nanmean(shuffled_aurocs)),
        "shuffled_auroc_std": float(np.nanstd(shuffled_aurocs, ddof=1)),
        "fraction_shuffled_at_least_genuine": float(
            np.mean(np.asarray(shuffled_aurocs) >= genuine_auroc)
        ),
        "random_repeats": random_repeats,
        "shuffled_repeats": shuffled_repeats,
    }


def analyze_memory_subspace(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    validate_source_run(config)
    target = memory_subspace_root(config)
    fingerprint = memory_subspace_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    metrics_path = target / "metrics.json"
    if done.exists() and metrics_path.exists() and not force:
        return metrics_path

    source = source_output_dir(config)
    layers = [int(layer) for layer in settings["layers"]]
    geometry_features = source / "geometry" / "multilayer_v1" / "features"
    answer_features = source / "comparisons" / "haloscope_matched_v1" / "features"

    synthetic: dict[str, tuple[list[dict[str, Any]], dict[str, dict[int, torch.Tensor]]]] = {}
    for adapter in ("adapter_a", "adapter_b"):
        prompt_records, prompt = _load_layers(
            geometry_features / "synthetic" / adapter, layers
        )
        answer_records, answer = _load_layers(
            answer_features / "synthetic" / adapter, layers
        )
        answer = _align_features(prompt_records, answer_records, answer)
        synthetic[adapter] = (prompt_records, _representations(prompt, answer))

    benchmark = str(settings["benchmark"])
    benchmark_prompt_records, benchmark_prompt = _load_layers(
        geometry_features / "benchmarks" / benchmark, layers
    )
    benchmark_answer_records, benchmark_answer = _load_layers(
        answer_features / "benchmarks" / benchmark, layers
    )
    benchmark_answer = _align_features(
        benchmark_prompt_records, benchmark_answer_records, benchmark_answer
    )
    benchmark_representations = _representations(benchmark_prompt, benchmark_answer)
    scored = np.asarray([record["correct"] >= 0 for record in benchmark_prompt_records])
    labels = np.asarray(
        [max(0, record["correct"]) for record in benchmark_prompt_records], dtype=int
    )[scored]
    scored_records = [
        record
        for record, include in zip(benchmark_prompt_records, scored, strict=True)
        if include
    ]
    baselines = _baselines(config, settings, scored_records)

    results: dict[str, Any] = {
        "scope": {
            "benchmark": benchmark,
            "benchmark_labels_used_for_fitting_or_selection": False,
            "selection": "synthetic validation entities only",
            "synthetic_pairing": "same question and same replayed answer across adapters",
            "centering": "synthetic training center",
            "normalization": "per-example L2 after centering",
            "truthful_qa_included": False,
        },
        "representations": {},
        "baselines": {
            name: _ranking_metrics(labels, values) for name, values in baselines.items()
        },
        **_metadata(config, settings),
    }

    records_a, reps_a = synthetic["adapter_a"]
    records_b, reps_b = synthetic["adapter_b"]
    for representation_index, representation in enumerate(settings["representations"]):
        checkpoint = target / f"{representation}.json"
        if checkpoint.exists() and not force:
            with checkpoint.open("r", encoding="utf-8") as handle:
                results["representations"][representation] = json.load(handle)
            continue
        fits: dict[int, dict[str, Any]] = {}
        metadata = None
        for layer_index, layer in enumerate(layers):
            paired_records, exposed, withheld = _paired(
                records_a,
                reps_a[representation][layer],
                records_b,
                reps_b[representation][layer],
            )
            if metadata is None:
                metadata = paired_records
            elif [row["fact_id"] for row in metadata] != [
                row["fact_id"] for row in paired_records
            ]:
                raise RuntimeError("Synthetic identities changed across layers")
            split = np.asarray([record["split"] for record in paired_records])
            fits[layer] = _fit_layer(
                exposed,
                withheld,
                split == "train",
                [int(rank) for rank in settings["rank_candidates"]],
                int(settings["seed"]) + representation_index * 100 + layer_index,
            )
        split = np.asarray([record["split"] for record in metadata])
        selected = _select(
            fits,
            split == "validation",
            split == "test",
            [int(rank) for rank in settings["rank_candidates"]],
        )
        representation_result: dict[str, Any] = {"methods": {}}
        for method_index, (method, selection) in enumerate(selected.items()):
            layer = int(selection["layer"])
            rank = selection["rank"]
            raw = benchmark_representations[representation][layer][
                torch.from_numpy(scored)
            ]
            risk = _benchmark_score(raw, fits[layer], method, rank)
            ranking = _ranking_metrics(labels, risk)
            ranking["ci95"] = _bootstrap_ranking(
                labels,
                risk,
                int(settings["bootstrap_samples"]),
                int(settings["seed"]) + representation_index * 1000 + method_index,
            )
            comparison_baselines = {
                name: values
                for name, values in baselines.items()
                if name == "token_entropy" or name.startswith("probe_ensemble/")
            }
            comparisons = {
                name: _paired_bootstrap(
                    labels,
                    risk,
                    baseline,
                    int(settings["bootstrap_samples"]),
                    int(settings["seed"])
                    + 5000
                    + representation_index * 100
                    + method_index * 10
                    + index,
                )
                for index, (name, baseline) in enumerate(comparison_baselines.items())
            }
            representation_result["methods"][method] = {
                "selection": selection,
                "benchmark_ranking": ranking,
                "random_control": _control_summary(
                    raw,
                    labels,
                    fits[layer],
                    method,
                    rank,
                    metadata,
                    ranking["auroc"],
                    int(settings["control_repeats"]),
                    (
                        int(settings["control_repeats"])
                        if method == "cosine"
                        else int(settings["subspace_control_repeats"])
                    ),
                    int(settings["seed"])
                    + 10000
                    + representation_index * 100
                    + method_index,
                ),
                "paired_bootstrap": comparisons,
            }
        results["representations"][representation] = representation_result
        atomic_json(checkpoint, representation_result)

    atomic_json(metrics_path, results)
    summary_path = _write_summary(results, target / "summary.md")
    mark_complete(
        done,
        {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            **_metadata(config, settings),
        },
    )
    return metrics_path


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.4f}" if np.isfinite(number) else "NA"


def _write_summary(results: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Controlled-memory cosine and subspace analysis",
        "",
        "Subspaces and all layer/rank choices use only synthetic exposed/withheld training and "
        "validation entities. TriviaQA labels are used only for final evaluation.",
        "",
        "The synthetic adapter pair receives the same question and same replayed answer, so its "
        "difference cannot be explained by answer wording.",
        "",
        "## TriviaQA baselines",
        "",
        "| Baseline | AUROC | AUPRC | AURC |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in results["baselines"].items():
        lines.append(
            f"| {name} | {_fmt(metrics['auroc'])} | {_fmt(metrics['auprc'])} | "
            f"{_fmt(metrics['aurc'])} |"
        )
    lines.extend(
        [
            "",
            "## Controlled-memory scores",
            "",
            "| Representation | Method | Layer | Rank | Synthetic validation AUROC | "
            "Synthetic test AUROC | TriviaQA AUROC | 95% CI | AUPRC | AURC | "
            "Shuffled AUROC | Random AUROC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for representation, result in results["representations"].items():
        for method, item in result["methods"].items():
            selection = item["selection"]
            ranking = item["benchmark_ranking"]
            lower, upper = ranking["ci95"]["auroc"]
            lines.append(
                f"| {representation} | {method} | {selection['layer']} | "
                f"{selection['rank'] if selection['rank'] is not None else 'NA'} | "
                f"{_fmt(selection['validation_exposure_auroc'])} | "
                f"{_fmt(selection['test_exposure_auroc'])} | {_fmt(ranking['auroc'])} | "
                f"[{_fmt(lower)}, {_fmt(upper)}] | {_fmt(ranking['auprc'])} | "
                f"{_fmt(ranking['aurc'])} | "
                f"{_fmt(item['random_control']['shuffled_auroc_mean'])} | "
                f"{_fmt(item['random_control']['random_auroc_mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired TriviaQA comparisons",
            "",
            "Positive AUROC differences favor the controlled-memory score.",
            "",
            "| Representation | Method | Baseline | AUROC difference | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for representation, result in results["representations"].items():
        for method, item in result["methods"].items():
            for baseline, comparison in item["paired_bootstrap"].items():
                auroc = comparison["auroc"]
                lower, upper = auroc["ci95"]
                lines.append(
                    f"| {representation} | {method} | {baseline} | "
                    f"{_fmt(auroc['difference'])} | [{_fmt(lower)}, {_fmt(upper)}] |"
                )
    lines.extend(
        [
            "",
            "Positive AUPRC differences and negative AURC differences also favor the memory "
            "score; those intervals are retained in `metrics.json`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
