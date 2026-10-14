from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .adapter_invariant import (
    SOURCES,
    _arrays,
    _comparison_run_root,
    _identity_auroc,
    _identity_leakage,
    _load_same_answer_synthetic,
    _load_synthetic,
    _project_out,
    _nuisance_basis,
)
from .collect import configured_feature_keys, load_feature_shards
from .comparison import _load_feature_layers
from .config import config_fingerprint, output_dir, validate_source_run
from .geometry import _bootstrap_ranking, _ranking_metrics
from .io_utils import atomic_json, atomic_torch_save, ensure_manifest, mark_complete
from .versioning import artifact_metadata


ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION = 1


def load_adapter_invariant_subspace_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Adapter-invariant subspace configuration must be a mapping")
    if int(settings.get("adapter_invariant_subspace_schema_version", -1)) != (
        ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Expected adapter-invariant subspace schema "
            f"{ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION}, found "
            f"{settings.get('adapter_invariant_subspace_schema_version')}"
        )
    for key, allow_zero in (
        ("nuisance_rank_candidates", True),
        ("correctness_rank_candidates", False),
    ):
        values = settings.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list")
        ranks = [int(value) for value in values]
        if any(value < (0 if allow_zero else 1) for value in ranks):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{key} must contain {qualifier} integers")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"{key} must not contain duplicates")
    leakage = float(settings.get("max_adapter_identity_auroc", 0.0))
    if not 0.5 <= leakage <= 1.0:
        raise ValueError("max_adapter_identity_auroc must be in [0.5, 1.0]")
    correctness = float(settings.get("minimum_worst_adapter_auroc", 0.0))
    if not 0.5 <= correctness <= 1.0:
        raise ValueError("minimum_worst_adapter_auroc must be in [0.5, 1.0]")
    if int(settings.get("minimum_group_examples", 0)) < 2:
        raise ValueError("minimum_group_examples must be at least 2")
    if int(settings.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    layers = settings.get("same_answer_layers")
    if not isinstance(layers, list) or not all(
        isinstance(layer, int) for layer in layers
    ):
        raise ValueError("same_answer_layers must be a list of integers")
    if len(set(layers)) != len(layers):
        raise ValueError("same_answer_layers must not contain duplicates")
    int(settings["seed"])
    return settings


def adapter_invariant_subspace_fingerprint(
    config: dict[str, Any], settings: dict[str, Any]
) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "adapter_invariant_subspace": settings,
        "adapter_invariant_subspace_schema_version": (
            ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def adapter_invariant_subspace_root(config: dict[str, Any]) -> Path:
    return (
        output_dir(config)
        / "adapter_invariant_subspace"
        / f"v{ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION}"
    )


def _metadata(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_invariant_subspace_schema_version": (
            ADAPTER_INVARIANT_SUBSPACE_SCHEMA_VERSION
        ),
        "adapter_invariant_subspace_fingerprint": (
            adapter_invariant_subspace_fingerprint(config, settings)
        ),
        **artifact_metadata(),
    }


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def _correctness_vectors(
    values: torch.Tensor,
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    minimum: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]]]:
    vectors = []
    by_source: dict[str, list[torch.Tensor]] = {source: [] for source in SOURCES}
    metadata = []
    for relation in sorted(set(arrays["relation"][fit_mask])):
        relation_vectors = {}
        for source in SOURCES:
            group = fit_mask & (arrays["relation"] == relation) & (
                arrays["source"] == source
            )
            correct = group & (arrays["correct"] == 1)
            error = group & (arrays["correct"] == 0)
            if int(correct.sum()) < minimum or int(error.sum()) < minimum:
                continue
            direction = (
                values[torch.from_numpy(correct)].mean(0)
                - values[torch.from_numpy(error)].mean(0)
            )
            if float(direction.norm()) <= 1e-12:
                continue
            relation_vectors[source] = _unit(direction)
        # A relation contributes only as a matched A/B pair. This prevents a missing class
        # in one adapter from silently turning the shared subspace into a source-specific one.
        if set(relation_vectors) != set(SOURCES):
            continue
        for source in SOURCES:
            direction = relation_vectors[source]
            vectors.append(direction)
            by_source[source].append(direction)
            metadata.append({"source": source, "relation": str(relation)})
    missing = [source for source, items in by_source.items() if not items]
    if missing:
        raise ValueError(f"No estimable relation-conditioned directions for {missing}")
    return (
        torch.stack(vectors),
        {source: torch.stack(items) for source, items in by_source.items()},
        metadata,
    )


def _subspace_overlap(
    source_vectors: dict[str, torch.Tensor], rank: int
) -> dict[str, Any]:
    bases = {}
    for source, vectors in source_vectors.items():
        _, _, right = torch.linalg.svd(vectors, full_matrices=False)
        source_rank = min(rank, right.shape[0])
        bases[source] = right[:source_rank].T.contiguous()
    singular = torch.linalg.svdvals(bases["adapter_a"].T @ bases["adapter_b"])
    return {
        "effective_rank_a": int(bases["adapter_a"].shape[1]),
        "effective_rank_b": int(bases["adapter_b"].shape[1]),
        "principal_cosines": [float(value) for value in singular],
        "mean_cosine_squared": float(singular.square().mean()),
    }


def _class_prototypes(
    projected: torch.Tensor,
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    minimum: int,
) -> dict[str, dict[int, torch.Tensor]]:
    prototypes: dict[str, dict[int, torch.Tensor]] = {}
    for source in SOURCES:
        prototypes[source] = {}
        for correct in (0, 1):
            mask = fit_mask & (arrays["source"] == source) & (
                arrays["correct"] == correct
            )
            if int(mask.sum()) < minimum:
                raise ValueError(
                    f"{source} needs at least {minimum} examples of correctness={correct}"
                )
            prototypes[source][correct] = projected[torch.from_numpy(mask)].mean(0)
    return prototypes


def _fit_subspace_pattern(
    raw_features: torch.Tensor,
    records: list[dict[str, Any]],
    fit_mask: np.ndarray,
    nuisance_rank: int,
    correctness_rank: int,
    minimum: int,
) -> dict[str, Any]:
    arrays = _arrays(records)
    fit_tensor = torch.from_numpy(fit_mask)
    mean = raw_features[fit_tensor].mean(0)
    std = raw_features[fit_tensor].std(0).clamp_min(1e-6)
    standardized = (raw_features.float() - mean) / std
    nuisance, nuisance_count = _nuisance_basis(
        standardized, arrays, fit_mask, nuisance_rank, minimum
    )
    residual = _project_out(standardized, nuisance)
    vectors, source_vectors, vector_metadata = _correctness_vectors(
        residual, arrays, fit_mask, minimum
    )
    if correctness_rank > vectors.shape[0]:
        raise ValueError(
            f"Requested correctness rank {correctness_rank}, but only "
            f"{vectors.shape[0]} relation-conditioned directions are available"
        )
    _, singular, right = torch.linalg.svd(vectors, full_matrices=False)
    basis = right[:correctness_rank].T.contiguous()
    projected = residual @ basis
    prototypes = _class_prototypes(projected, arrays, fit_mask, minimum)
    identity_prototypes = {
        correct: {
            source: prototypes[source][correct]
            for source in SOURCES
        }
        for correct in (0, 1)
    }
    return {
        "mean": mean,
        "std": std,
        "nuisance_basis": nuisance,
        "nuisance_rank": nuisance_rank,
        "nuisance_direction_count": nuisance_count,
        "correctness_basis": basis,
        "correctness_rank": correctness_rank,
        "correctness_direction_count": int(vectors.shape[0]),
        "correctness_singular_values": singular[:correctness_rank],
        "direction_metadata": vector_metadata,
        "class_prototypes": prototypes,
        "identity_prototypes": identity_prototypes,
        "source_subspace_overlap": _subspace_overlap(source_vectors, correctness_rank),
    }


def _project(raw_features: torch.Tensor, fit: dict[str, Any]) -> torch.Tensor:
    standardized = (raw_features.float() - fit["mean"]) / fit["std"]
    residual = _project_out(standardized, fit["nuisance_basis"])
    return residual @ fit["correctness_basis"]


def _risk_scores(raw_features: torch.Tensor, fit: dict[str, Any]) -> np.ndarray:
    projected = _project(raw_features, fit)
    distances = {
        correct: torch.stack(
            [
                (projected - fit["class_prototypes"][source][correct])
                .square()
                .sum(1)
                for source in SOURCES
            ]
        ).min(0).values
        for correct in (0, 1)
    }
    return (distances[1] - distances[0]).numpy()


def _subspace_identity_leakage(
    raw_features: torch.Tensor,
    sources: np.ndarray,
    correctness: np.ndarray,
    fit: dict[str, Any],
) -> dict[str, Any]:
    projected = _project(raw_features, fit)
    scores = np.empty(len(sources), dtype=float)
    conditioned = {}
    for correct in (0, 1):
        mask = correctness == correct
        values = projected[torch.from_numpy(mask)]
        prototypes = fit["identity_prototypes"][correct]
        distance_a = (values - prototypes["adapter_a"]).square().sum(1)
        distance_b = (values - prototypes["adapter_b"]).square().sum(1)
        score = (distance_a - distance_b).numpy()
        scores[mask] = score
        conditioned[str(correct)] = _identity_auroc(sources[mask], score)
    overall = _identity_auroc(sources, scores)
    available = [
        overall["orientation_free_auroc"],
        *[value["orientation_free_auroc"] for value in conditioned.values()],
    ]
    finite = [value for value in available if np.isfinite(value)]
    return {
        "overall": overall,
        "conditioned_on_correctness": conditioned,
        "worst_orientation_free_auroc": max(finite) if finite else float("nan"),
    }


def _evaluate(
    raw_features: torch.Tensor,
    records: list[dict[str, Any]],
    mask: np.ndarray,
    fit: dict[str, Any],
) -> dict[str, Any]:
    arrays = _arrays(records)
    indices = np.flatnonzero(mask)
    selected = raw_features[torch.from_numpy(mask)]
    risk = _risk_scores(selected, fit)
    labels = arrays["correct"][indices]
    sources = arrays["source"][indices]
    result: dict[str, Any] = {
        "pooled": _ranking_metrics(labels, risk),
        "by_adapter": {},
        "risk_adapter_identity": _identity_leakage(sources, risk, labels),
        "subspace_adapter_identity": _subspace_identity_leakage(
            selected, sources, labels, fit
        ),
    }
    for source in SOURCES:
        source_mask = sources == source
        result["by_adapter"][source] = _ranking_metrics(
            labels[source_mask], risk[source_mask]
        )
    result["worst_adapter_auroc"] = min(
        result["by_adapter"][source]["auroc"] for source in SOURCES
    )
    result["worst_identity_auroc"] = max(
        result["risk_adapter_identity"]["worst_orientation_free_auroc"],
        result["subspace_adapter_identity"]["worst_orientation_free_auroc"],
    )
    return result


def _select_pattern(
    raw_features: torch.Tensor,
    records: list[dict[str, Any]],
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for nuisance_value in settings["nuisance_rank_candidates"]:
        for correctness_value in settings["correctness_rank_candidates"]:
            nuisance_rank = int(nuisance_value)
            correctness_rank = int(correctness_value)
            try:
                fit = _fit_subspace_pattern(
                    raw_features,
                    records,
                    fit_mask,
                    nuisance_rank,
                    correctness_rank,
                    int(settings["minimum_group_examples"]),
                )
                validation = _evaluate(
                    raw_features, records, validation_mask, fit
                )
            except ValueError as error:
                candidates.append(
                    {
                        "nuisance_rank": nuisance_rank,
                        "correctness_rank": correctness_rank,
                        "error": str(error),
                    }
                )
                continue
            if not np.isfinite(validation["worst_adapter_auroc"]):
                candidates.append(
                    {
                        "nuisance_rank": nuisance_rank,
                        "correctness_rank": correctness_rank,
                        "error": "Validation needs both correctness classes in both adapters",
                    }
                )
                continue
            if not np.isfinite(validation["worst_identity_auroc"]):
                candidates.append(
                    {
                        "nuisance_rank": nuisance_rank,
                        "correctness_rank": correctness_rank,
                        "error": "Validation cannot estimate adapter-identity leakage",
                    }
                )
                continue
            candidates.append(
                {
                    "nuisance_rank": nuisance_rank,
                    "correctness_rank": correctness_rank,
                    "source_subspace_overlap": fit["source_subspace_overlap"],
                    "validation": validation,
                }
            )
    valid = [candidate for candidate in candidates if "validation" in candidate]
    if not valid:
        raise RuntimeError("No valid adapter-invariant subspace candidate")
    max_identity = float(settings["max_adapter_identity_auroc"])
    min_correctness = float(settings["minimum_worst_adapter_auroc"])
    invariant = [
        candidate
        for candidate in valid
        if candidate["validation"]["worst_identity_auroc"] <= max_identity
        and candidate["validation"]["worst_adapter_auroc"] >= min_correctness
    ]
    if invariant:
        chosen = max(
            invariant,
            key=lambda item: (
                item["validation"]["worst_adapter_auroc"],
                -item["validation"]["worst_identity_auroc"],
                -item["correctness_rank"],
                -item["nuisance_rank"],
            ),
        )
    else:
        chosen = min(
            valid,
            key=lambda item: (
                item["validation"]["worst_identity_auroc"],
                -item["validation"]["worst_adapter_auroc"],
                item["correctness_rank"],
                item["nuisance_rank"],
            ),
        )
    fit = _fit_subspace_pattern(
        raw_features,
        records,
        fit_mask,
        int(chosen["nuisance_rank"]),
        int(chosen["correctness_rank"]),
        int(settings["minimum_group_examples"]),
    )
    validation = chosen["validation"]
    return fit, {
        "chosen_nuisance_rank": int(chosen["nuisance_rank"]),
        "chosen_correctness_rank": int(chosen["correctness_rank"]),
        "adapter_identity_constraint_satisfied": (
            validation["worst_identity_auroc"] <= max_identity
        ),
        "correctness_constraint_satisfied": (
            validation["worst_adapter_auroc"] >= min_correctness
        ),
        "pattern_found": (
            validation["worst_identity_auroc"] <= max_identity
            and validation["worst_adapter_auroc"] >= min_correctness
        ),
        "max_adapter_identity_auroc": max_identity,
        "minimum_worst_adapter_auroc": min_correctness,
        "candidates": candidates,
    }


def _leave_one_relation_out(
    raw_features: torch.Tensor,
    records: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    arrays = _arrays(records)
    results = {}
    for relation in sorted(set(arrays["relation"])):
        train = (arrays["split"] == "train") & (arrays["relation"] != relation)
        validation = (arrays["split"] == "validation") & (
            arrays["relation"] != relation
        )
        test = (arrays["split"] == "test") & (arrays["relation"] == relation)
        fit, selection = _select_pattern(
            raw_features, records, train, validation, settings
        )
        results[str(relation)] = {
            "selection": selection,
            "source_subspace_overlap": fit["source_subspace_overlap"],
            "heldout_test": _evaluate(raw_features, records, test, fit),
        }
    return results


def _benchmark_results(
    config: dict[str, Any],
    feature_mode: str,
    fit: dict[str, Any],
    settings: dict[str, Any],
    seed_offset: int,
    same_answer_layer: int | None,
) -> dict[str, Any]:
    results = {}
    for benchmark_index, benchmark in enumerate(config["benchmarks"]["datasets"]):
        if same_answer_layer is None:
            records, features = load_feature_shards(
                output_dir(config) / "features" / "benchmarks" / benchmark,
                feature_mode,
            )
        else:
            directory = (
                _comparison_run_root(config)
                / "comparisons"
                / "haloscope_matched_v1"
                / "features"
                / "benchmarks"
                / benchmark
            )
            if not any(directory.glob("shard_*.pt")):
                results[benchmark] = {
                    "status": "unavailable",
                    "reason": f"No same-answer benchmark features in {directory}",
                }
                continue
            records, layers = _load_feature_layers(directory, [same_answer_layer])
            features = layers[same_answer_layer]
        scored = np.asarray([record["correct"] >= 0 for record in records])
        labels = np.asarray(
            [max(0, record["correct"]) for record in records], dtype=int
        )[scored]
        risk = _risk_scores(features[torch.from_numpy(scored)], fit)
        ranking = _ranking_metrics(labels, risk)
        ranking["ci95"] = _bootstrap_ranking(
            labels,
            risk,
            int(settings["bootstrap_samples"]),
            int(settings["seed"]) + seed_offset * 100 + benchmark_index,
        )
        results[benchmark] = ranking
    return results


def _serializable_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: fit[key]
        for key in (
            "mean",
            "std",
            "nuisance_basis",
            "nuisance_rank",
            "nuisance_direction_count",
            "correctness_basis",
            "correctness_rank",
            "correctness_direction_count",
            "correctness_singular_values",
            "direction_metadata",
            "class_prototypes",
            "identity_prototypes",
            "source_subspace_overlap",
        )
    }


def _analyze_mode(
    config: dict[str, Any],
    settings: dict[str, Any],
    feature_mode: str,
    records: list[dict[str, Any]],
    raw_features: torch.Tensor,
    mode_index: int,
    same_answer_layer: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = _arrays(records)
    train = arrays["split"] == "train"
    validation = arrays["split"] == "validation"
    test = arrays["split"] == "test"
    fit, selection = _select_pattern(
        raw_features, records, train, validation, settings
    )
    result = {
        "same_answer_layer": same_answer_layer,
        "selection": selection,
        "source_subspace_overlap": fit["source_subspace_overlap"],
        "validation": _evaluate(raw_features, records, validation, fit),
        "synthetic_test": _evaluate(raw_features, records, test, fit),
        "leave_one_relation_out": _leave_one_relation_out(
            raw_features, records, settings
        ),
        "benchmarks": _benchmark_results(
            config,
            feature_mode,
            fit,
            settings,
            mode_index,
            same_answer_layer,
        ),
    }
    return result, _serializable_fit(fit)


def analyze_adapter_invariant_subspace(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    validate_source_run(config)
    target = adapter_invariant_subspace_root(config)
    fingerprint = adapter_invariant_subspace_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    metrics_path = target / "metrics.json"
    done = target / "_SUCCESS.json"
    if done.exists() and metrics_path.exists() and not force:
        return metrics_path

    results: dict[str, Any] = {
        "method": {
            "target": "multidimensional correct-versus-hallucinated structure",
            "subspace": (
                "SVD span of relation-conditioned A/B correctness directions"
            ),
            "adapter_removal": (
                "project out conditional adapter differences before fitting"
            ),
            "score": (
                "rotation-invariant nearest-source class-centroid distance margin"
            ),
            "selection": "synthetic validation only",
            "external_benchmark_labels_used_for_selection": False,
        },
        "feature_modes": {},
        **_metadata(config, settings),
    }
    pattern_payload = {"feature_modes": {}, **_metadata(config, settings)}
    mode_specs: list[tuple[str, int | None]] = [
        (mode, None) for mode in configured_feature_keys(config)
    ]
    mode_specs.extend(
        (f"same_answer__layer_{layer}", int(layer))
        for layer in settings["same_answer_layers"]
    )
    for mode_index, (feature_mode, same_answer_layer) in enumerate(mode_specs):
        checkpoint = target / "feature_modes" / f"{feature_mode}.json"
        pattern_checkpoint = target / "feature_modes" / f"{feature_mode}.pt"
        if checkpoint.exists() and pattern_checkpoint.exists() and not force:
            with checkpoint.open("r", encoding="utf-8") as handle:
                results["feature_modes"][feature_mode] = json.load(handle)
            saved = torch.load(pattern_checkpoint, map_location="cpu", weights_only=False)
            pattern_payload["feature_modes"][feature_mode] = saved["pattern"]
            continue
        if same_answer_layer is None:
            records, raw_features = _load_synthetic(config, feature_mode)
        else:
            records, raw_features = _load_same_answer_synthetic(
                config, same_answer_layer
            )
        try:
            mode_result, serializable = _analyze_mode(
                config,
                settings,
                feature_mode,
                records,
                raw_features,
                mode_index,
                same_answer_layer,
            )
        except (ValueError, RuntimeError) as error:
            if same_answer_layer is None:
                raise
            mode_result = {
                "same_answer_layer": same_answer_layer,
                "status": "not_estimable",
                "reason": str(error),
            }
            results["feature_modes"][feature_mode] = mode_result
            atomic_json(checkpoint, mode_result)
            continue
        results["feature_modes"][feature_mode] = mode_result
        pattern_payload["feature_modes"][feature_mode] = serializable
        atomic_json(checkpoint, mode_result)
        atomic_torch_save(
            pattern_checkpoint,
            {
                "feature_mode": feature_mode,
                "pattern": serializable,
                **_metadata(config, settings),
            },
        )

    atomic_json(metrics_path, results)
    atomic_torch_save(target / "patterns.pt", pattern_payload)
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
        "# Adapter-invariant multidimensional correctness subspace",
        "",
        "All nuisance-rank and correctness-rank choices use synthetic validation only. "
        "Natural benchmark labels are evaluation-only.",
        "",
        "| Feature mode | Found | Nuisance rank | Correctness rank | A/B subspace overlap | "
        "Test A AUROC | Test B AUROC | Risk identity AUROC | Subspace identity AUROC | "
        "Worst held-out relation AUROC | TriviaQA AUROC | TruthfulQA AUROC |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in results["feature_modes"].items():
        if result.get("status") == "not_estimable":
            lines.append(f"| {mode} | no | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        test = result["synthetic_test"]
        relation_worst = min(
            item["heldout_test"]["worst_adapter_auroc"]
            for item in result["leave_one_relation_out"].values()
        )
        trivia = result["benchmarks"].get("trivia_qa", {})
        truthful = result["benchmarks"].get("truthful_qa", {})
        selection = result["selection"]
        lines.append(
            f"| {mode} | {'yes' if selection['pattern_found'] else 'no'} | "
            f"{selection['chosen_nuisance_rank']} | "
            f"{selection['chosen_correctness_rank']} | "
            f"{_fmt(result['source_subspace_overlap']['mean_cosine_squared'])} | "
            f"{_fmt(test['by_adapter']['adapter_a']['auroc'])} | "
            f"{_fmt(test['by_adapter']['adapter_b']['auroc'])} | "
            f"{_fmt(test['risk_adapter_identity']['worst_orientation_free_auroc'])} | "
            f"{_fmt(test['subspace_adapter_identity']['worst_orientation_free_auroc'])} | "
            f"{_fmt(relation_worst)} | {_fmt(trivia.get('auroc'))} | "
            f"{_fmt(truthful.get('auroc'))} |"
        )
    lines.extend(
        [
            "",
            "A successful pattern must pass validation constraints for per-adapter correctness "
            "and both scalar-risk and full-subspace adapter leakage. Held-out-relation and "
            "benchmark results remain independent tests.",
            "",
            "The centroid-distance score is invariant to rotations or sign changes of the saved "
            "correctness basis. It does not use benchmark labels to orient the subspace.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
