from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from .collect import configured_feature_keys, load_feature_shards
from .comparison import _load_feature_layers
from .config import config_fingerprint, output_dir, source_output_dir, validate_source_run
from .geometry import _bootstrap_ranking, _ranking_metrics
from .io_utils import atomic_json, atomic_torch_save, ensure_manifest, mark_complete
from .versioning import artifact_metadata


ADAPTER_INVARIANT_SCHEMA_VERSION = 1
SOURCES = ("adapter_a", "adapter_b")


def load_adapter_invariant_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Adapter-invariant configuration must be a mapping")
    if int(settings.get("adapter_invariant_schema_version", -1)) != (
        ADAPTER_INVARIANT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Expected adapter-invariant schema {ADAPTER_INVARIANT_SCHEMA_VERSION}, "
            f"found {settings.get('adapter_invariant_schema_version')}"
        )
    ranks = settings.get("nuisance_rank_candidates")
    if not isinstance(ranks, list) or not ranks or any(int(rank) < 0 for rank in ranks):
        raise ValueError("nuisance_rank_candidates must contain non-negative integers")
    if len(set(int(rank) for rank in ranks)) != len(ranks):
        raise ValueError("nuisance_rank_candidates must not contain duplicates")
    leakage = float(settings.get("max_adapter_identity_auroc", 0.0))
    if not 0.5 <= leakage <= 1.0:
        raise ValueError("max_adapter_identity_auroc must be in [0.5, 1.0]")
    alignment = float(settings.get("minimum_direction_cosine", -2.0))
    if not -1.0 <= alignment <= 1.0:
        raise ValueError("minimum_direction_cosine must be in [-1.0, 1.0]")
    correctness = float(settings.get("minimum_worst_adapter_auroc", 0.0))
    if not 0.5 <= correctness <= 1.0:
        raise ValueError("minimum_worst_adapter_auroc must be in [0.5, 1.0]")
    if int(settings.get("minimum_group_examples", 0)) < 2:
        raise ValueError("minimum_group_examples must be at least 2")
    if int(settings.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    layers = settings.get("same_answer_layers")
    if not isinstance(layers, list) or not layers or not all(
        isinstance(layer, int) for layer in layers
    ):
        raise ValueError("same_answer_layers must be a non-empty list of integers")
    if len(set(layers)) != len(layers):
        raise ValueError("same_answer_layers must not contain duplicates")
    int(settings["seed"])
    return settings


def adapter_invariant_fingerprint(
    config: dict[str, Any], settings: dict[str, Any]
) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "adapter_invariant": settings,
        "adapter_invariant_schema_version": ADAPTER_INVARIANT_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def adapter_invariant_root(config: dict[str, Any]) -> Path:
    return (
        output_dir(config)
        / "adapter_invariant"
        / f"v{ADAPTER_INVARIANT_SCHEMA_VERSION}"
    )


def _metadata(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_invariant_schema_version": ADAPTER_INVARIANT_SCHEMA_VERSION,
        "adapter_invariant_fingerprint": adapter_invariant_fingerprint(config, settings),
        **artifact_metadata(),
    }


def _load_synthetic(
    config: dict[str, Any], feature_mode: str
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    directory = output_dir(config) / "features" / "synthetic"
    records: list[dict[str, Any]] = []
    features = []
    for source in SOURCES:
        source_records, source_features = load_feature_shards(
            directory / source, feature_mode
        )
        records.extend(source_records)
        features.append(source_features.float())
    return records, torch.cat(features)


def _arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "source": np.asarray([record["source"] for record in records]),
        "split": np.asarray([record["split"] for record in records]),
        "relation": np.asarray([record["relation"] for record in records]),
        "correct": np.asarray([record["correct"] for record in records], dtype=int),
    }


def _load_same_answer_synthetic(
    config: dict[str, Any], layer: int
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    run_root = _comparison_run_root(config)
    root = run_root / "comparisons" / "haloscope_matched_v1" / "features" / "synthetic"
    records_a, layers_a = _load_feature_layers(root / "adapter_a", [layer])
    records_b, layers_b = _load_feature_layers(root / "adapter_b", [layer])
    index_b = {record["fact_id"]: index for index, record in enumerate(records_b)}
    base_records = _load_records(run_root / "features" / "synthetic" / "base")
    correctness = {record["fact_id"]: int(record["correct"]) for record in base_records}
    records: list[dict[str, Any]] = []
    features = []
    for source, source_records, source_values in (
        ("adapter_a", records_a, layers_a[layer]),
        ("adapter_b", records_b, layers_b[layer]),
    ):
        for index, record in enumerate(source_records):
            if record["fact_id"] not in index_b or record["fact_id"] not in correctness:
                raise RuntimeError(
                    f"Missing paired same-answer metadata for {record['fact_id']}"
                )
            records.append(
                {
                    **record,
                    "source": source,
                    "correct": correctness[record["fact_id"]],
                }
            )
            features.append(source_values[index])
    if set(record["fact_id"] for record in records_a) != set(index_b):
        raise RuntimeError("Adapter A and B same-answer facts are not exactly paired")
    return records, torch.stack(features).float()


def _load_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
    if not records:
        raise FileNotFoundError(f"No feature records found in {directory}")
    return records


def _comparison_run_root(config: dict[str, Any]) -> Path:
    candidates = [output_dir(config), source_output_dir(config)]
    for candidate in dict.fromkeys(candidates):
        directory = candidate / "comparisons" / "haloscope_matched_v1" / "features"
        if all(
            any((directory / "synthetic" / source).glob("shard_*.pt"))
            for source in SOURCES
        ):
            return candidate
    raise FileNotFoundError(
        "No matched same-answer comparison features were found in the output or source run"
    )


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def _project_out(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[1] == 0:
        return values
    return values - (values @ basis) @ basis.T


def _nuisance_basis(
    standardized: torch.Tensor,
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    rank: int,
    minimum: int,
) -> tuple[torch.Tensor, int]:
    width = standardized.shape[1]
    if rank == 0:
        return torch.empty((width, 0), dtype=standardized.dtype), 0
    directions = []
    # Estimate adapter differences conditionally on correctness and relation. This prevents
    # the nuisance estimate from simply becoming another correctness direction.
    for relation in sorted(set(arrays["relation"][fit_mask])):
        for correct in (0, 1):
            group = fit_mask & (arrays["relation"] == relation) & (
                arrays["correct"] == correct
            )
            source_values = []
            for source in SOURCES:
                mask = group & (arrays["source"] == source)
                if int(mask.sum()) < minimum:
                    source_values = []
                    break
                source_values.append(standardized[torch.from_numpy(mask)].mean(0))
            if source_values:
                difference = source_values[0] - source_values[1]
                if float(difference.norm()) > 1e-12:
                    directions.append(_unit(difference))
    if rank > len(directions):
        raise ValueError(
            f"Requested nuisance rank {rank}, but only {len(directions)} conditional "
            "adapter directions are available"
        )
    matrix = torch.stack(directions)
    _, _, vectors = torch.linalg.svd(matrix, full_matrices=False)
    return vectors[:rank].T.contiguous(), len(directions)


def _correctness_direction(
    values: torch.Tensor,
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    source: str,
    minimum: int,
) -> torch.Tensor:
    source_mask = fit_mask & (arrays["source"] == source)
    correct = source_mask & (arrays["correct"] == 1)
    error = source_mask & (arrays["correct"] == 0)
    if int(correct.sum()) < minimum or int(error.sum()) < minimum:
        raise ValueError(
            f"{source} needs at least {minimum} correct and incorrect fitting examples"
        )
    direction = values[torch.from_numpy(correct)].mean(0) - values[
        torch.from_numpy(error)
    ].mean(0)
    if float(direction.norm()) <= 1e-12:
        raise ValueError(f"{source} correctness direction has zero norm")
    return _unit(direction)


def _fit_pattern(
    raw_features: torch.Tensor,
    records: list[dict[str, Any]],
    fit_mask: np.ndarray,
    nuisance_rank: int,
    minimum: int,
) -> dict[str, Any]:
    arrays = _arrays(records)
    fit_tensor = torch.from_numpy(fit_mask)
    mean = raw_features[fit_tensor].mean(0)
    std = raw_features[fit_tensor].std(0).clamp_min(1e-6)
    standardized = (raw_features - mean) / std
    nuisance, nuisance_count = _nuisance_basis(
        standardized, arrays, fit_mask, nuisance_rank, minimum
    )
    residual = _project_out(standardized, nuisance)
    source_directions = {
        source: _correctness_direction(
            residual, arrays, fit_mask, source, minimum
        )
        for source in SOURCES
    }
    alignment = float(
        source_directions["adapter_a"] @ source_directions["adapter_b"]
    )
    shared_sum = source_directions["adapter_a"] + source_directions["adapter_b"]
    if float(shared_sum.norm()) <= 1e-12:
        raise ValueError("A and B correctness directions cancel exactly")
    shared = _unit(shared_sum)
    return {
        "mean": mean,
        "std": std,
        "nuisance_basis": nuisance,
        "nuisance_rank": nuisance_rank,
        "nuisance_direction_count": nuisance_count,
        "source_directions": source_directions,
        "shared_direction": shared,
        "source_direction_cosine": alignment,
    }


def _scores(raw_features: torch.Tensor, fit: dict[str, Any]) -> np.ndarray:
    standardized = (raw_features.float() - fit["mean"]) / fit["std"]
    residual = _project_out(standardized, fit["nuisance_basis"])
    return (residual @ fit["shared_direction"]).numpy()


def _identity_auroc(sources: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = (sources == "adapter_b").astype(int)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "orientation_free_auroc": float("nan")}
    auroc = float(roc_auc_score(labels, scores))
    return {"auroc": auroc, "orientation_free_auroc": max(auroc, 1.0 - auroc)}


def _identity_leakage(
    sources: np.ndarray, scores: np.ndarray, correctness: np.ndarray
) -> dict[str, Any]:
    overall = _identity_auroc(sources, scores)
    conditioned = {
        str(label): _identity_auroc(sources[correctness == label], scores[correctness == label])
        for label in (0, 1)
    }
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
    score = _scores(raw_features[torch.from_numpy(mask)], fit)
    labels = arrays["correct"][indices]
    sources = arrays["source"][indices]
    result: dict[str, Any] = {
        "pooled": _ranking_metrics(labels, -score),
        "by_adapter": {},
        "adapter_identity": _identity_leakage(sources, score, labels),
    }
    for source in SOURCES:
        source_mask = sources == source
        result["by_adapter"][source] = _ranking_metrics(
            labels[source_mask], -score[source_mask]
        )
    result["worst_adapter_auroc"] = min(
        result["by_adapter"][source]["auroc"] for source in SOURCES
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
    for rank_value in settings["nuisance_rank_candidates"]:
        rank = int(rank_value)
        try:
            fit = _fit_pattern(
                raw_features,
                records,
                fit_mask,
                rank,
                int(settings["minimum_group_examples"]),
            )
        except ValueError as error:
            candidates.append({"nuisance_rank": rank, "error": str(error)})
            continue
        validation = _evaluate(raw_features, records, validation_mask, fit)
        if not np.isfinite(validation["worst_adapter_auroc"]):
            candidates.append(
                {
                    "nuisance_rank": rank,
                    "error": "Validation needs correct and incorrect examples in both adapters",
                }
            )
            continue
        if not np.isfinite(
            validation["adapter_identity"]["worst_orientation_free_auroc"]
        ):
            candidates.append(
                {
                    "nuisance_rank": rank,
                    "error": "Validation cannot estimate adapter-identity leakage",
                }
            )
            continue
        candidates.append(
            {
                "nuisance_rank": rank,
                "source_direction_cosine": fit["source_direction_cosine"],
                "validation": validation,
            }
        )
    valid = [candidate for candidate in candidates if "validation" in candidate]
    if not valid:
        raise RuntimeError("No valid adapter-invariant pattern candidate")
    threshold = float(settings["max_adapter_identity_auroc"])
    minimum_alignment = float(settings["minimum_direction_cosine"])
    minimum_correctness = float(settings["minimum_worst_adapter_auroc"])
    aligned = [
        candidate
        for candidate in valid
        if candidate["source_direction_cosine"] >= minimum_alignment
    ]
    invariant = [
        candidate
        for candidate in aligned
        if candidate["validation"]["adapter_identity"]["worst_orientation_free_auroc"]
        <= threshold
        and candidate["validation"]["worst_adapter_auroc"] >= minimum_correctness
    ]
    if invariant:
        chosen = max(
            invariant,
            key=lambda item: (
                item["validation"]["worst_adapter_auroc"],
                -item["validation"]["adapter_identity"]["worst_orientation_free_auroc"],
                -item["nuisance_rank"],
            ),
        )
    elif aligned:
        chosen = min(
            aligned,
            key=lambda item: (
                item["validation"]["adapter_identity"]["worst_orientation_free_auroc"],
                -item["validation"]["worst_adapter_auroc"],
                item["nuisance_rank"],
            ),
        )
    else:
        chosen = max(
            valid,
            key=lambda item: (
                item["source_direction_cosine"],
                -item["validation"]["adapter_identity"]["worst_orientation_free_auroc"],
                item["validation"]["worst_adapter_auroc"],
            ),
        )
    fit = _fit_pattern(
        raw_features,
        records,
        fit_mask,
        int(chosen["nuisance_rank"]),
        int(settings["minimum_group_examples"]),
    )
    selection = {
        "chosen_nuisance_rank": int(chosen["nuisance_rank"]),
        "adapter_identity_constraint_satisfied": (
            chosen["validation"]["adapter_identity"]["worst_orientation_free_auroc"]
            <= threshold
        ),
        "direction_alignment_constraint_satisfied": (
            chosen["source_direction_cosine"] >= minimum_alignment
        ),
        "correctness_constraint_satisfied": (
            chosen["validation"]["worst_adapter_auroc"] >= minimum_correctness
        ),
        "pattern_found": (
            chosen["source_direction_cosine"] >= minimum_alignment
            and chosen["validation"]["adapter_identity"]["worst_orientation_free_auroc"]
            <= threshold
            and chosen["validation"]["worst_adapter_auroc"] >= minimum_correctness
        ),
        "max_adapter_identity_auroc": threshold,
        "minimum_direction_cosine": minimum_alignment,
        "minimum_worst_adapter_auroc": minimum_correctness,
        "candidates": candidates,
    }
    return fit, selection


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
        results[relation] = {
            "selection": selection,
            "train_source_direction_cosine": fit["source_direction_cosine"],
            "heldout_test": _evaluate(raw_features, records, test, fit),
        }
    return results


def _benchmark_results(
    config: dict[str, Any],
    feature_mode: str,
    fit: dict[str, Any],
    settings: dict[str, Any],
    seed_offset: int,
    same_answer_layer: int | None = None,
) -> dict[str, Any]:
    results = {}
    for benchmark_index, benchmark in enumerate(config["benchmarks"]["datasets"]):
        if same_answer_layer is None:
            records, features = load_feature_shards(
                output_dir(config) / "features" / "benchmarks" / benchmark,
                feature_mode,
            )
        else:
            benchmark_directory = (
                _comparison_run_root(config)
                / "comparisons"
                / "haloscope_matched_v1"
                / "features"
                / "benchmarks"
                / benchmark
            )
            if not any(benchmark_directory.glob("shard_*.pt")):
                results[benchmark] = {
                    "status": "unavailable",
                    "reason": f"No same-answer benchmark features in {benchmark_directory}",
                }
                continue
            records, layers = _load_feature_layers(
                benchmark_directory,
                [same_answer_layer],
            )
            features = layers[same_answer_layer]
        scored = np.asarray([record["correct"] >= 0 for record in records])
        labels = np.asarray(
            [max(0, record["correct"]) for record in records], dtype=int
        )[scored]
        risk = -_scores(features[torch.from_numpy(scored)], fit)
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
        "mean": fit["mean"],
        "std": fit["std"],
        "nuisance_basis": fit["nuisance_basis"],
        "nuisance_rank": fit["nuisance_rank"],
        "nuisance_direction_count": fit["nuisance_direction_count"],
        "source_directions": fit["source_directions"],
        "shared_direction": fit["shared_direction"],
        "source_direction_cosine": fit["source_direction_cosine"],
    }


def _analyze_mode(
    config: dict[str, Any],
    settings: dict[str, Any],
    feature_mode: str,
    records: list[dict[str, Any]],
    raw_features: torch.Tensor,
    mode_index: int,
    same_answer_layer: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = _arrays(records)
    train = arrays["split"] == "train"
    validation = arrays["split"] == "validation"
    test = arrays["split"] == "test"
    fit, selection = _select_pattern(
        raw_features, records, train, validation, settings
    )
    mode_result = {
        "same_answer_layer": same_answer_layer,
        "label_counts": {
            split_name: {
                source: {
                    "correct": int(
                        (
                            (arrays["split"] == split_name)
                            & (arrays["source"] == source)
                            & (arrays["correct"] == 1)
                        ).sum()
                    ),
                    "incorrect": int(
                        (
                            (arrays["split"] == split_name)
                            & (arrays["source"] == source)
                            & (arrays["correct"] == 0)
                        ).sum()
                    ),
                }
                for source in SOURCES
            }
            for split_name in ("train", "validation", "test")
        },
        "selection": selection,
        "train_source_direction_cosine": fit["source_direction_cosine"],
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
            same_answer_layer=same_answer_layer,
        ),
    }
    return mode_result, _serializable_fit(fit)


def analyze_adapter_invariant(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    validate_source_run(config)
    target = adapter_invariant_root(config)
    fingerprint = adapter_invariant_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    metrics_path = target / "metrics.json"
    done = target / "_SUCCESS.json"
    if done.exists() and metrics_path.exists() and not force:
        return metrics_path

    results: dict[str, Any] = {
        "method": {
            "target": "correct versus hallucinated within each adapter",
            "shared_pattern": "unit A correctness direction plus unit B correctness direction",
            "adapter_removal": (
                "project out adapter differences estimated within correctness/relation groups"
            ),
            "selection": "synthetic validation only, with adapter-identity leakage constraint",
            "external_benchmark_labels_used_for_selection": False,
        },
        "feature_modes": {},
        **_metadata(config, settings),
    }
    pattern_payload = {"feature_modes": {}, **_metadata(config, settings)}
    mode_specs: list[tuple[str, int | None]] = [
        (feature_mode, None) for feature_mode in configured_feature_keys(config)
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
                same_answer_layer=same_answer_layer,
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
            {"feature_mode": feature_mode, "pattern": serializable, **_metadata(config, settings)},
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
        "# Adapter-invariant correctness pattern",
        "",
        "The pattern is fitted only on synthetic training entities. Nuisance rank is selected "
        "on synthetic validation entities under an adapter-identity leakage constraint. Natural "
        "benchmark labels are evaluation-only.",
        "",
        "| Feature mode | Found | Nuisance rank | A/B direction cosine | Test A AUROC | "
        "Test B AUROC | Identity AUROC | Worst held-out relation AUROC | TriviaQA AUROC | "
        "TruthfulQA AUROC |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in results["feature_modes"].items():
        if result.get("status") == "not_estimable":
            lines.append(
                f"| {mode} | no | NA | NA | NA | NA | NA | NA | NA | NA |"
            )
            continue
        test = result["synthetic_test"]
        relation_worst = min(
            item["heldout_test"]["worst_adapter_auroc"]
            for item in result["leave_one_relation_out"].values()
        )
        trivia = result["benchmarks"].get("trivia_qa", {})
        truthful = result["benchmarks"].get("truthful_qa", {})
        lines.append(
            f"| {mode} | {'yes' if result['selection']['pattern_found'] else 'no'} | "
            f"{result['selection']['chosen_nuisance_rank']} | "
            f"{_fmt(result['train_source_direction_cosine'])} | "
            f"{_fmt(test['by_adapter']['adapter_a']['auroc'])} | "
            f"{_fmt(test['by_adapter']['adapter_b']['auroc'])} | "
            f"{_fmt(test['adapter_identity']['worst_orientation_free_auroc'])} | "
            f"{_fmt(relation_worst)} | {_fmt(trivia.get('auroc'))} | "
            f"{_fmt(truthful.get('auroc'))} |"
        )
    lines.extend(
        [
            "",
            "Same-answer layers can be marked `not_estimable` when the replayed base-model "
            "answers do not contain enough correct examples to define a correctness direction.",
            "",
            "Interpretation safeguards:",
            "",
            "- A/B direction cosine must be positive for a same-orientation correctness pattern.",
            "- Adapter identity AUROC should remain close to 0.5.",
            "- Both per-adapter test AUROCs and held-out-relation AUROCs must exceed chance.",
            "- Natural benchmark results are meaningful only after these synthetic checks pass.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
