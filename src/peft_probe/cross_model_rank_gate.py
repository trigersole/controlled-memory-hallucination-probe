from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import output_dir
from .versioning import artifact_metadata


RANK_GATE_SCHEMA_VERSION = 1


def load_cross_model_rank_gate_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Cross-model rank-gate configuration must be a mapping")
    if int(settings.get("cross_model_rank_gate_schema_version", -1)) != RANK_GATE_SCHEMA_VERSION:
        raise ValueError(
            f"cross_model_rank_gate_schema_version must be {RANK_GATE_SCHEMA_VERSION}"
        )
    if int(settings.get("source_support_version", -1)) != 3:
        raise ValueError("source_support_version must be 3")
    coverages = [float(value) for value in settings.get("coverages", [])]
    if not coverages or any(not 0 < value < 1 for value in coverages):
        raise ValueError("coverages must be a non-empty list with values in (0, 1)")
    if len(set(coverages)) != len(coverages):
        raise ValueError("coverages must not contain duplicates")
    if int(settings.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    int(settings["seed"])
    return settings


def _rank_percentiles(risk: np.ndarray) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    if not len(risk):
        return np.asarray([], dtype=float)
    order = np.argsort(risk, kind="mergesort")
    percentiles = np.empty(len(risk), dtype=float)
    percentiles[order] = (np.arange(len(risk), dtype=float) + 0.5) / len(risk)
    return percentiles


def _coverage_metrics(
    labels_correct: np.ndarray, risk: np.ndarray, coverage: float
) -> dict[str, float]:
    labels = np.asarray(labels_correct, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(labels) != len(risk) or not len(labels):
        raise ValueError("Non-empty labels and risks of equal length are required")
    accepted_count = min(len(labels), max(1, int(np.floor(coverage * len(labels)))))
    order = np.argsort(risk, kind="mergesort")
    accepted = np.zeros(len(labels), dtype=bool)
    accepted[order[:accepted_count]] = True
    rejected = ~accepted
    correct = labels == 1
    error = ~correct
    baseline_accuracy = float(correct.mean())
    baseline_error_rate = float(error.mean())
    selective_accuracy = float(correct[accepted].mean())
    rejected_error_rate = float(error[rejected].mean()) if rejected.any() else float("nan")

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "n": int(len(labels)),
        "target_coverage": float(coverage),
        "coverage": float(accepted.mean()),
        "accepted": int(accepted.sum()),
        "rejected": int(rejected.sum()),
        "baseline_accuracy": baseline_accuracy,
        "selective_accuracy": selective_accuracy,
        "accuracy_lift": ratio(selective_accuracy, baseline_accuracy),
        "baseline_error_rate": baseline_error_rate,
        "rejected_error_rate": rejected_error_rate,
        "error_enrichment": ratio(rejected_error_rate, baseline_error_rate),
        "error_recall": ratio(int(np.sum(rejected & error)), int(error.sum())),
        "correct_acceptance": ratio(int(np.sum(accepted & correct)), int(correct.sum())),
    }


def _cluster_bootstrap_coverages(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    risk: np.ndarray,
    coverages: list[float],
    samples: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    cluster_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        cluster_indices.setdefault(str(row["entity_id"]), []).append(index)
    clusters = [np.asarray(cluster_indices[key], dtype=int) for key in sorted(cluster_indices)]
    rng = np.random.default_rng(seed)
    metric_names = (
        "selective_accuracy",
        "accuracy_lift",
        "error_recall",
        "correct_acceptance",
        "rejected_error_rate",
    )
    draws = {
        str(coverage): {name: [] for name in metric_names} for coverage in coverages
    }
    for _ in range(samples):
        sampled_clusters = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled_clusters])
        for coverage in coverages:
            metrics = _coverage_metrics(labels[indices], risk[indices], coverage)
            for name in metric_names:
                value = metrics[name]
                if np.isfinite(value):
                    draws[str(coverage)][name].append(float(value))
    return {
        coverage: {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else [float("nan"), float("nan")]
            for name, values in metrics.items()
        }
        for coverage, metrics in draws.items()
    }


def _correct_probability(
    report: dict[str, Any], names: list[str], matrix: np.ndarray
) -> np.ndarray:
    fitted = report["fitted_model"]
    requested = fitted["feature_names"]
    missing = sorted(set(requested) - set(names))
    if missing:
        raise RuntimeError(f"Saved model requests unavailable features: {missing}")
    selected = matrix[:, [names.index(name) for name in requested]]
    mean = np.asarray(fitted["mean"], dtype=float)
    std = np.asarray(fitted["std"], dtype=float)
    coefficients = np.asarray(fitted["coefficients"], dtype=float)
    logits = ((selected - mean) / std) @ coefficients + float(fitted["intercept"])
    probability = np.empty_like(logits)
    positive = logits >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    return probability


def _evaluate_dataset(
    rows: list[dict[str, Any]],
    names: list[str],
    matrix: np.ndarray,
    source_report: dict[str, Any],
    settings: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    from .confidence_gate import _ranking_metrics

    labels = np.asarray([row["correct"] for row in rows], dtype=int)
    risk = 1.0 - _correct_probability(source_report, names, matrix)
    coverages = [float(value) for value in settings["coverages"]]
    point = {
        str(coverage): _coverage_metrics(labels, risk, coverage)
        for coverage in coverages
    }
    intervals = _cluster_bootstrap_coverages(
        rows,
        labels,
        risk,
        coverages,
        int(settings["bootstrap_samples"]),
        int(settings["seed"]) + seed_offset,
    )
    for coverage in point:
        point[coverage]["ci95"] = intervals[coverage]
    return {
        "ranking": _ranking_metrics(labels, risk),
        "risk_quantiles": {
            str(quantile): float(np.quantile(risk, quantile))
            for quantile in (0.01, 0.10, 0.50, 0.90, 0.99)
        },
        "rank_percentile_quantiles": {
            str(quantile): float(np.quantile(_rank_percentiles(risk), quantile))
            for quantile in (0.10, 0.50, 0.90)
        },
        "coverage_gates": point,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Label-free rank-normalized support gate",
        "",
        "Coverage levels were fixed before evaluation. Ranking and acceptance use no natural "
        "benchmark labels; labels are used only to calculate the reported metrics.",
        "",
        "| Evaluation | AUROC | Target coverage | Actual coverage | Selective accuracy | "
        "95% CI | Accuracy lift | Error recall | Correct acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in report["evaluations"].items():
        for coverage, metrics in result["coverage_gates"].items():
            interval = metrics["ci95"]["selective_accuracy"]
            lines.append(
                "| {} | {:.4f} | {:.0%} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] | "
                "{:.2f}x | {:.4f} | {:.4f} |".format(
                    dataset,
                    result["ranking"]["auroc"],
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
            "The gate accepts the requested fraction with the lowest frozen-model risk. "
            "Target risk ranks are computed without labels, so this evaluates selective "
            "prediction under calibration shift rather than fitting a benchmark threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_cross_model_rank_gate(
    config: dict[str, Any],
    support_settings: dict[str, Any],
    rank_settings: dict[str, Any],
    force: bool = False,
) -> Path:
    from .cross_model_support import (
        SCHEMA_VERSION,
        _aligned_dataset,
        _feature_table,
        cross_model_support_root,
    )
    from .io_utils import atomic_json

    expected_version = int(rank_settings["source_support_version"])
    if SCHEMA_VERSION != expected_version:
        raise RuntimeError(
            f"Rank gate expects support v{expected_version}, code provides v{SCHEMA_VERSION}"
        )
    source_root = cross_model_support_root(config)
    source_report_path = source_root / "results" / "report.json"
    if not source_report_path.is_file():
        raise FileNotFoundError(f"Missing completed v3 support report: {source_report_path}")
    with source_report_path.open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    if source_report.get("method") != f"cross_model_support_v{expected_version}":
        raise RuntimeError("The saved support report has an incompatible method version")

    target = source_root / f"rank_gate_v{RANK_GATE_SCHEMA_VERSION}"
    report_path = target / "report.json"
    if report_path.exists() and not force:
        return report_path
    target.mkdir(parents=True, exist_ok=True)

    evaluations = {}
    synthetic = _aligned_dataset(source_root, "synthetic")
    synthetic_test = [row for row in synthetic if row["split"] == "test"]
    names, matrix = _feature_table(synthetic_test)
    evaluations["synthetic_test"] = _evaluate_dataset(
        synthetic_test, names, matrix, source_report, rank_settings, 0
    )
    for index, dataset in enumerate(support_settings["datasets"]):
        if dataset == "synthetic":
            continue
        rows = [
            row
            for row in _aligned_dataset(source_root, dataset)
            if int(row["correct"]) >= 0
        ]
        dataset_names, dataset_matrix = _feature_table(rows)
        evaluations[dataset] = _evaluate_dataset(
            rows,
            dataset_names,
            dataset_matrix,
            source_report,
            rank_settings,
            100 + index,
        )
    report = {
        "method": f"cross_model_rank_gate_v{RANK_GATE_SCHEMA_VERSION}",
        "source_method": source_report["method"],
        "coverages_fixed_before_evaluation": rank_settings["coverages"],
        "bootstrap_unit": "entity_id",
        "bootstrap_samples": int(rank_settings["bootstrap_samples"]),
        "evaluations": evaluations,
        **artifact_metadata(),
    }
    atomic_json(report_path, report)
    (target / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report_path
